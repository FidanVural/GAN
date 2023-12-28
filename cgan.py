import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size): # features_d used as a layer fixed number for channels. As we go through the network we multiply features_d with the numbers.
        super(Critic, self).__init__()

        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img + 1, features_d, kernel_size=4, stride=2, padding=1
            ), # burada channels_img + 1 yapıyoruz aşağıda embedding'i ekstra channel olarak incelediğimiz için !!!
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            # nn.Sigmoid(), # değerlerimizi 0 ile 1 arasına sıkıştırmak istemiyoruz !!!
        )

        self.embed = nn.Embedding(num_classes, img_size*img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False, # we set the bias=False because we use BatchNorm.
            ),
            # ---------------------------------------------------------------------------------------------------------------------------------
            # In WGAN with gradient penalty paper, they use InstanceNorm instead of BatchNorm. They are quite similar.
            # The difference between InstanceNorm and BatchNorm is that while BatchNorm computes the one std and one mean for the whole batch,
            # InstanceNorm calculates N (batch size) std and mean for every image.
            # ---------------------------------------------------------------------------------------------------------------------------------
            nn.InstanceNorm2d(out_channels, affine=True), # affine=True --> this module has learnable parameters.
            nn.LeakyReLU(0.2), # we used 0.2 for the slope because of following the implementation of dcgan paper.
        )

    def forward(self, x, labels):
        """
        print("\n----------------------------------------------------------------------------------")
        print(self.embed)
        print("----------------------------------------------------------------------------------\n")
        print(labels)
        print(labels.shape)
        print("----------------------------------------------------------------------------------")
        """
        # Burada embedding'i image'e ekstra bir kanal olarak ekleyebilmek için embedding'i yeniden boyutlandırıyoruz.
        # N X C X H X W
        # Burada olan şey labels değişkeni 64 (batch_size) tane image'ın label'larını içerir.
        # Biz de her resme ait olan label'i o resme ekleyebilmek için label'ları resimler ile aynı boyuta getiririz. Her bir label boyutu C X H x W olur ki bu case'de 1 X 64 X 64
        # Böylece her resme o resme ait olan label bilgisini eklemiş oluruz.

        embed = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size) # channel sayısı 1 çünkü image'larımızın da channel sayısı 1.

        """
        print("\n")
        print(embed)
        print("\n")
        """

        x = torch.cat([x, embed], dim=1) # N X C X IMG_SIZE(H) X IMG_SIZE(W)
        # print(x.shape)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()

        self.img_size = img_size
        
        self.gen = nn.Sequential(
            self._block(z_dim + embed_size, features_g*16, 4, 1, 0), # !!!
            self._block(features_g*16, features_g*8, 4, 2, 1),
            self._block(features_g*8, features_g*4, 4, 2, 1),
            self._block(features_g*4, features_g*2, 4, 2, 1),
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), # [-1, 1]
        )

        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), # paper implementation
        )

    def forward(self, x, labels):
        # latent vector z : N x noise_dim x 1 x 1
        embed = self.embed(labels).unsqueeze(2).unsqueeze(3)
        """
        print(embed.shape) # torch.Size([64, 100, 1, 1])
        print("----------------------------------")
        """
        x = torch.cat([x, embed], dim=1)
        """
        print(x.shape) # torch.Size([64, 200, 1, 1])
        print("----------------------------------")
        """
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            # The isinstance() function returns True if the specified object is of the specified type, otherwise False.
            # I mean if the m is the type of nn.Conv2d, nn.ConvTranspose2d and nn.BatchNorm2d then returns True.
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
