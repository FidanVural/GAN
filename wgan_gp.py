import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, channels_img, features_d): # features_d used as a layer fixed number for channels. As we go through the network we multiply features_d with the numbers.
        super(Critic, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            # nn.Sigmoid(), # değerlerimizi 0 ile 1 arasına sıkıştırmak istemiyoruz !!!
        )

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

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0),
            self._block(features_g*16, features_g*8, 4, 2, 1),
            self._block(features_g*8, features_g*4, 4, 2, 1),
            self._block(features_g*4, features_g*2, 4, 2, 1),
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), # [-1,, 1]
        )

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

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            # The isinstance() function returns True if the specified object is of the specified type, otherwise False.
            # I mean if the m is the type of nn.Conv2d, nn.ConvTranspose2d and nn.BatchNorm2d then returns True.
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
