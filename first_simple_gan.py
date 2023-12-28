import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(), # we used sigmoid because discriminator decides whether image is fake or real
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, img_dim),
            nn.Tanh(), # we use tanh to make sure that output value range between -1 and 1
            # we normalize the input between -1 and 1 so we want to get output between -1 and 1
        )

    def forward(self, x):
        return self.gen(x)


# HYPERPARAMETERS
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
lr = 3e-4
z_dim = 128 # 64, 128, 256
image_dim = 784 # 28 x 28 x 1 # because of the mnist
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# TRANSFORMS
transforms = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])

# DATASET
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=False) # Bir kere veriyi indirdikten sonra burayı False yaparız.
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Total number of batches: {len(loader)}") # return the total number of batches

# OPTIMIZER
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# LOSS
criterion = nn.BCELoss()

# This is for the tensorboard
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")

step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        # print(real.shape) # torch.Size([32, 1, 28, 28])
        real = real.view(-1, 784).to(device)
        cur_batch_size = real.shape[0]

        # TRAIN DISCRIMINATOR --> max[log(D(real)) + log(1 - D(G(z))]
        noise = torch.randn(cur_batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)

        # Normalde maximize ederiz. Ancak BCE loss'un kendinden bir "-"si işareti var önünde bu nedenle minimize etmek isteyeceğiz.
        # Ki zaten loss'larda da genelde minimize etmek isteriz. "-" bu nedenle var.
        # BCELoss'un kendi sitesine bir bak !!!
        # criterion içerisine ilk yazılan x_n ve ikinci yazılan y_n.
        # Yukarıdaki formülün ilk kısmını ve ikinci kısmını elde edebilmek için BCELoss'un y değerine birinde torch.ones diğerinde torch.zeros koyduk.

        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        # print("Disc(fake): ", disc(fake).shape) # Disc(fake):  torch.Size([32, 1])
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2

        disc.zero_grad()

        lossD.backward(retain_graph=True)
        # after loss.backward() (when gradients are computed), we need to use optimizer.step() to proceed gradient descent.
        opt_disc.step()

        # TRAIN GENERATOR --> min[log(1 - D(G(z))]
        # Normalde generator için olan loss fonksiyonu bu ancak biz bunun yerine şunu kullanırız.
        # max[log(D(G(z)))] --> aynı şeyi ifade eder.
        # BCE loss'un önündeki eksiden dolayı da yine aslında min etmeye çalışmış oluruz.

        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        gen.zero_grad()

        lossG.backward()

        opt_gen.step()

        ###
        # We're writing this part to see the image results of every epoch on the tensorboard
        ###

        if batch_idx == 0: # Her bir epoch'ta batch_size == 0 olduğunda tensorboard!da görelim.

            print(
                f"Epoch [{epoch}/{num_epochs}] \ "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                # print(real.shape) # torch.Size([32, 784]) bu size'ı görselleştirebilmek için reshape etmemeiz gerekiyor.
                data = real.reshape(-1, 1, 28, 28)
                img_frid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_frid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1

writer_real.close()
writer_fake.close()




