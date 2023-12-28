import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from dcgan import Discriminator, Generator, initialize_weights

import matplotlib.pyplot as plt

# Set random seed for reproducibility
manualSeed = 500
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# HYPERPARAMETERS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
lr = 2e-4 # 0.0002 --> a learning rate of 0.0002 works well on DCGAN
batch_size = 128
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 50
features_disc = 64 # because the beginning channel_size is 1024 in the paper. 64*16 = 1024. 16 comes from the discriminator first layer channel multiplying value.
# If you wanna change this value check the disriminator layers channel multiplying values. Go discriminator class.
features_gen = 64


def show_loss(loss_gen, loss_disc):
    plt.plot(loss_disc, label="discriminator_loss")
    plt.plot(loss_gen, label="generator_loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def show_fake_images(images):
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(images, (1, 2, 0)))
    plt.show()


transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)), # if channels_img equals 1 then this normalization is ok.
    # But if we change the channels_img equals 3 then we can apply the normalization like the below.
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.5,), (0.5,)),
])

# DATASET

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=False)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Total number of batches: {len(loader)}") # return the total number of batches

gen = Generator(z_dim, channels_img, features_gen).to(device)
disc = Discriminator(channels_img, features_disc).to(device)
initialize_weights(gen)
initialize_weights(disc)

# OPTIMIZERS

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# LOSS

criterion = nn.BCELoss()
# If you use the nn.BCEWithLogitsLoss() then you don't need to end Sigmoid() to the final layer.
# Because nn.BCEWithLogitsLoss() includes Sigmoid().

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

# SUMMARY WRITER

writer_real = SummaryWriter(f"logs/DCGAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/DCGAN_MNIST/fake")

step = 0

gen.train()
disc.train()

LOSS_GEN = []
LOSS_DISC = []

images = []

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        noise = torch.randn((cur_batch_size, z_dim, 1, 1)).to(device)

        fake = gen(noise)
        # TRAIN DISCRIMINATOR --> max[log(D(real)) + log(1 - D(G(z))]
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_d = (loss_disc_real + loss_disc_fake)/2

        disc.zero_grad()

        loss_d.backward(retain_graph=True)

        opt_disc.step()

        # TRAIN GENERATOR --> min[log(1 - D(G(z))]
        # Normalde generator için olan loss fonksiyonu bu ancak biz bunun yerine şunu kullanırız.
        # max[log(D(G(z)))] --> aynı şeyi ifade eder.
        # BCE loss'un önündeki eksiden dolayı da yine aslında min etmeye çalışmış oluruz.

        output = disc(fake).reshape(-1) # formula
        loss_g = criterion(output, torch.ones_like(output))

        gen.zero_grad()

        loss_g.backward()

        opt_gen.step()

        LOSS_DISC.append(loss_d.item())
        LOSS_GEN.append(loss_g.item())

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] \ "
                f"Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise) # fixed_noise for visualization

                if (epoch == num_epochs-1):
                    images.append(vutils.make_grid(fake, padding=2, normalize=True))

                # batch_size = 128 but print the 32 images
                img_frid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_frid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1


show_loss(LOSS_GEN, LOSS_DISC)
show_fake_images(images)


