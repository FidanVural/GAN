# WGAN with GP

import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from wgan import Critic, Generator, initialize_weights
from utils_gp import gp_gradient_penalty

# HYPERPARAMETERS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
lr = 1e-4 # learning rate in the paper
batch_size = 64
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 50 # paper
features_disc = 64 # because the beginning channel_size is 1024 in the paper. 64*16 = 1024. 16 comes from the discriminator first layer channel multiplying value.
# If you wanna change this value check the disriminator layers channel multiplying values. Go discriminator class.
features_gen = 64

# these values come from the original paper
critic_iterartions = 5
LAMBDA_GP = 10


transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)), # if channels_img equals 1 then this normalization is ok.
    # But if we change the channels_img equals 3 then we can apply the normalization like the below.
    transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)])
])

# DATASET

# MNIST yerine celebrity dataset'ini kullanacağız.
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=False)
# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=False)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Total number of batches: {len(loader)}") # return the total number of batches

gen = Generator(z_dim, channels_img, features_gen).to(device)
critic = Critic(channels_img, features_disc).to(device)
initialize_weights(gen)
initialize_weights(critic)

# OPTIMIZERS
# betas was given in the paper
opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))

# LOSS

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

# SUMMARY WRITER

writer_real = SummaryWriter(f"logs/WGAN/real")
writer_fake = SummaryWriter(f"logs/WGAN/fake")

step = 0

gen.train()
critic.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # CRITIC
        for _ in range(critic_iterartions):
            noise = torch.randn((cur_batch_size, z_dim, 1, 1)).to(device)
            fake = gen(noise)

            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            gp = gp_gradient_penalty(critic, real, fake, device=device)

            # - hariç paper'daki loss'un aynısı. - koymamızın nedeni ise paper'da normalde bu değer maksimize edilmek isteniyor.
            # Ancak bizim optimization algoritmalarımız minimize etmek üzerine.
            # Bu nedenle önüne - koyduk ki minimize edilsin.
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp

            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # GENERATOR ---> min -E[critic(gen(z))] # z means fake_image

        output = critic(fake).reshape(-1)
        loss_gen = -(torch.mean(output))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()


        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] \ "
                f"Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # batch_size = 128 but print the 32 images
                img_frid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_frid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1
