import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from dataset import ColorizationDataset
# from dataset2 import ColorizationDataset2
from pix2pix import Generator, Discriminator, initialize_weights
import config
from utils import save_images, save_checkpoint, load_checkpoint

from torch.utils.data import DataLoader
from tqdm import tqdm

import onnx_model_saving

print(f"Device: {config.DEVICE}")

torch.backends.cudnn.benchmark = True

"""
It enables benchmark mode in cudnn.
benchmark mode is good whenever your input sizes for your network do not vary. 
This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). 
This usually leads to faster runtime.
But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.
https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2
"""

def plot(g_loss, d_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss, label="Generator")
    plt.plot(d_loss, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=1).to(config.DEVICE)

    initialize_weights(disc)
    initialize_weights(gen)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE_DISC, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GEN, betas=(0.5, 0.999))

    # -------------------------------
    MSE_loss = nn.MSELoss() # mse_loss denemesi !
    # -------------------------------
    # disc içerisine Sigmoid koymadık çünkü burada BCEWiBCEWithLogitsLoss kullandık.
    # Eğer BCELoss kullanılacaksa o zaman disc son katmanına Sigmoid eklenmeli !!!

    L1_loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE_DISC)
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE_GEN)

    train_input_path = "./data/train_black"
    train_output_path = "./data/train_color"
    val_input_path = "./data/test_black"
    val_output_path = "./data/test_color"

    train_input_images = [(train_input_path + "/" + i) for i in os.listdir(train_input_path)]
    train_output_images = [(train_output_path + "/" + i) for i in os.listdir(train_output_path)]

    val_input_images = [(val_input_path + "/" + i) for i in os.listdir(val_input_path)]
    val_output_images = [(val_output_path + "/" + i) for i in os.listdir(val_output_path)]

    train_dataset = ColorizationDataset(root_dir_input_image=train_input_images, root_dir_output_image=train_output_images)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    val_dataset = ColorizationDataset(root_dir_input_image=val_input_images, root_dir_output_image=val_output_images)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    true_image, true_target = next(iter(train_loader))
    print(true_image.shape, true_target.shape)

    """
    plt.imshow(true_image[0].permute(1, 2, 0))
    plt.show()
    plt.imshow(true_target[0].permute(1, 2, 0))
    plt.show()
    """

    d_scaler = torch.cuda.amp.GradScaler()
    g_scaler = torch.cuda.amp.GradScaler()

    decay_epoch = 20

    lambda_func = lambda epoch: 1 - max(0, epoch - decay_epoch) / (config.NUM_EPOCHS - decay_epoch)

    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda_func)
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda_func)

    D_LOSS = []
    G_LOSS = []
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for idx, batch in enumerate(loop):
            x, y = batch
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            # DISCRIMINATOR

            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                d_real = disc(x, y)
                d_fake = disc(x, y_fake.detach())

                d_real_loss = MSE_loss(d_real, torch.ones_like(d_real))
                d_fake_loss = MSE_loss(d_fake, torch.zeros_like(d_fake))
                d_loss = (d_real_loss + d_fake_loss) / 2

            disc.zero_grad()
            d_scaler.scale(d_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # GENERATOR

            with torch.cuda.amp.autocast():
                d_fake = disc(x, y_fake)
                g_fake_loss = MSE_loss(d_fake, torch.ones_like(d_fake))
                L1 = L1_loss(y_fake, y) * config.L1_LAMBDA
                g_loss = g_fake_loss + L1

            gen.zero_grad()
            g_scaler.scale(g_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            D_LOSS.append(d_loss.item())
            G_LOSS.append(g_loss.item())

            if idx % 100 == 0:
                loop.set_postfix(
                    d_real=torch.sigmoid(d_real).mean().item(),
                    d_fake=torch.sigmoid(d_fake).mean().item(),
                    d_loss=d_loss.item(),
                    g_loss=g_loss.item(),
                )

                with torch.no_grad():
                    fake_image = gen(true_image.to(config.DEVICE))

                    img_true = torchvision.utils.make_grid(true_image[:4], nrow=4, normalize=True)
                    target = torchvision.utils.make_grid(true_target[:4], nrow=4, normalize=True)
                    img_fake = torchvision.utils.make_grid(fake_image[:4], nrow=4, normalize=True)

                    save_images(img_true, target, img_fake, epoch, idx)

        lr_scheduler_D.step()
        lr_scheduler_G.step()

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(disc, opt_disc, file_name=config.CHECKPOINT_DISC)
            save_checkpoint(gen, opt_gen, file_name=config.CHECKPOINT_GEN)

        onnx_model_saving.onnx_model(gen)

    plot(G_LOSS, D_LOSS)

# If we work with more than one worker we do this all the time. Otherwise, we get errors.
if __name__ == "__main__":
    main()


