import os.path

import torch
import config
from torchvision.utils import save_image


def save_images(img_true, target, img_fake, epoch, idx):
    if epoch == 1:
        img_true_path = os.path.join(config.IMAGES_SAVING_PATH, f"input_{epoch}_{idx}.png")
        save_image(img_true, img_true_path)

        target_path = os.path.join(config.IMAGES_SAVING_PATH, f"target_{epoch}_{idx}.png")
        save_image(target, target_path)

    img_fake_path = os.path.join(config.IMAGES_SAVING_PATH, f"fake_{epoch}_{idx}.png")
    save_image(img_fake, img_fake_path)


def save_checkpoint(model, optimizer, file_name = "my_checkpoint.pth.tar"):
    print("SAVING CHECKPOINT !")

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, file_name)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("LOADING CHECKPOINT")

    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint and it will lead to many hours of debugging :/ !!!
    for param in optimizer.param_groups:
        param["lr"] = lr

    return model