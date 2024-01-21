import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from pix2pix import Generator
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

IMAGE_SIZE = 256

INP_PATH = "./test/input.jpg"
OUT_PATH = "./test/output.jpg"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CHECKPOINT_GEN = "./models/gen_model.pth.tar"


def colorize_image(inp_img):
    transforms = A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    img = np.array(Image.open(inp_img).convert("L"))
    img = img[..., None]
    height, width, _ = img.shape
    image = transforms(image=img)["image"]
    print(image.shape)
    image = image.unsqueeze(0)


    model_gen = Generator(in_channels=1).to(DEVICE)
    model_gen.load_state_dict(torch.load(CHECKPOINT_GEN, map_location=DEVICE)["state_dict"], strict=False)
    model_gen.eval()

    output_image = model_gen(image.to(DEVICE))
    output_image = F.interpolate(output_image, size=(height, width), mode="bicubic", align_corners=False)
    output_image = torchvision.utils.make_grid(output_image, normalize=True)
    save_image(output_image, OUT_PATH)

colorize_image(INP_PATH)


