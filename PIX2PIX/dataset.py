import numpy as np
import os
from torch.utils.data import Dataset
import config
from skimage import io
from PIL import Image


class ColorizationDataset(Dataset):
    def __init__(self, root_dir_input_image, root_dir_output_image):

        self.root_dir_input = root_dir_input_image
        self.root_dir_output = root_dir_output_image

    def __getitem__(self, idx):
        inp_path = self.root_dir_input[idx]
        out_path = self.root_dir_output[idx]

        # print(inp_path, out_path)
        img_i = np.array(Image.open(inp_path).convert("L"))
        img_i = img_i[..., None]
        img_o = np.array(Image.open(out_path).convert("RGB"))

        augmentations = config.both_transforms(image=img_i, image0=img_o)

        a_input_image, a_output_image = augmentations["image"], augmentations["image0"]

        a_input_image = config.transforms_for_only_input(image=a_input_image)["image"]
        a_output_image = config.transforms_for_only_output(image=a_output_image)["image"]

        return a_input_image, a_output_image

    def __len__(self):
        return len(self.root_dir_input)



