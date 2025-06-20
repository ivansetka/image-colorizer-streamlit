import numpy as np
import torch
from skimage import color
from torchvision.transforms import transforms

from utils.config import model_image_size_map


def rgb2lab(image, model_name):
    grayscale_image = np.array(image) / 255.0
    rgb_image = np.stack([grayscale_image] * 3, axis=2)

    lab_image = color.rgb2lab(rgb_image)
    image = lab_image[:, :, 0]

    if model_name == "Convolutional model":
        return torch.tensor(np.array(image) / 100.0, dtype=torch.float32).unsqueeze(dim=0)

    return torch.tensor(image / 50.0 - 1.0, dtype=torch.float32).unsqueeze(dim=0)


def preprocess(image, model_name):
    image_size = model_image_size_map[model_name]
    image = transforms.Resize(image_size)(image)

    image = rgb2lab(image, model_name)

    return image.unsqueeze(0)
