import numpy as np
from skimage import color


def lab2rgb(image, model_name):
    if model_name == "Convolutional model":
        image[:, :, :1] = image[:, :, :1] * 100.0
        image[:, :, 1:] = image[:, :, 1:] * 255.0 - 128.0
    else:
        image[:, :, :1] = (image[:, :, :1] + 1.0) * 50.0
        image[:, :, 1:] = image[:, :, 1:] * 128.0

    image = image.astype(np.float64)
    return color.lab2rgb(image)


def postprocess(image, model_name):
    image = image.squeeze(0)
    image = image.cpu().permute(1, 2, 0).numpy()
    image = lab2rgb(image, model_name)

    return (image * 255).astype(np.uint8)
