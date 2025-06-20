from models.cnn.model import CNNModel
from models.diffusion.model import DiffusionModelWrapper
from models.gan.model import PretrainedGeneratorWrapper
from models.transformer.model import TransformerModel


model_map = {
    "Convolutional model": ("cnn.pth", CNNModel),
    "GAN model": ("gan.pth", PretrainedGeneratorWrapper),
    "Diffusion-based model": ("diffusion.pth", DiffusionModelWrapper),
    "Transformer-based model": ("transformer.pth", TransformerModel)
}

model_image_size_map = {
    "Convolutional model": (224, 224),
    "GAN model": (256, 256),
    "Diffusion-based model": (128, 128),
    "Transformer-based model": (256, 256)
}
