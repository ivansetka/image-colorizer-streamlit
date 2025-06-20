import torch
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torch import nn
from torchvision.models.resnet import resnet18, ResNet18_Weights


class _DiscriminatorBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, negative_slope=0.2, normalize=True, dropout=0):
        super(_DiscriminatorBlock, self).__init__()
        self.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1))
        self.append(nn.LeakyReLU(negative_slope, inplace=True))

        if normalize:
            self.insert(1, nn.BatchNorm2d(out_channels))

        if dropout:
            self.append(nn.Dropout(dropout))


class Discriminator(nn.Module):
    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList()

        in_channels = 3
        for index, (out_channels, dropout) in enumerate(layers):
            block = _DiscriminatorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1 if index == len(layers) - 1 else 2,
                normalize=index != 0,
                dropout=dropout
            )
            self.layers.append(block)
            in_channels = out_channels

        self.final = nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.final(x)


class PretrainedGeneratorWrapper:
    def __init__(self, image_size=(256, 256), n_input=1, n_output=2, device='cpu'):
        super(PretrainedGeneratorWrapper, self).__init__()
        self.device = device
        self.generator = DynamicUnet(
            encoder=create_body(resnet18(weights=ResNet18_Weights.DEFAULT), n_in=n_input, cut=-2),
            n_out=n_output,
            img_size=image_size
        ).to(device)

    def __call__(self, x, *args, **kwargs):
        return self.generator(x)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location=self.device)
        self.generator.load_state_dict(weights["generator_state_dict"])

    def colorize(self, x):
        self.generator.eval()
        generated_ab = self(x)

        return torch.cat((x, generated_ab), dim=1)
