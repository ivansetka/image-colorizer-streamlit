import torch
from torch import nn

from .networks import (
    LowLevelFeaturesNetwork,
    MidLevelFeaturesNetwork,
    GlobalFeaturesNetwork,
    FusionBlock,
    ColorizationNetwork
)


class CNNModel(nn.Module):
    def __init__(self, device='cpu'):
        super(CNNModel, self).__init__()
        self.device = device
        self.low_net = LowLevelFeaturesNetwork()
        self.mid_net = MidLevelFeaturesNetwork()
        self.global_net = GlobalFeaturesNetwork()
        self.fusion_block = FusionBlock()
        self.colorization_net = ColorizationNetwork()

    def forward(self, x):
        x_low = self.low_net(x)
        x_mid = self.mid_net(x_low)
        x_global, _ = self.global_net(x_low)
        x_fused = self.fusion_block(x_mid, x_global)

        return self.colorization_net(x_fused)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location=self.device)
        self.load_state_dict(weights["model_state_dict"])

    def colorize(self, x):
        self.eval()
        generated_ab = self(x)

        return torch.cat((x, generated_ab), dim=1)
