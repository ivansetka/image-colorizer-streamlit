import math

import torch
from torch import nn

from .diffusion import Diffusion
from .networks import (
    _ResBlock,
    _ResidualNormBlock,
    _MultiheadLinearAttention,
    _DownBlock,
    _UpBlock,
    _MultiheadAttention
)


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, emb_dim):
        super(_SinusoidalPosEmb, self).__init__()
        self.half_emb_dim = emb_dim // 2

    def forward(self, t):
        emb = math.log(10000) / (self.half_emb_dim - 1)
        emb = torch.exp(torch.arange(self.half_emb_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]

        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class _DownUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, is_last=False, dropout=0., double_in=True):
        super(_DownUNetBlock, self).__init__()
        self.block1 = _ResBlock(in_channels, in_channels, time_emb_dim=time_emb_dim)
        self.block2 = _ResBlock(in_channels, in_channels, time_emb_dim=time_emb_dim)
        self.attention = _ResidualNormBlock(in_channels, _MultiheadLinearAttention(in_channels))
        self.down_block = (
            _DownBlock(in_channels * 2 if double_in else in_channels, out_channels, dropout=dropout)
            if not is_last else
            nn.Conv2d(in_channels * 2 if double_in else in_channels, out_channels, 3, padding=1)
        )

    def __iter__(self):
        return iter((self.block1, self.block2, self.attention, self.down_block))


class _UpUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, is_last=False, dropout=0.):
        super(_UpUNetBlock, self).__init__()
        self.block1 = _ResBlock(in_channels + out_channels, in_channels, time_emb_dim=time_emb_dim)
        self.block2 = _ResBlock(in_channels + out_channels, in_channels, time_emb_dim=time_emb_dim)
        self.attention = _ResidualNormBlock(in_channels, _MultiheadLinearAttention(in_channels))
        self.up_block = (
            _UpBlock(in_channels, out_channels, dropout=dropout)
            if not is_last else
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

    def __iter__(self):
        return iter((self.block1, self.block2, self.attention, self.up_block))


class UNetModel(nn.Module):
    def __init__(self, layers, dim=128, dropout=0.):
        super(UNetModel, self).__init__()

        out_channels = layers[0]
        self.init_conv = nn.Conv2d(3, out_channels, 7, padding=3)

        time_emb_dim = dim * 4
        self.time_mlp = nn.Sequential(
            _SinusoidalPosEmb(dim),
            nn.Linear(dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.down = nn.ModuleList()
        for i in range(len(layers) - 1):
            in_channels, out_channels = layers[i], layers[i + 1]

            block = _DownUNetBlock(in_channels, out_channels, time_emb_dim, i == len(layers) - 2, dropout)
            self.down.append(block)

        self.bottleneck1 = _ResBlock(out_channels, out_channels, time_emb_dim)
        self.bottleneck2 = _ResidualNormBlock(out_channels, _MultiheadAttention(out_channels))
        self.bottleneck3 = _ResBlock(out_channels, out_channels, time_emb_dim)

        self.up = nn.ModuleList()
        for i in range(len(layers) - 1, 0, -1):
            in_channels, out_channels = layers[i], layers[i - 1]

            block = _UpUNetBlock(in_channels, out_channels, time_emb_dim, i == 1, dropout)
            self.up.append(block)

        self.final = nn.Sequential(
            _ResBlock(out_channels * 2, out_channels, time_emb_dim),
            nn.Conv2d(out_channels, 2, 1),
        )

    def forward(self, x, t, greyscale_features):
        t = self.time_mlp(t)
        x = self.init_conv(x)
        residual = x.clone()

        down_outputs = []
        for i, (block1, block2, attention, down_block) in enumerate(self.down):
            x = block1(x, t)
            down_outputs.append(x)

            x = block2(x, t)
            x = attention(x)
            down_outputs.append(x)

            x = torch.cat((x, greyscale_features[i]), dim=1)
            x = down_block(x)

        x = self.bottleneck1(x, t)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x, t)

        for block1, block2, attention, up_block in self.up:
            x = torch.cat((x, down_outputs.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, down_outputs.pop()), dim=1)
            x = block2(x, t)

            x = attention(x)
            x = up_block(x)

        x = torch.cat((x, residual), dim=1)
        return self.final(x)

    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path, map_location=self.device))

    def colorize(self, x):
        generated_ab = self(x)
        return torch.cat((x, generated_ab), dim=1)


class GrayscaleEncoder(nn.Module):
    def __init__(self, layers, dropout=0.):
        super(GrayscaleEncoder, self).__init__()

        out_channels = layers[0]
        self.init_conv = nn.Conv2d(1, out_channels, 7, padding=3)

        self.down = nn.ModuleList()
        for i in range(len(layers) - 1):
            in_channels, out_channels = layers[i], layers[i + 1]

            block = _DownUNetBlock(in_channels, out_channels, None, i == len(layers) - 2, dropout, False)
            self.down.append(block)

    def forward(self, x):
        x = self.init_conv(x)

        grayscale_features = []
        for block1, block2, attention, down_block in self.down:
            x = block1(x, None)
            x = block2(x, None)
            x = attention(x)

            grayscale_features.append(x)
            x = down_block(x)

        return grayscale_features


class DiffusionModelWrapper:
    def __init__(self, device='cpu'):
        super(DiffusionModelWrapper, self).__init__()
        self.device = device
        self.diffusion = Diffusion(device=device)
        self.unet_model = UNetModel(layers=(128, 256, 384, 512, 512))
        self.grayscale_encoder = GrayscaleEncoder(layers=(128, 256, 384, 512, 512))

    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location=self.device)
        self.unet_model.load_state_dict(weights["model_state_dict"])
        self.grayscale_encoder.load_state_dict(weights["encoder_state_dict"])

    def colorize(self, x):
        self.unet_model.eval()
        self.grayscale_encoder.eval()
        
        B, _, H, W = x.shape

        with torch.no_grad():
            ab = torch.randn(B, 2, H, W).to(x.device)

            for i in range(self.diffusion.noise_steps - 1, 0, -1):
                t = torch.full(size=(B,), fill_value=i, dtype=torch.long).to(x.device)
                noisy_image = torch.cat((x, ab), dim=1)
                grayscale_features = self.grayscale_encoder(x)

                predicted_noise = self.unet_model(noisy_image, t, grayscale_features)

                beta_t = self.diffusion.beta[t].view(-1, 1, 1, 1)
                alpha_t = self.diffusion.alpha[t].view(-1, 1, 1, 1)
                alpha_hat_t = self.diffusion.alpha_hat[t].view(-1, 1, 1, 1)

                if i > 1:
                    noise = torch.randn_like(ab)
                else:
                    noise = torch.zeros_like(ab)

                ab = (1 / torch.sqrt(alpha_t)) * (
                        ab - (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t) * predicted_noise
                ) + torch.sqrt(beta_t) * noise

        return torch.cat((x, ab), dim=1)
