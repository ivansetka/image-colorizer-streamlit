import torch
import torch.nn.functional as F
from torch import nn


class _UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Dropout(dropout / 2) if dropout > 0. else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.up(x)


class _DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.):
        super().__init__()
        self.down = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            nn.Conv2d(in_channels * 4, out_channels, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)

        return self.down(x)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        weight = (weight - mean) / torch.sqrt(var + eps)

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        ) if time_emb_dim else None

        self.conv_norm1 = nn.Sequential(
            WeightStandardizedConv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels)
        )
        self.conv_norm2 = nn.Sequential(
            WeightStandardizedConv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels)
        )

        self.activation = nn.SiLU()

        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t=None):
        residual = self.residual(x)
        x = self.conv_norm1(x)

        if t is not None:
            t = self.time_mlp(t)
            t = t[:, :, None, None]

            scale, shift = t.chunk(2, dim=1)
            x = x * (scale + 1.) + shift

        x = self.activation(x)
        x = self.conv_norm2(x)
        x = self.activation(x)

        return x + residual


class _ResidualNormBlock(nn.Module):
    def __init__(self, dim, module):
        super(_ResidualNormBlock, self).__init__()
        self.module = module
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        return x + self.module(self.norm(x))


class _MultiheadLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(_MultiheadLinearAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.InstanceNorm2d(dim, affine=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # (B, 3 * hidden_dim, H, W)
        qkv = self.qkv(x)

        # (B, hidden_dim, H, W) each
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # (B, heads, dim_head, HW)
        q = q.view(B, self.heads, -1, H * W)
        k = k.view(B, self.heads, -1, H * W)
        v = v.view(B, self.heads, -1, H * W)

        q = F.softmax(q, dim=-2)
        k = F.softmax(k, dim=-1)

        q = q * self.scale
        v = v / (H * W)

        # (B, heads, dim_head, dim_head)
        context = k @ v.transpose(-2, -1)

        # (B, heads, dim_head, HW)
        x = context @ q

        # (B, hidden_dim, H, W)
        x = x.view(B, -1, H, W)

        return self.final_conv(x)


class _MultiheadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32):
        super(_MultiheadAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.final_conv = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # (B, 3 * hidden_dim, H, W)
        qkv = self.qkv(x)

        # (B, hidden_dim, H, W) each
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # (B, heads, dim_head, HW)
        q = q.view(B, self.heads, -1, H * W)
        k = k.view(B, self.heads, -1, H * W)
        v = v.view(B, self.heads, -1, H * W)

        # (B, heads, HW, dim_head)
        q = q.transpose(-2, -1) * self.scale

        # (B, heads, HW, HW)
        attention = q @ k
        attention = attention.softmax(dim=-1)

        # (B, heads, HW, dim_head)
        x = attention @ v.transpose(-2, -1)

        # (B, hidden_dim, H, W)
        x = x.transpose(-2, -1).reshape(B, -1, H, W)

        return self.final_conv(x)
