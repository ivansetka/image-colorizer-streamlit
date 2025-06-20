import math

import torch
from torch import nn


class _ConvNormActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(_ConvNormActBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if normalize else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2, normalize=True):
        super(_UpBlock, self).__init__()
        out_channels = out_channels * (upscale_factor ** 2)
        self.block = nn.Sequential(
            _ConvNormActBlock(in_channels, out_channels, normalize),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        return self.block(x)


class _BlendConvBlock(nn.Module):
    def __init__(self, in_channels, in_channels_feature, out_channels, normalize=True):
        super(_BlendConvBlock, self).__init__()
        self.up = _UpBlock(in_channels, out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(in_channels_feature)
        self.fuse = _ConvNormActBlock(in_channels_feature + out_channels, out_channels, normalize)

    def forward(self, x, backbone_feature):
        x_up = self.up(x)
        backbone_feature = self.norm(backbone_feature)
        x_cat = torch.cat((x_up, backbone_feature), dim=1)

        return self.fuse(self.relu(x_cat))


class _SinusoidalPosEmb(nn.Module):
    # Copyright (c) Facebook, Inc. and its affiliates.
    # https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class _MultiheadAttention(nn.Module):
    def __init__(self, heads=8, dim_head=32, dropout=0.0, is_self_attention=False):
        super(_MultiheadAttention, self).__init__()
        hidden_dim = dim_head * heads
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout)
        self.is_self_attention = is_self_attention

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, query_pos_emb=None, key_pos_emb=None):
        residual = query

        if self.is_self_attention:
            value = query
            query = key = (query + query_pos_emb if query_pos_emb is not None else query)
        else:
            value = key
            query = query + query_pos_emb if query_pos_emb is not None else query
            key = key + key_pos_emb if key_pos_emb is not None else key

        attention, _ = self.attention(query=query, key=key, value=value)
        attention = self.dropout(attention)

        return self.norm(residual + attention)


class _MLP(nn.Module):
    def __init__(self, dim, hidden_dim=2048, dropout=0.0):
        super(_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.layers(x)
        return self.norm(x)


class _FeaturesPosEmb(nn.Module):
    def __init__(self, features_dim, dim=256):
        super(_FeaturesPosEmb, self).__init__()
        self.pos_emb = _SinusoidalPosEmb(dim // 2, normalize=True)
        self.levels_emb = nn.Parameter(torch.randn(len(features_dim), dim))
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, dim, kernel_size=1) for in_channels in features_dim
        ])

    def forward(self, encoder_features):
        features, pos_embs = [], []
        for encoder_feature, projection, level_emb in zip(encoder_features, self.projections, self.levels_emb):
            # (B, dim, H, W) -> (B, dim, HW) ->  (HW, B, dim)
            feature = projection(encoder_feature).flatten(2) + level_emb.view(1, -1, 1)
            feature = feature.permute(2, 0, 1)

            # (B, dim, H, W) -> (B, dim, HW) ->  (HW, B, dim)
            pos_emb = self.pos_emb(encoder_feature).flatten(2)
            pos_emb = pos_emb.permute(2, 0, 1)

            features.append(feature)
            pos_embs.append(pos_emb)

        return features, pos_embs
