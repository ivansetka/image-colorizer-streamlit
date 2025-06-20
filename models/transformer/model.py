import torch
import torch.nn as nn
from timm import create_model

from .networks import (
    _UpBlock,
    _BlendConvBlock,
    _MultiheadAttention,
    _MLP,
    _FeaturesPosEmb
)


class Encoder(torch.nn.Module):
    def __init__(self, features_dim, backbone_name='convnext_tiny'):
        super(Encoder, self).__init__()
        self.backbone = create_model(backbone_name, pretrained=True, features_only=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        backbone_features_dim = [info['num_chs'] for info in self.backbone.feature_info]

        self.blend_blocks = nn.ModuleList()
        in_channels = backbone_features_dim.pop()

        for out_channels, in_channels_feature in zip(features_dim, reversed(backbone_features_dim)):
            self.blend_blocks.append(
                _BlendConvBlock(in_channels, in_channels_feature, out_channels)
            )
            in_channels = out_channels

        self.out = _UpBlock(in_channels, features_dim[-1], upscale_factor=4, normalize=False)

    def forward(self, x):
        backbone_features = self.backbone(x)

        x = backbone_features.pop()
        encoder_features = []

        for blend_block in self.blend_blocks:
            x = blend_block(x, backbone_features.pop())
            encoder_features.append(x)

        return encoder_features, self.out(x)


class ColorizationModule(nn.Module):
    def __init__(self, features_dim, dim=256, num_queries=100, heads=8, hidden_dim_ffn=2048, module_layers=9):
        super(ColorizationModule, self).__init__()
        self.module_layers = module_layers
        self.num_features = len(features_dim)

        self.features_pos_emb = _FeaturesPosEmb(features_dim, dim=dim)

        self.color_query = nn.Parameter(torch.randn(num_queries, dim))
        self.query_emb = nn.Parameter(torch.randn(num_queries, dim))

        self.cross_attention_layers = nn.ModuleList()
        self.self_attention_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

        for _ in range(module_layers):
            self.cross_attention_layers.append(
                _MultiheadAttention(heads=heads, dim_head=dim // heads, is_self_attention=False)
            )
            self.self_attention_layers.append(
                _MultiheadAttention(heads=heads, dim_head=dim // heads, is_self_attention=True)
            )
            self.mlp_layers.append(
                _MLP(dim=dim, hidden_dim=hidden_dim_ffn)
            )

        self.norm = nn.LayerNorm(dim)
        self.final = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, encoder_features):
        features, pos_emb = self.features_pos_emb(encoder_features)

        # (Q, C) - > (Q, B, C)
        _, B, _ = features[0].shape
        query_emb = self.query_emb.unsqueeze(1).expand(-1, B, -1)
        color_query = self.color_query.unsqueeze(1).expand(-1, B, -1)

        for i in range(self.module_layers):
            cross_attention = self.cross_attention_layers[i]
            self_attention = self.self_attention_layers[i]
            mlp = self.mlp_layers[i]

            color_query = cross_attention(
                query=color_query,
                key=features[i % self.num_features],
                query_pos_emb=query_emb,
                key_pos_emb=pos_emb[i % self.num_features]
            )
            color_query = self_attention(
                query=color_query,
                query_pos_emb=query_emb
            )
            color_query = mlp(color_query)

        # (Q, B, C) -> (B, Q, C)
        color_query = self.norm(color_query).transpose(0, 1)
        color_query = self.final(color_query)

        return torch.einsum("BQC, BCHW -> BQHW", color_query, x)


class TransformerModel(nn.Module):
    def __init__(
            self,
            features_dim=(512, 512, 256, 256),
            backbone_name='convnext_tiny',
            dim=256,
            num_queries=100,
            heads=8,
            hidden_dim_ffn=2048,
            module_layers=9,
            device='cpu'
    ):
        super(TransformerModel, self).__init__()
        self.device = device
        self.encoder = Encoder(
            features_dim,
            backbone_name
        )
        self.color_module = ColorizationModule(
            features_dim[:-1],
            dim,
            num_queries,
            heads,
            hidden_dim_ffn,
            module_layers
        )

        self.final = nn.Conv2d(num_queries + 3, 2, kernel_size=1)

    def forward(self, x):
        x_cat = torch.cat((x, x, x), dim=1)

        encoder_features, out = self.encoder(x_cat)
        out = self.color_module(out, encoder_features)
        out = torch.cat((out, x_cat), dim=1)

        return self.final(out)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location=self.device)
        self.load_state_dict(weights["model_state_dict"])

    def colorize(self, x):
        self.eval()
        generated_ab = self(x)

        return torch.cat((x, generated_ab), dim=1)
