import torch
import torch.nn as nn
from .model_config import VisionTransformerConfig
from ...base_model import BaseModel


class VisionTransformer(BaseModel):
    Config = VisionTransformerConfig

    def __init__(self, config: VisionTransformerConfig):
        super(VisionTransformer, self).__init__()
        self.config = config

        # Input embedding layer
        self.embedding = nn.Conv2d(
            3,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_hidden_size,
            dropout=config.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=config.num_layers
        )
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
