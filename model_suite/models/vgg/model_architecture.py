import torch
import torch.nn as nn
from .model_config import VGGConfig
from ...base_model import BaseModel


# I believe VGG doesn't initially include BatchNorm or Dropout but here they are.
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers, batch_norm=True):
        super(VGGBlock, self).__init__()

        layers = []
        for _ in range(num_conv_layers):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG(BaseModel):
    Config = VGGConfig

    def __init__(self, config):
        super(VGG, self).__init__()

        self.config = config

        self.features = nn.Sequential(
            VGGBlock(3, 64, 2, self.config.batch_normalization),
            VGGBlock(64, 128, 2, self.config.batch_normalization),
            VGGBlock(128, 256, 3, self.config.batch_normalization),
            VGGBlock(256, 512, 3, self.config.batch_normalization),
            VGGBlock(512, 512, 3, self.config.batch_normalization),
        )
        if self.config.architecture_type == "VGG19":
            self.features.add_module(
                "extra_vgg16_block",
                VGGBlock(512, 512, 3, self.config.batch_normalization),
            )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(4096, config.num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
