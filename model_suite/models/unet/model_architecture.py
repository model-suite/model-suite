import torch
from torch import nn
from .model_config import UNetConfig
from ...base_model import BaseModel

__all__ = ["UNet"]


class UNet(BaseModel):
    def __init__(self, config: UNetConfig):
        self.config = config
        super().__init__()

        if self.config.spatial_dimensions == 2:
            self.__conv = nn.Conv2d
            self.__max_pool = nn.MaxPool2d
            self.__batch_norm = nn.BatchNorm2d
            self.__conv_transpose = nn.ConvTranspose2d
        elif self.config.spatial_dimensions == 3:
            self.__conv = nn.Conv3d
            self.__max_pool = nn.MaxPool3d
            self.__batch_norm = nn.BatchNorm3d
            self.__conv_transpose = nn.ConvTranspose3d

        self.inc = self._get_double_conv(
            self.config.in_channels, self.config.channels[0]
        )
        self.down1 = self.__get_down(self.config.in_channels, self.config.channels[0])
        self.down2 = self.__get_down(self.config.channels[0], self.config.channels[1])
        self.down3 = self.__get_down(self.config.channels[1], self.config.channels[2])

        self.up1 = self.__get_up(self.config.channels[3], self.config.channels[2])
        self.up2 = self.__get_up(self.config.channels[2], self.config.channels[1])
        self.up3 = self.__get_up(self.config.channels[1], self.config.channels[0])
        self.outc = self.__conv(self.config.channels[0], self.config.out_channels, 1)


    def _get_double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            self.__conv(in_channels, out_channels, self.config.kernel_size),
            self.__batch_norm(out_channels),
            nn.ReLU(inplace=True),
            self.__conv(out_channels, out_channels, self.config.kernel_size),
            self.__batch_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def __get_down(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            self.__max_pool(self.config.strides),
            self._get_double_conv(in_channels, out_channels),
        )

    def __get_up(self, in_channels: int, out_channels: int) -> nn.Sequential:
        if self.config.is_bilinear:
            return nn.Sequential(
                nn.Upsample(scale_factor=self.config.strides, mode="bilinear"),
                self._get_double_conv(in_channels, out_channels),
            )
        else:
            return nn.Sequential(
                self.__conv_transpose(
                    in_channels, out_channels, self.config.strides, self.config.strides
                ),
                self._get_double_conv(in_channels, out_channels),
            )

    def __get_out_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            self.__conv(in_channels, out_channels, 1), nn.Softmax(dim=1)
        )

    def forward(x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = self.up2(x)
        x = self.up3(x)
        logits = self.outc(x)

        if self.config.out_activation == "softmax":
            logits = nn.Softmax(dim=1)(logits)
        elif self.config.out_activation == "sigmoid":
            logits = nn.Sigmoid()(logits)

        return logits
