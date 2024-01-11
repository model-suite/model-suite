from ...base_config import BaseModelConfig

__all__ = ["UNetConfig"]


class UNetConfig(BaseModelConfig):
    architecture_name = "UNet"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
