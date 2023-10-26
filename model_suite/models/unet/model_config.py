from ...base_config import BaseModelConfig

__all__ = ["UNetConfig"]


class UNetConfig(BaseModelConfig):
    architecture_name = "UNet"

    def __init__(
        self,
        spatial_dimensions: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        channels=(64, 128, 256, 512, 1024),
        kernel_size: int = 3,
        strides=(2, 2, 2, 2),
        out_activation: str = "softmax",
        is_bilinear: bool = True,
    ):
        self.spatial_dimensions = spatial_dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.out_activation = out_activation
        self.is_bilinear = is_bilinear
