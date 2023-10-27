from ...base_config import BaseModelConfig


class ResNetConfig(BaseModelConfig):
    architecture_name = "ResNet"

    def __init__(
        self,
        input_shape: tuple(int, ...) = (224, 224, 3),
        num_classes: int = 1000,
        custom: tuple or bool = False,
        resnet_type: int = 18,
        dropout_rate: float = 0.0,
    ):
        if len(custom[0] != 4):
            raise ValueError("Invalid custom. len(custom[0]) must be 4.")
        if custom != False and type(custom) == tuple:
            self.custom = custom
        if resnet_type not in [18, 34, 50, 101, 152] and custom == False:
            raise ValueError(
                "Invalid resnet_type. Use 18, 34, 50, 101, or 152. Or use custom: tuple = (conv_layers: type = tuple, num_layers: type = int)."
            )
        else:
            # Can use a switch statement here.
            if resnet_type == 18:
                custom = ((2, 2, 2, 2), 18)
            elif resnet_type == 34:
                custom = ((3, 4, 6, 3), 34)
            elif resnet_type == 50:
                custom = ((3, 4, 6, 3), 50)
            elif resnet_type == 101:
                custom = ((3, 4, 23, 3), 101)
            elif resnet_type == 152:
                custom = ((3, 8, 36, 3), 152)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
