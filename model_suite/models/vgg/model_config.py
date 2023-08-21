import warnings


class VGGConfig:
    def __init__(
        self,
        input_shape=(224, 224, 3),
        num_classes=1000,  # ImageNet, 1000 object classes.
        architecture_type="VGG16",
        batch_normalization=False,
        dropout_rate=0.0,
    ):
        self.architecture_type = architecture_type
        if architecture_type not in ["VGG16", "VGG19"]:
            raise ValueError("Invalid architecture_type. Use 'VGG16' or 'VGG19'.")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate
