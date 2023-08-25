from ...base_config import BaseModelConfig


class AlexNetConfig(BaseModelConfig):
    architecture_name = "AlexNet"

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, **kwargs):
        self.num_classes = num_classes
        self.dropout = dropout
