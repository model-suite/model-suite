from ...base_config import BaseModelConfig


class VisionTransformerConfig(BaseModelConfig):
    architecture_name = "ViT"

    def __init__(
        self,
        num_classes: int = 1000,
        patch_size: int = 16,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        mlp_hidden_size: int = 3072,
    ):
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_hidden_size = mlp_hidden_size
