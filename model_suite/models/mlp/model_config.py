import warnings
from ...base_config import BaseModelConfig


class MultiLayerPerceptronConfig(BaseModelConfig):
    architecture_name = "MultiLayerPerceptron"

    def __init__(
        self,
        dropout_rate: float = 0.2,
        input_shape: int = 400,
        output_shape: int = 10,
        hidden_layers: tuple or False = (200,),
        activation: str = "relu",
        final_activation: str = "sigmoid",
        batch_normalization: bool = True,
        **kwargs
    ):
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.final_activation = final_activation
        self.batch_normalization = batch_normalization

        if not hidden_layers:
            warnings.warn(
                "Since there are no hidden layers the activation function will not be used."
            )
