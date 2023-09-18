import torch
from torch import nn

from .model_config import VariationalAutoencoderConfig


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        hidden_layers: tuple or False = (200,),
        input_shape: int = 400,
        output_shape: int = 10,
        batch_normalization: bool = True,
        dropout_rate: float = 0.2,
        final_activation: str = "sigmoid",
    ):
        super().__init__()

        # Create a list to hold the layers of the MLP
        layers = []
        if hidden_layers:
            # Add the input layer
            layers.append(nn.Linear(input_shape, hidden_layers[0]))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(hidden_layers[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            # Add hidden layers
            for i in range(len(hidden_layers) - 1):
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                if batch_normalization:
                    layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

            # Add the output layer
            layers.append(nn.Linear(hidden_layers[-1], output_shape))
            if final_activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif final_activation == "softmax":
                layers.append(
                    nn.Softmax(dim=1)
                )  # Make sure to specify the appropriate dimension for softmax
        else:
            layers.append(nn.Linear(input_shape, output_shape))
            if final_activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif final_activation == "softmax":
                layers.append(
                    nn.Softmax(dim=1)
                )  # Make sure to specify the appropriate dimension for softmax

        # Combine all layers into the MLP
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        conv_filters: tuple = (32, 64, 64, 64),
        conv_kernel_size: tuple = (3, 3, 3, 3),
        conv_strides: tuple = (1, 2, 2, 1),
        input_shape: int = 400,
        output_shape: int = 10,
        batch_normalization: bool = True,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        final_activation: str = "sigmoid",
        convolution_dim: int = 2,
        encoder: bool = True,
    ):
        super().__init__()

        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=1),
        }

        convolution_layers = {
            1: nn.Conv1d if encoder else nn.ConvTranspose1d,
            2: nn.Conv2d if encoder else nn.ConvTranspose2d,
            3: nn.Conv3d if encoder else nn.ConvTranspose3d,
        }

        ConvolutionLayer = convolution_layers[convolution_dim]

        batch_normalization_layers = {
            1: nn.BatchNorm1d,
            2: nn.BatchNorm2d,
            3: nn.BatchNorm3d,
        }

        BatchNormalizationLayer = batch_normalization_layers[convolution_dim]

        # Create a list to hold the layers of the MLP
        layers = []
        for conv_layer_idx in range(len(conv_filters)):
            layers.append(
                ConvolutionLayer(
                    in_channels=conv_filters[conv_layer_idx],
                    out_channels=conv_filters[conv_layer_idx],
                    kernel_size=conv_kernel_size[conv_layer_idx],
                    stride=conv_strides[conv_layer_idx],
                )
            )
            if batch_normalization:
                layers.append(BatchNormalizationLayer(conv_filters[conv_layer_idx]))
            layers.append(activations[activation])
            layers.append(nn.Dropout(dropout_rate))


class VariationalAutoencoder(nn.Module):
    Config = VariationalAutoencoderConfig

    def __init__(self, config: VariationalAutoencoderConfig):
        super().__init__()
        self.config = config

        raise NotImplementedError