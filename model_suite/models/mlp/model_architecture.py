import torch
from torch import nn
from .model_config import MultiLayerPerceptronConfig
from ...base_model import BaseModel


class MultiLayerPerceptron(BaseModel):
    Config = MultiLayerPerceptronConfig

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create a list to hold the layers of the MLP
        layers = []
        if config.hidden_layers:
            # Add the input layer
            layers.append(nn.Linear(config.input_shape, config.hidden_layers[0]))
            if config.batch_normalization:
                layers.append(nn.BatchNorm1d(config.hidden_layers[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))

            # Add hidden layers
            for i in range(len(config.hidden_layers) - 1):
                layers.append(
                    nn.Linear(config.hidden_layers[i], config.hidden_layers[i + 1])
                )
                if config.batch_normalization:
                    layers.append(nn.BatchNorm1d(config.hidden_layers[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout_rate))

            # Add the output layer
            layers.append(nn.Linear(config.hidden_layers[-1], config.output_shape))
            if config.final_activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif config.final_activation == "softmax":
                layers.append(
                    nn.Softmax(dim=1)
                )  # Make sure to specify the appropriate dimension for softmax
        else:
            layers.append(nn.Linear(config.input_shape, config.output_shape))
            if config.final_activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif config.final_activation == "softmax":
                layers.append(
                    nn.Softmax(dim=1)
                )  # Make sure to specify the appropriate dimension for softmax

        # Combine all layers into the MLP
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
