from ...base_config import BaseModelConfig


class AutoencoderConfig(BaseModelConfig):
    architecture_name = "Autoencoder"

    def __init__(
        self,
        encoder_decoder_type: str = "mlp",
        encoder_conv_filters: tuple = (32, 64, 64, 64),
        encoder_conv_kernel_size: tuple = (3, 3, 3, 3),
        encoder_conv_strides: tuple = (1, 2, 2, 1),
        decoder_conv_t_filters: tuple = (64, 64, 32, 1),
        decoder_conv_t_kernel_size: tuple = (3, 3, 3, 3),
        decoder_conv_t_strides: tuple = (1, 2, 2, 1),
        # this mlp configuration is almost the same as the mlp model
        encoder_mlp_hidden_layers: tuple = (512, 256),
        encoder_mlp_dropout_rate: float = 0.2,
        encoder_mlp_activation: str = "relu",
        encoder_mlp_final_activation: str = "relu",
        encoder_mlp_batch_normalization: bool = True,
        decoder_mlp_hidden_layers: tuple = (256, 512),
        decoder_mlp_dropout_rate: float = 0.2,
        decoder_mlp_activation: str = "relu",
        decoder_mlp_final_activation: str = "sigmoid",
        decoder_mlp_batch_normalization: bool = True,
        encoder_mlp_input_shape: int = 400,
        latent_dim: int = 10,
    ):
        self.encoder_decoder_type = encoder_decoder_type
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.encoder_mlp_hidden_layers = encoder_mlp_hidden_layers
        self.encoder_mlp_dropout_rate = encoder_mlp_dropout_rate
        self.encoder_mlp_activation = encoder_mlp_activation
        self.encoder_mlp_final_activation = encoder_mlp_final_activation
        self.encoder_mlp_batch_normalization = encoder_mlp_batch_normalization
        self.decoder_mlp_hidden_layers = decoder_mlp_hidden_layers
        self.decoder_mlp_dropout_rate = decoder_mlp_dropout_rate
        self.decoder_mlp_activation = decoder_mlp_activation
        self.decoder_mlp_final_activation = decoder_mlp_final_activation
        self.decoder_mlp_batch_normalization = decoder_mlp_batch_normalization
        self.encoder_mlp_input_shape = encoder_mlp_input_shape
        self.latent_dim = latent_dim
