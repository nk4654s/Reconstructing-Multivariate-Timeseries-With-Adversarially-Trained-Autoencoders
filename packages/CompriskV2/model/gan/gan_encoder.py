import torch

from Comprisk.model.auto_encoder import EncoderBase


class GANEncoder(EncoderBase):
    # TODO keep improving model
    """A simple Encoder using 2 Dense Layers for now.

    Args:
        num_features (int): The number of features input into the model.
            This variant of the encoder expects `sensor` and `weather`
            data (see `self.input_keys`).
        num_hidden_dim (int): The number of hidden nodes in the LSTM.
        num_hidden_layers (int): The number of LSTM layers.
        dropout_prob (float): The dropout probability in the LSTM.
    """

    def __init__(
        self,
        num_features_in: int,
        weight_init,
    ) -> None:
        super().__init__()

        self._num_features_in = num_features_in
        self._activation = torch.nn.LeakyReLU()
        self._weight_init = weight_init

        # (22 * 60)

        self._conv_layer1 = torch.nn.Conv1d(
            in_channels=self._num_features_in,
            out_channels=self._num_features_in,
            kernel_size=3,
            stride=1,
            padding=1
        )  # (22, 60)

        self._pool_layer1 = torch.nn.AvgPool1d(
            kernel_size=3,
            stride=2,
            padding=1,
        )  # (22, 30)

        self._conv_layer2 = torch.nn.Conv1d(
            in_channels=self._num_features_in,
            out_channels=self._num_features_in,
            kernel_size=3,
            stride=1,
            padding=1
        )  # (22, 30)

        self._pool_layer2 = torch.nn.AvgPool1d(
            kernel_size=3,
            stride=2,
            padding=1,
        )  # (22, 15)

        #########################
        # Weight Initialization #
        #########################
        self._weight_init(self._conv_layer1.weight)
        self._weight_init(self._conv_layer2.weight)


    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output = torch.swapaxes(batch, -2, -1)

        output = self._conv_layer1(output)
        output = self._activation(output)
        output = self._pool_layer1(output)

        output = self._conv_layer2(output)
        output = self._activation(output)
        output = self._pool_layer2(output)

        return output
