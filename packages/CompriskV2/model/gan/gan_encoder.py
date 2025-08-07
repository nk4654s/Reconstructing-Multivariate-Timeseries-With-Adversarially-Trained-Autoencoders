import torch

from CompriskV2.model.auto_encoder import EncoderBase


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
        num_features: int,
        ) -> None:
        super().__init__()

        self._num_features = num_features
        self._activation = torch.nn.LeakyReLU()

        # (22 * 60)

        self._conv_layer1 = torch.nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features * 2,
            kernel_size=3,
            stride=1,
            padding=1
        )  # (44, 60)

        self._pool_layer1 = torch.nn.AvgPool1d(kernel_size=16, stride=1)  # (44, 45)

        self._conv_layer2 = torch.nn.Conv1d(
            in_channels=num_features * 2,
            out_channels=num_features * 4,
            kernel_size=3,
            stride=1,
            padding=1
        )  # (88, 45)

        self._pool_layer2 = torch.nn.AvgPool1d(kernel_size=16, stride=1)  # (88, 30)

        self._conv_layer3 = torch.nn.Conv1d(
            in_channels=num_features * 4,
            out_channels=num_features * 8,
            kernel_size=3,
            stride=1,
            padding=1
        )  # (176, 30)

        self._pool_layer3 = torch.nn.AvgPool1d(kernel_size=16, stride=1)  # (176, 15)

        """self._conv_layer4 = torch.nn.Conv1d(
            in_channels=num_features * 8,
            out_channels=num_features * 16,
            kernel_size=3,
            stride=1,
            padding=1
        )  # (352, 15)

        self._pool_layer4 = torch.nn.AvgPool1d(kernel_size=15, stride=1)  # (352, 1)
"""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output = torch.swapaxes(batch, -2, -1)

        output = self._conv_layer1(output)
        output = self._activation(output)
        output = self._pool_layer1(output)

        output = self._conv_layer2(output)
        output = self._activation(output)
        output = self._pool_layer2(output)

        output = self._conv_layer3(output)
        output = self._activation(output)
        output = self._pool_layer3(output)

        #output = self._conv_layer4(output)
        #output = self._activation(output)
        #output = self._pool_layer4(output)

        return output
