from torch import nn, Tensor
from Comprisk.model.auto_encoder import EncoderBase


class LSTMEncoder(EncoderBase):
    """A simple Encoder based on an LSTM layer followed by a Linear layer.

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
        num_hidden_dim: int,
        num_hidden_layers: int,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self._num_features = num_features
        self._num_hidden_dim = num_hidden_dim
        self._num_hidden_layers = num_hidden_layers
        self._dropout_prob = dropout_prob

        self._lstm = nn.LSTM(
            num_features,
            num_hidden_dim,
            num_layers=num_hidden_layers,
            dropout=dropout_prob,
            batch_first=True,
            device=self.device
        )

    def forward(self, batch: Tensor) -> Tensor:
        # forward pass the batch through the lstm layers
        _, hidden_cell = self._lstm(batch)

        # output the last layer
        return hidden_cell
