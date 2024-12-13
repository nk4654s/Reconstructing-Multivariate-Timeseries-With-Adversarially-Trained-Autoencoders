"""This module contains auto encoder modules based on LSTM layers."""
import torch
import torch.nn as nn
from Comprisk.model.ae import EncoderBase, DecoderBase


class LSTMEncoder(EncoderBase):
    """A simple Encoder based on an LSTM layer followed by a Linear layer.
    
    Args:
        num_features (int): The number of features input into the model.
            This variant of the encoder expects `sensor` and `weather`
            data (see `self.input_keys`).
        num_latent_dim (int): The number of features output to the decoder.
        num_hidden_dim (int): The number of hidden nodes in the LSTM.
        num_hidden_layers (int): The number of LSTM layers.
        dropout_prob (float): The dropout probability in the LSTM.
    """
    def __init__(
        self,
        num_features: int,
        num_latent_dim: int,
        num_hidden_dim: int,
        num_hidden_layers: int,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self._num_features = num_features
        self._num_latent_dim = num_latent_dim
        self._num_hidden_dim = num_hidden_dim
        self._num_hidden_layers = num_hidden_layers
        self._dropout_prob = dropout_prob

        self._lstm_1 = nn.LSTM(
            num_features,
            num_hidden_dim,
            num_layers=num_hidden_layers,
            dropout=dropout_prob,
            batch_first=True,
        )
        self._linear_1 = nn.Linear(num_hidden_dim, num_latent_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = self._lstm_1(batch)
        x = torch.relu(x)
        x = self._linear_1(x)

        return x


class LSTMDecoder(DecoderBase):
    """A simple Decoder based on an LSTM layer followed by a Linear layer.
    
    Args:
        num_features (int): The number of features output by the model.
        num_latent_dim (int): The number of features input to the decoder.
        num_hidden_dim (int): The number of hidden nodes in the LSTM.
        num_hidden_layers (int): The number of LSTM layers.
        dropout_prob (float): The dropout probability in the LSTM.
        predict_key (str): The key to use for prediction (e.g. 'sensors', 'target').
    """
    def __init__(
        self,
        num_features: int,
        num_latent_dim: int,
        num_hidden_dim: int,
        num_hidden_layers: int,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self._num_features = num_features
        self._num_latent_dim = num_latent_dim
        self._num_hidden_dim = num_hidden_dim
        self._num_hidden_layers = num_hidden_layers
        self._dropout_prob = dropout_prob

        self._lstm_1 = nn.LSTM(
            num_latent_dim,
            num_hidden_dim,
            num_layers=num_hidden_layers,
            dropout=dropout_prob,
            batch_first=True,
        )
        self._linear_1 = nn.Linear(num_hidden_dim, num_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self._lstm_1(z)
        x = torch.relu(x)
        return self._linear_1(x)
