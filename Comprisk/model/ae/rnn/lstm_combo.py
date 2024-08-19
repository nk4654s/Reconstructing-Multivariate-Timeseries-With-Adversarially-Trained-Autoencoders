"""This module contains auto encoder modules based on LSTM layers."""
from typing import List
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

    @property
    def input_keys(self) -> List[str]:
        return ["sensors", "weather"]

    def forward(self, batch: torch.tensor) -> torch.tensor:
        x, _ = self._lstm_1(batch)
        x = torch.relu(x)

        return self._linear_1(x)


class LSTMDecoder(DecoderBase):
    """A simple Decoder based on an LSTM layer folowed by a Linear layer.
    
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
        predict_key: str = "sensors",
    ) -> None:
        super().__init__()

        self._predict_key = predict_key
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

    def forward(
        self, z: torch.tensor, inputs: torch.Dict[str, torch.tensor]
    ) -> torch.Tensor:
        x, _ = self._lstm_1(z)
        x = torch.relu(x)
        return self._linear_1(x)

    @property
    def output_keys(self) -> List[str]:
        return [self._predict_key]


class LSTMEncoderBypass(LSTMEncoder):
    """This Encoder variant equals the `LSTMEncoder`, but it expects
    only `sensor` data as an input (see `self.input_keys`).
    """
    @property
    def input_keys(self) -> List[str]:
        return ["sensors"]


class LSTMDecoderBypass(LSTMDecoder):
    """This Decoder variant equals the `LSTMDecoder`, but it combines the
    latent representation and `weather` data. Hence, the `num_latent_dim` 
    argument has to be `+1` bigger than what the encoder outputs.

    > For detailed argument description see super class `LSTMDecoder`
    """
    def __init__(
        self,
        num_features: int,
        num_latent_dim: int,
        num_hidden_dim: int,
        num_hidden_layers: int,
        dropout_prob: float = 0,
        predict_key: str = "sensors",
    ) -> None:
        super().__init__(
            num_features,
            num_latent_dim,
            num_hidden_dim,
            num_hidden_layers,
            dropout_prob,
            predict_key
        )
        self._linear_1 = nn.Linear(num_hidden_dim, num_features)

    def forward(
        self, z: torch.tensor, inputs: torch.Dict[str, torch.tensor]
    ) -> torch.Tensor:
        x = torch.cat([z, inputs["weather"]], dim=-1)
        x, _ = self._lstm_1(x)
        x = torch.relu(x)
        return self._linear_1(x)
