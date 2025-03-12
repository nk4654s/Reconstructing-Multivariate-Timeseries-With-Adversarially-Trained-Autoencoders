"""This module contains auto encoder modules based on LSTM layers."""

from typing import Tuple

import torch
from torch import nn

from Comprisk.model.auto_encoder import DecoderBase


class DecoderFewPass(DecoderBase):
    """The decoder part which is based on the default decoder base class. The decoder uses a single LSTM
    module and a simple linear layer by pytorch, hence the configuration possible are:
    Args:
        num_hidden_layers (int): The number of LSTM layers.
        num_hidden_dim (int): The number of nodes within one LSTM layer.
        dropout_prob (float): The probability a neuron is disabled.
        num_features_in (int): The number of features which the model receives. This should equal the encoder part.
        num_features_out (int): The number of features which the model outputs.
    """
    def __init__(self,
                 num_hidden_layers: int,
                 num_hidden_dim: int,
                 num_features_in: int,
                 num_features_out: int,
                 dropout_prob: float = 0.0):
        super().__init__()

        self.input_size = num_features_in
        self.output_size = num_features_out
        self.hidden_size = num_hidden_dim
        self.num_layers = num_hidden_layers

        self.lstm = nn.LSTM(input_size=num_features_in, hidden_size=num_hidden_dim,
                            num_layers=num_hidden_layers, batch_first=True,
                            dropout=dropout_prob, bias=True, device=self.device)
        self.linear = nn.Linear(in_features=num_hidden_dim, out_features=num_features_out, device=self.device)

    def forward(self, batch: torch.Tensor, hidden_cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, hidden_cell = self.lstm(batch, hidden_cell)
        output = self.linear(output)
        return output, hidden_cell
