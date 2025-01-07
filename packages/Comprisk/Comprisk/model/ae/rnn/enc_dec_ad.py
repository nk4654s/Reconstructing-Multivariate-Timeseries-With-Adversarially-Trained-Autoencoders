"""This submodule contains an AutoEncoder based on the paper [LSTM-based Encoder-Decoder for
Multi-sensor Anomaly Detection](https://arxiv.org/pdf/1607.00148.pdf). The AutoEncoder is called EncDecAD."""
# https://arxiv.org/pdf/1607.00148.pdf
from typing import Tuple, List
import torch
import torch.nn as nn
from Comprisk.model.ae import EncoderBase, DecoderBase, AutoEncoder


class EDAEncoder(EncoderBase):
    """The encoder part which is based on the default encoder base class. The encoder uses a single LSTM
    module by pytorch, hence the configuration possible are:
    Args:
        num_hidden_layers (int): The number of LSTM layers.
        num_hidden_dim (int): The number of nodes within one LSTM layer.
        num_features (int): The number of features which the model receives.
        dropout_prob (float): The probability a neuron is disabled.
    """
    def __init__(self,
                 num_hidden_layers: int,
                 num_hidden_dim: int,
                 num_features: int,
                 dropout_prob: float = 0.0):
        super(EDAEncoder, self).__init__()

        self.input_size = num_features
        self.hidden_size = num_hidden_dim
        self.num_layers = num_hidden_layers

        self.lstm = nn.LSTM(input_size=num_features, hidden_size=num_hidden_dim,
                            num_layers=num_hidden_layers, batch_first=True,
                            dropout=dropout_prob, bias=True, device=self.device)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        _, self.hidden_cell = self.lstm(batch)
        return self.hidden_cell


class EDADecoder(DecoderBase):
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
        super(EDADecoder, self).__init__()

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


class EncDecAD(AutoEncoder):
    """The EncDecAD autoencoder uses a LSTM to find anomalies in a series of multi-sensor data. Read more about it here:
    https://arxiv.org/pdf/1607.00148.pdf.
    """
    def __init__(self,
                 num_hidden_layers: int,
                 num_hidden_dim: int,
                 dropout_prob: float = 0.0,
                 features_in: List[str] = None,
                 features_out: List[str] = None,
                 loss: str = "MSELoss",
                 activation: str = "tanh",
                 optimizer: str = "Adam",
                 lr: float = 0.001,
                 lr_decay: float = 0.999
    ) -> None:
        self._num_hidden_layers = num_hidden_layers
        self._num_hidden_dim = num_hidden_dim
        self._dropout = dropout_prob

        super(EncDecAD, self).__init__(
            features_in=features_in,
            features_out=features_out,
            loss=loss,
            activation=activation,
            optimizer=optimizer,
            lr=lr,
            lr_decay=lr_decay)

        self._input_only = [i for i, name in enumerate(self._features_in) if name not in self._features_out]
        self._output_only = [i for i, name in enumerate(self._features_in) if name in self._features_out]

    def encoder_init(self) -> nn.Module:
        return EDAEncoder(self._num_hidden_layers,
                          self._num_hidden_dim,
                          self._input_dim,
                          self._dropout)

    def decoder_init(self) -> nn.Module:
        return EDADecoder(self._num_hidden_layers,
                          self._num_hidden_dim,
                          self._input_dim,
                          self._output_dim,
                          self._dropout)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        hidden_cell = self._encoder(batch)

        batch_size, sequence, _ = batch.shape
        output = torch.zeros((batch_size, sequence, self._output_dim), device=self.device)

        input_only = batch[:, -1, self._input_only]
        output_only = batch[:, -1, self._output_only]
        input_decoder = torch.cat((input_only, output_only), dim=1)
        input_decoder = torch.unsqueeze(input_decoder, 1)

        for i in range(batch.shape[1] - 1, -1, -1):
            output_decoder, hidden_cell = self._decoder(input_decoder, hidden_cell)

            if i != 0:
                input_only = batch[:, i, self._input_only]
                output_only = batch[:, i, self._output_only]
                input_decoder = torch.cat((input_only, output_only), dim=1)
                input_decoder = torch.unsqueeze(input_decoder, 1)

            output[:, i, :] = output_decoder[:, 0, :]

        return self._activation(output)
