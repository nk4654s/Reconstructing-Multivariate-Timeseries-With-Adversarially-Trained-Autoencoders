"""This submodule contains an AutoEncoder based on the paper [LSTM-based Encoder-Decoder for
Multi-sensor Anomaly Detection](https://arxiv.org/pdf/1607.00148.pdf). The AutoEncoder is called EncDecAD."""

# https://arxiv.org/pdf/1607.00148.pdf
import torch
import torch.nn as nn

from Comprisk.model.auto_encoder import AutoEncoder
from Comprisk.model.auto_encoder.rnn.decoder import DecoderFewPass
from Comprisk.model.auto_encoder.rnn.encoder import LSTMEncoder


class MultimodalRecurrentAutoEncoder(AutoEncoder):
    """The EncDecAD autoencoder uses a LSTM to find anomalies in a series of multi-sensor data. Read more about it here:
    https://arxiv.org/pdf/1607.00148.pdf.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        num_hidden_dim: int,
        dropout_prob: float = 0.0,
        *args,
        **kwargs
    ) -> None:
        self._num_hidden_layers = num_hidden_layers
        self._num_hidden_dim = num_hidden_dim
        self._dropout = dropout_prob

        self._encoder_class = LSTMEncoder
        self._decoder_class = DecoderFewPass
        self._decoder_args = {
            "num_hidden_layers": self._num_hidden_layers,
            "num_hidden_dim": self._num_hidden_dim,
            "dropout_prob": self._dropout,
        }

        super(MultimodalRecurrentAutoEncoder, self).__init__(*args, **kwargs)

        self._input_only = [
            i
            for i, name in enumerate(self._features_in)
            if name not in self._features_out
        ]
        self._output_only = [
            i for i, name in enumerate(self._features_in) if name in self._features_out
        ]

    def encoder_init(self) -> nn.Module:
        return self._encoder_class(
            num_features=self._input_dim,
            num_hidden_dim=self._num_hidden_dim,
            num_hidden_layers=self._num_hidden_layers,
            dropout_prob=self._dropout,
        )

    def decoder_init(self) -> nn.Module:
        return self._decoder_class(
            num_features_in=self._input_dim,
            num_features_out=self._output_dim,
            **self._decoder_args,
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        hidden_cell = self._encoder(batch)

        batch_size, sequence, _ = batch.shape
        output = torch.zeros(
            (batch_size, sequence, self._output_dim), device=self.device
        )

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
