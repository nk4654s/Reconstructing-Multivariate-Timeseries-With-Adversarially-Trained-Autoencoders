import torch
from torch import nn, tensor
from torch.nn.functional import softplus

from Comprisk.model.certainty import CertaintyBase


class AutoRegressiveCertainty(CertaintyBase):
    def __init__(self, num_hidden: int = 256, num_layers: int = 3, dropout: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._dropout = dropout

        self._lstm = nn.LSTM(
            self._input_dim,
            self._num_hidden,
            num_layers=self._num_layers,
            dropout=dropout,
            batch_first=True,
            device=self.device
        )

        self._fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self._num_hidden, self._num_hidden // 2),
            nn.ReLU(),
            nn.Linear(self._num_hidden // 2, self._num_hidden // 4),
            nn.ReLU(),
            nn.Linear(self._num_hidden // 4, self._output_dim * 2),
        )

    def forward(self, batch: tensor) -> tuple[tensor, tensor]:
        _, (hn, _) = self._lstm(batch)

        x = self._fc(torch.unsqueeze(hn[-1], dim=1))
        mean = x[:, :, :self._output_dim]
        var = x[:, :, self._output_dim:]

        return self._activation(mean), softplus(var)

