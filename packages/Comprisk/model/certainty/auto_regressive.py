import torch
from torch import nn, tensor
from torch.nn.functional import softplus

from Comprisk.model.certainty import CertaintyBase


class AutoRegressiveCertainty(CertaintyBase):
    def __init__(self, num_hidden: int = 256, num_layers: int = 1, dropout: float = 0.2, *args, **kwargs):
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

        self._mean = nn.Sequential(
            nn.Linear(self._num_hidden, self._output_dim),
        )

        self._var = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self._num_hidden, self._num_hidden // 2),
            nn.ReLU(),
            nn.Linear(self._num_hidden // 2, self._num_hidden // 4),
            nn.ReLU(),
            nn.Linear(self._num_hidden // 4, self._output_dim),
        )

    def forward(self, batch: tensor) -> tuple[tensor, tensor]:
        _, (x, _) = self._lstm(batch)
        x = torch.unsqueeze(x[-1], dim=1)

        # x = self._fc(torch.unsqueeze(hn[-1], dim=1))
        # mean = x[:, :, :self._output_dim]
        # var = x[:, :, self._output_dim:]

        return self._activation(self._mean(x)), softplus(self._var(x))

