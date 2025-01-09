import torch
import torch.nn as nn
from Comprisk.model.base import BaseModel


class FSLSTMModel(BaseModel):
    def __init__(
        self,
        num_features: int = 22,
        num_targets: int = 22,
        num_hidden_dim: int = 32,
        num_hidden_layers: int = 2,
        dropout_prob: float = 0.3,
        loss: str = "MSELoss",
        activation: str = "sigmoid",
        optimizer: str = "Adam",
        lr: float = 0.001,
        lr_decay: float = 0.99,
    ) -> None:
        super().__init__(loss, activation, optimizer, lr, lr_decay)

        self._num_features = num_features
        self._num_targets = num_targets
        self._num_hidden_dim = num_hidden_dim
        self._num_hidden_layers = num_hidden_layers
        self._dropout_prob = dropout_prob

        self._lstm_1 = nn.LSTM(
            self._num_features,
            self._num_hidden_dim,
            num_layers=self._num_hidden_layers,
            dropout=self._dropout_prob,
            batch_first=True,
        )

        self._linear_1 = nn.Linear(self._num_hidden_dim, self._num_targets)

        self.save_hyperparameters()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = self._lstm_1(batch)
        x = torch.relu(x)
        x = self._linear_1(x)

        return self._activation(x)
