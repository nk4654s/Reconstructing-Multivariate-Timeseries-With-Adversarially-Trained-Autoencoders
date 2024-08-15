import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary as torchsummary

from Dataflux.pytorch.metadata import Metadata
from Comprisk.model.base import BaseModel


class FSLSTMModel(BaseModel):
    def __init__(
        self,
        num_features: int = 27,
        num_targets: int = 26,
        num_hidden_dim: int = 110,
        num_hidden_layers: int = 2,
        dropout_prob: float = 0.3,
        loss: str = "MSELoss",
        activation: str = "sigmoid",
        optimizer: str = "Adam",
        lr: float = 0.001,
        lr_decay: float = 0.99,
        monte_carlo: int = 0,
    ) -> None:
        super().__init__(loss, activation, optimizer, lr, lr_decay, monte_carlo)

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

    def summary(self, meta: Metadata) -> str:
        samples = meta.num_input_sequence
        sensors = meta.num_features_sensors
        weather = meta.num_features_weather

        input_data = {
            "sensors": torch.tensor(
                np.zeros((1, samples, sensors), dtype=np.float32)
            ),
            "weather": torch.tensor(
                np.zeros((1, samples, weather), dtype=np.float32)
            ),
        }

        return str(torchsummary(self, input_data={"batch": input_data}))

    @property
    def target(self) -> str:
        """The target name of inputs to predict."""
        return ["targets"]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([batch["sensors"], batch["weather"]], dim=-1)
        batch_size, _, _ = inputs.shape

        _, (hn, cn) = self._lstm_1(inputs)
        x = torch.relu(hn[-1])
        x = self._linear_1(x)

        outputs = torch.unsqueeze(x, 1)
        for _ in range(9):
            # roll the inputs to replace the las value with the predicted one
            inputs = torch.roll(inputs, -1, 0)

            # no weather data are available, just add a zero
            x = torch.unsqueeze(x, 1)
            x = torch.cat([x, torch.zeros(batch_size, 1, 1, device=x.device)], dim=-1)
            inputs[:, -1, :] = x[:, 0, :]

            # continue prediction
            _, (hn, cn) = self._lstm_1(inputs, (hn, cn))
            x = self._linear_1(hn[-1])

            x = torch.unsqueeze(x, 1)
            outputs = torch.cat([outputs, x], dim=1)
            x = torch.squeeze(x, 1)

        return self._activation(outputs)
