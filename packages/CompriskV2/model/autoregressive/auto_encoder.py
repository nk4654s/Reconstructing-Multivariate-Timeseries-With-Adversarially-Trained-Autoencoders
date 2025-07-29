import numpy as np
import torch
from torch import nn, zeros, Tensor
from torch.nn.parameter import Parameter
from CompriskV2.model import LinkedParameterModel


class ScoreData:
    def __init__(self) -> None:
        self._data = []

    @property
    def ready(self) -> bool:
        return not isinstance(self._data, list)

    def record(self, data: Tensor) -> None:
        if not self.ready:
            temp = data.view(data.size(0) * data.size(1), -1)
            self._data += temp.cpu().detach().tolist()

    def finalize(self, device) -> tuple[Tensor, Tensor]:
        if self.ready:
            return None

        self._data = np.array(self._data)
        weight = 1 / self._data.std(axis=0)
        weight[weight < 1] = 1
        weight = Parameter(Tensor(weight).to(device))

        total = ((self._data - self._data.mean(axis=0)) ** 2).mean(axis=0)
        total = Parameter(Tensor(total).to(device))
        self._data = None

        return weight, total


class AutoregressiveAutoencoder(LinkedParameterModel):
    def __init__(
        self,
        num_hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        dropout: float = 0.3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._features_out_indicies = [self._names_in.index(_) for _ in self._names_out]

        self._encoder = nn.LSTM(
            self._features_in,
            num_hidden_dim,
            num_layers=num_hidden_layers,
            dropout=dropout,
            batch_first=True,
            device=self.device,
        )

        self._decoder = nn.LSTM(
            self._features_out,
            num_hidden_dim,
            num_layers=num_hidden_layers,
            dropout=dropout,
            batch_first=True,
            device=self.device,
        )

        self._reduce = nn.Linear(num_hidden_dim, self._features_out)

        # scoring parameters
        self._weight = Parameter(zeros(self._features_out))
        self._total = Parameter(zeros(self._features_out))
        self._ready = True

        self._score_data = ScoreData()

    @property
    def labels(self) -> tuple[str, ...]:
        return ("y_scaled", "y_hat", "score")

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        y, y_pred = self(*batch, return_score=False)

        loss = self._loss(y, y_pred)
        self.log("train/loss", loss)

        self._score_data.record(y)
        return loss

    def on_train_start(self) -> None:
        self._ready = False

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        y, y_pred, score = self(*batch)

        loss = self._loss(y, y_pred)
        self.log("val/loss", loss)
        self.log("val/score", torch.mean(score))

        self._score_data.record(y.unsqueeze(1))
        return loss

    def on_validation_end(self) -> None:
        if not self._ready:
            self._weight, self._total = self._score_data.finalize(self.device)
            self._ready = True

    @torch.no_grad()
    def score(self, y: Tensor, y_hat: Tensor) -> Tensor:
        if not self._ready:
            return torch.zeros_like(y)

        res = torch.pow(y - y_hat, 2)
        # 1 std error is noise floor
        res = torch.clip(res - torch.Tensor(1), 0, None)
        res = 1 - torch.div(res, self._total * self._weight)
        return torch.clip(res, 0, 1)

    def forward(
        self, x: Tensor, y: Tensor, return_score: bool = True
    ) -> tuple[Tensor, ...]:
        x, y = super().forward(x, y)

        _, hc = self._encoder(x[:, -2:, :])
        x_in = x[:, -1, self._features_out_indicies].unsqueeze(1)
        y_hat, _ = self._decoder(x_in, hc)

        y_hat = self._reduce(y_hat)
        if return_score:
            y = y[:, 0, :]
            y_hat = y_hat[:, 0, :]
            return y, y_hat, self.score(y, y_hat)

        return y, y_hat
