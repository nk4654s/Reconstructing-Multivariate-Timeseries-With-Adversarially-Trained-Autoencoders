import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import AdamW
from lightning import LightningModule


def r2_score(i, t):
    mean = t.mean(axis=0)
    ss_regression = ((i - mean) ** 2).sum(axis=0)
    ss_total = ((t - mean) ** 2).sum(axis=0)
    r2_score = ss_regression / ss_total

    return (1 / r2_score.clip(1, None)).mean()


class BaseModel(LightningModule):
    def __init__(self, optimizer: type = AdamW, lr: float = 1e-3, loss: type = nn.L1Loss) -> None:
        super().__init__()

        self._lr = lr
        self._optim = optimizer

        self._loss = loss()

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        y, y_pred = self.predict_step(*batch)

        loss = self._loss(y, y_pred)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        y, y_pred = self.predict_step(*batch)

        loss = self._loss(y, y_pred)
        self.log("val/loss", loss)
        self.log("val/r2", r2_score(y_pred, y))
        return loss

    def configure_optimizers(self):
        if isinstance(self._optim, type):
            self._optim = self._optim(self.parameters(), lr=self._lr)

        return self._optim

    def score(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError()

    def predict_step(self, x: Tensor, y: Tensor) -> tuple[Tensor, ...]:
        return self(x, y)


class LinkedParameterModel(BaseModel):
    def __init__(self, features_in: list[str], features_out: list[str], sequence_in: int, sequence_out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._features_in = len(features_in)
        self._features_out = len(features_out)

        features_in = sorted(features_in)
        features_out = sorted(features_out)
        self._features_out_indicies = [features_in.index(_) for _ in features_out]

        self._sequence_in = sequence_in
        self._sequence_out = sequence_out

        self._mean = nn.Parameter(torch.zeros(self._features_in))
        self._scale = nn.Parameter(torch.zeros(self._features_in))

        self._eps = 1e-6

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor, return_score: bool = True) -> tuple[Tensor, ...]:
        x = (x - self._mean) / (self._scale + self._eps)
        y = (y - self._mean[self._features_out_indicies]) / (self._scale[self._features_out_indicies] + self._eps)
        return x, y

    def set_standardizer(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self._mean = nn.Parameter(torch.Tensor(mean))
        self._scale = nn.Parameter(torch.Tensor(scale))
