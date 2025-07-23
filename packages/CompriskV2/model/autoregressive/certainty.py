import numpy as np
from torch import Tensor
import torch
import torch.nn as nn

from CompriskV2.model import LinkedParameterModel


def r2_score(i, t):
    mean = t.mean(axis=0)
    ss_regression = ((i - mean) ** 2).sum(axis=0)
    ss_total = ((t - mean) ** 2).sum(axis=0)
    r2_score = ss_regression / ss_total

    return (1 / r2_score.clip(1, None)).mean()


class AutoregressiveCertaintyModel(LinkedParameterModel):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        regression_training: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(loss="GaussianNLLLoss", **kwargs)
        assert (
            self._sequence_out == 1
        ), "This model does not work with other sequence output lengths than 1"

        self._loss = nn.GaussianNLLLoss()
        self._reg_loss = nn.L1Loss()
        self._loss_switch = regression_training

        ones = torch.ones(self._features_out)
        self._dist_custom = torch.distributions.Normal(-1.6 * ones, 1.5 * ones)

        self._lstm = nn.LSTM(
            self._features_in, hidden_dim, num_layers, dropout=dropout, batch_first=True
        )
        self._reduce = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )

        self._mean_fc = nn.Linear(hidden_dim // 4, self._features_out)
        self._var_fc = nn.Linear(hidden_dim // 4, self._features_out)

    @property
    def labels(self) -> tuple[str]:
        return ("y", "y_hat", "var", "score")

    def _switch_loss(self) -> bool:
        return np.random.rand() < self._loss_switch

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        y, mean, var = self.forward(*batch, return_score=False)

        if self._switch_loss():
            loss = self._reg_loss(mean, y)
            self.log("train/l1", loss)
        else:
            loss = self._loss(mean, y, var)
            self.log("train/loss", loss)

        self.log("train/var", torch.mean(var))
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        y, mean, var = self.forward(*batch, return_score=False)

        loss = self._loss(mean, y, var)
        self.log("val/loss", loss)
        self.log("val/l1", nn.L1Loss()(mean, y))
        self.log("val/r2", r2_score(mean, y))
        self.log("val/var", torch.mean(var))
        return loss

    def score(self, mu: Tensor, y: Tensor, var: Tensor) -> Tensor:
        var = 7 * (torch.sqrt(var) + torch.tensor(1e-6))

        # base tolerance of 0.5 is always ok
        var = torch.nan_to_num(var)
        var = torch.clip(var, 0.5, None)

        # cdf with mu=-1.6 and scale=1.5
        # loss == var -> 0.8 and loss < var -> score > 0.8
        score = self._dist_custom.cdf(-torch.abs(y - mu) / var) * 1.2
        return torch.clip(score, 0, 1)

    def forward(
        self, x: Tensor, y: Tensor, return_score: bool = True
    ) -> tuple[Tensor, ...]:
        # standardize transform batch
        x, y = super().forward(x, y)

        x, _ = self._lstm(x)
        x = torch.relu(x)

        x = self._reduce(x[:, -1, :])

        xm = self._mean_fc(x)
        xv = self._var_fc(x)
        xv = torch.pow(xv, 2)

        y = y[:, 0, :]
        if return_score:
            return y, xm, xv, self.score(xm, y, xv)

        return y, xm, xv
