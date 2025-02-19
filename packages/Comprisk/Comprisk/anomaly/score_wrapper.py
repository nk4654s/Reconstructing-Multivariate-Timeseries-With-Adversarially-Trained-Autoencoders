import os.path
from typing import Union

import numpy as np
import torch
import yaml
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from lightning import LightningModule
from tqdm import tqdm

from Comprisk.model import BaseModel
from Dataflux import get_logger, logging_mode

logger = get_logger(__name__, "logfile", logging_mode)
PREDICT_TYPE = Union[np.ndarray, torch.Tensor]


class ModelNotTrainedError(BaseException):
    pass


class ScoreWrapper(LightningModule):
    """
    This score transformer was developed by the company Anacision and uses the statistics of the model's performance
    to learn the loss distribution. Using a linear transformation defined by the learned distribution, future loss
    values are clipped into the range 0 to 1.

    Args:
        model (BaseModel): The module to wrap and from which the loss is used to calculate a score.
        noise_floor (float): How far a loss can deviate from the learned distribution before it becomes
            significant. Defaults to `1.0`.
        threshold_deviation (float): How far a loss can deviate to be defined as upper limit. Defaults to `5.0`
        threshold (float): The value, which should be reached when hitting the upper limit. Defaults to `0.8`.
    """
    def __init__(
        self,
        model: BaseModel,
        threshold: float = 0.9,
        threshold_deviation: float = 5.0,
        noise_floor: float = 1.0,
    ) -> None:
        super(ScoreWrapper, self).__init__()

        self._model = model

        self._threshold = Parameter(Tensor([threshold]))
        self._threshold_deviation = Parameter(Tensor([threshold_deviation]))
        self._noise_floor: Parameter = Parameter(Tensor([noise_floor]))

        self._trained: bool = False
        self._mean: Parameter | None = None
        self._scale: Parameter | None = None

    @property
    def device(self):
        return self._model.device

    @property
    def sub_model(self) -> nn.Module:
        return self._model

    @torch.no_grad()
    def train_wrapper(self, data: DataLoader, verbose: bool = False) -> None:
        # initialize the parameters
        loss = []
        self._model.eval()

        # find the mean and scale of the data to run a transformation
        for x, y in tqdm(data, desc="Prediction with Model", disable=not verbose):
            x = x.to(self._model.device)
            y = y.to(self._model.device)
            y_hat, y = self._model.predict_step(x, y)

            loss_ = torch.abs(y_hat - y).cpu().detach()
            loss_ = loss_.numpy().reshape(-1, y_hat.shape[-1])
            loss += list(loss_)

        # finalize the mean and calculate the variance
        loss = np.array(loss)
        var = ((loss - loss.mean(0))**2).sum(axis=0) / (len(loss) - 1)
        var[var == 0] = 1.0
        var = np.sqrt(var)

        # the mean and scale must be a parameter of this model such that it will be
        # visible in the model graph. This is required by ONNX
        self._mean = Parameter(Tensor(loss.mean(0)))
        self._scale = Parameter(Tensor(var))
        self._trained = True

        self.to(self.device)

        logger.debug("Trained %s successfully", self.__class__.__name__)
        logger.debug("Scale values %s", var)

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # get the prediction of 'x' using the model
        x = x.to(self._model.device)
        y = y.to(self._model.device)

        # do not normalize before the model, this will be handled by the model itself
        y_hat, y = self._model.predict_step(x, y)

        # transform the difference between 'x' and 'prediction' to a score
        # use a simple loss function, the absolute difference
        score = torch.abs(y - y_hat) - self._mean
        score = score / self._scale

        # solve the following linear equation to find the linear function's parameters
        # f(x) = m * x + b
        m = (self._threshold  - 1) / (self._threshold_deviation - self._noise_floor)
        b = 1 - m * self._noise_floor

        return torch.clip(m * score + b, 0, 1)

    def predict_step(self, x: Tensor, y: Tensor) -> Tensor:
        # the wrapper must be trained to transfer the prediction error to a score
        if not self._trained:
            raise ModelNotTrainedError()

        return self.forward(x, y)

    def export_to(self, path: str) -> None:
        mean = self._mean.cpu().detach().tolist()
        scale = self._scale.cpu().detach().tolist()

        with open(os.path.join(path, "score.yaml"), "w") as f:
            yaml.safe_dump({
                "mean": mean,
                "scale": scale
            }, f)

    def import_from(self, path: str) -> None:
        with open(os.path.join(path, "score.yaml"), "r") as f:
            data = yaml.safe_load(f)

        self._mean = Parameter(Tensor(data["mean"]))
        self._scale = Parameter(Tensor(data["scale"]))
        self._trained = True

        self.to(self.device)


