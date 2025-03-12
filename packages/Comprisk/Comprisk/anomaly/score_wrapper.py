import os.path
from typing import Tuple, Union

import numpy as np
import torch
import yaml
from Dataflux import get_logger, logging_mode
from lightning import LightningModule
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

from Comprisk.model import BaseModel

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
            significant. Defaults to 1 std (=`65.9`%).
        threshold_deviation (float): How far a loss can deviate to be defined as upper limit. Defaults to 3 std
            (=`99.9`%).
        threshold (float): The value, which should be reached when hitting the upper limit. Defaults to `0.8`.
    """

    def __init__(
        self,
        model: BaseModel,
        threshold: float = 0.9,
        threshold_deviation: float = 5,
        noise_floor: float = 1,
    ) -> None:
        super(ScoreWrapper, self).__init__()

        self._model = model
        self._model.eval()

        self._param_noise_floor = noise_floor
        self._param_threshold_deviation = threshold_deviation
        self._param_threshold = threshold

        # f(x) = a * x + b
        self._slope: Parameter | None = None
        self._intercept: Parameter | None = None

        self._trained: bool = False

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

        # find the mean and scale of the data to run a transformation
        for x, y in tqdm(data, desc="Prediction with Model", disable=not verbose):
            x = x.to(self._model.device)
            y = y.to(self._model.device)
            y_hat, y = self._model.predict_step(x, y)

            loss_ = torch.abs(y_hat - y).cpu().detach()
            loss_ = loss_.numpy().reshape(-1, y_hat.shape[-1])
            loss += list(loss_)

        # finalize loss to search for percentiles
        loss = np.array(loss)

        # find the quantiles of the loss distribution to calculate a suitable
        # function to transform future losses to scores
        noise_floor = loss.mean(0) + self._param_noise_floor * loss.std(0)
        sigma_loss = loss.mean(0) + self._param_threshold_deviation * loss.std(0)

        # solve the following linear equation to find the linear function's parameters
        # f(x) = m * x + b
        slope = (self._param_threshold - 1) / (sigma_loss - noise_floor)
        intercept = 1 - slope * noise_floor

        self._slope = Parameter(Tensor(slope))
        self._intercept = Parameter(Tensor(intercept))
        self.to(self.device)

        self._trained = True

        logger.debug("Trained %s successfully", self.__class__.__name__)

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, ...]:
        # get the prediction of 'x' using the model
        x = x.to(self._model.device)
        y = y.to(self._model.device)

        # normalize will be handled by the model itself
        y_hat, y = self._model.predict_step(x, y)
        if not self._trained:
            return y_hat, y

        score = torch.abs(y_hat - y)

        # f(x) = m * x + b
        f = self._slope * score + self._intercept

        _, y_hat = self._model.normalizer.reverse(x=x, y=y_hat)
        return torch.clip(f, 0, 1), y_hat

    def predict_step(self, x: Tensor, y: Tensor) -> Tuple[Tensor, ...]:
        # the wrapper must be trained to transfer the prediction error to a score
        if not self._trained:
            y_hat, y = self._model.predict_step(x, y)
            return y_hat, y

        return self.forward(x, y)

    def export_to(self, path: str) -> None:
        a = self._slope.cpu().detach().tolist()
        b = self._intercept.cpu().detach().tolist()

        with open(os.path.join(path, "score.yaml"), "w") as f:
            yaml.safe_dump({"slope": a, "intercept": b}, f)

    def import_from(self, path: str) -> None:
        with open(os.path.join(path, "score.yaml"), "r") as f:
            data = yaml.safe_load(f)

        self._slope = Parameter(Tensor(data["slope"]))
        self._intercept = Parameter(Tensor(data["intercept"]))
        self._trained = True

        self.to(self.device)
