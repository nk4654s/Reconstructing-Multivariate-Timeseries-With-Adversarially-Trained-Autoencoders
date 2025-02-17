from typing import Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

from Comprisk.model import BaseModel
from Dataflux import get_logger, logging_mode
from Dataflux.pytorch import BATCH_TYPE

logger = get_logger(__name__, "logfile", logging_mode)
PREDICT_TYPE = Union[np.ndarray, torch.Tensor]


class ModelNotTrainedError(BaseException):
    pass


class ScoreWrapper:
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
        threshold: float = 0.8,
        threshold_deviation: float = 5.0,
        noise_floor: float = 1.0,
    ) -> None:
        super(ScoreWrapper, self).__init__(model)

        self._model = model

        self._threshold = Parameter(Tensor([threshold]))
        self._threshold_deviation = Parameter(Tensor([threshold_deviation]))
        self._noise_floor: Parameter = Parameter(Tensor([noise_floor]))

        self._trained: bool = False
        self._mean: Parameter | None = None
        self._scale: Parameter | None = None

    @property
    def sub_model(self) -> nn.Module:
        return self._model

    @torch.no_grad()
    def train_wrapper(self, data: DataLoader, verbose: bool = False) -> None:
        # initialize the parameters
        loss = []
        mean: np.ndarray | None = None

        # find the mean and scale of the data to run a transformation
        for x, y in tqdm(data, desc="Prediction with Model", disable=not verbose):
            y_ = self._model.forward(x.to(self._model.device))
            if mean is None:
                _, _, features = y.shape
                mean = np.zeros(shape=(1, features))

            loss_ = torch.abs(y_.cpu() - y)
            loss_ = loss_.numpy().reshape(-1, y_.shape[-1])
            mean += loss_.sum(axis=0)
            loss += list(loss_)

        # finalize the mean and calculate the variance
        loss = np.array(loss)
        mean = mean / len(loss)
        var = ((loss - mean)**2).sum(axis=0) / (len(loss) - 1)
        var[var == 0] = 1.0
        var = np.sqrt(var)

        # the mean and scale must be a parameter of this model such that it will be
        # visible in the model graph. This is required by ONNX
        self._mean = Parameter(Tensor(mean / len(data)))
        self._scale = Parameter(Tensor(var))
        self._trained = True

        logger.debug("Trained %s successfully", self.__name__)
        logger.debug("Mean values %s", mean / len(data))
        logger.debug("Scale values %s", var)

    def predict_step(self, x: BATCH_TYPE) -> Tensor:
        # the wrapper must be trained to transfer the prediction error to a score
        if not self._trained:
            raise ModelNotTrainedError()

        with torch.enable_grad():
            x, y = x

            # get the prediction of 'x' using the model
            x = x.to(self._model.device)
            prediction: PREDICT_TYPE = self._model.forward(x)

            # transform the difference between 'x' and 'prediction' to a score
            # use a simple loss function, the absolute difference
            score = (y - prediction) - self._mean
            score = torch.abs(score / self._scale)

            # solve the following linear equation to find the linear function's parameters
            # f(x) = m * x + b
            m = (self._threshold  - 1) / (self._threshold_deviation - self._noise_floor)
            b = 1 - m * self._noise_floor

            score = torch.clip(m * score + b, 0, 1)

        return score
