"""This module contains the ScoreTransformer by Anacision GmbH and operates similar to the LinearTransformer."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from Comprisk.anomaly.base_transformer import BaseTransformer, loss_fn


def linear(x: pd.DataFrame, factor: np.ndarray, translate: np.ndarray) -> pd.DataFrame:
    return factor * x - factor * translate # type: ignore


class AnaTransformer(BaseTransformer):
    def __init__(self, noise_deviation: float = 1.0, threshold_deviation: float = 5.0,
                 threshold: float = 0.8, clip_below_zero: bool = True) -> None:

        self._clip = clip_below_zero
        self._scaler = StandardScaler()

        self._noise_deviation = noise_deviation
        self._threshold = threshold
        self._threshold_deviation = threshold_deviation

    def fit(self, predicted: pd.DataFrame, target: pd.DataFrame) -> None:
        # get the loss and train the standard scaler, a normal distribution is
        # required for the following math
        loss = loss_fn(predicted, target)
        self._scaler.fit_transform(loss)

    def transform(self, predicted: pd.DataFrame, target: pd.DataFrame,
                  factors: np.ndarray | float = 0.01) -> pd.DataFrame:
        loss = loss_fn(predicted, target)
        loss = np.abs(self._scaler.transform(loss))

        # solve the following linear equation to find the linear function's parameters
        # f(x) = m * x + b
        m = (self._threshold - 1) / (self._threshold_deviation - self._noise_deviation)
        b = 1 - m * self._noise_deviation

        # clip the result if required
        clip = 0 if self._clip else None
        health_scores = np.clip(m * loss + b, clip, 1)
        health_scores = pd.DataFrame(health_scores, columns=predicted.columns, index=predicted.index)

        return health_scores
