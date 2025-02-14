"""This module contains the ScoreTransformer by Anacision GmbH and operates similar to the LinearTransformer."""
from typing import Dict, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from Comprisk.anomaly.base_transformer import BaseTransformer, loss_fn


def linear(x: pd.DataFrame, factor: np.ndarray, translate: np.ndarray) -> pd.DataFrame:
    return factor * x - factor * translate # type: ignore


class AnaTransformer(BaseTransformer):
    """
    This score transformer was developed by the company Anacision and uses the statistics of the model's performance
    to learn the loss distribution. Using a linear transformation defined by the learned distribution, future loss
    values are clipped into the range 0 to 1.

    Args:
        noise_deviation (float): How far a loss can deviate from the learned distribution before it becomes
            significant. Defaults to `1.0`.
        threshold_deviation (float): How far a loss can deviate to be defined as upper limit. Defaults to `5.0`
        threshold (float): The value, which should be reached when hitting the upper limit. Defaults to `0.8`.
        clip_below_zero (bool): Force values to be in the range 0 to 1. Defaults to true.
    """
    def __init__(self, noise_deviation: float = 1.0, threshold_deviation: float = 5.0,
                 threshold: float = 0.8, clip_below_zero: bool = True) -> None:

        self._clip = clip_below_zero
        self._scaler = StandardScaler()

        self._noise_deviation = noise_deviation
        self._threshold = threshold
        self._threshold_deviation = threshold_deviation

        self._names: List[str] = []

    def fit(self, predicted: pd.DataFrame, target: pd.DataFrame) -> None:
        self._names = list(predicted.columns)

        # get the loss and train the standard scaler, a normal distribution is
        # required for the following math
        loss = loss_fn(predicted, target)
        self._scaler.fit_transform(loss)

    def transform(self, predicted: pd.DataFrame, target: pd.DataFrame, **kwargs) -> pd.DataFrame:
        loss = loss_fn(predicted, target)
        loss = np.abs(self._scaler.transform(loss))

        # solve the following linear equation to find the linear function's parameters
        # f(x) = m * x + b
        m = (self._threshold - 1) / (self._threshold_deviation - self._noise_deviation)
        b = 1 - m * self._noise_deviation

        # clip the result if required
        clip = 0 if self._clip else None
        health_scores = 1 - np.clip(m * loss + b, clip, 1)
        health_scores = pd.DataFrame(health_scores, columns=predicted.columns, index=predicted.index)

        return health_scores

    def export(self) -> Dict:
        export = {}
        for i, name in enumerate(self._names):
            entry = {
                "noise_deviation": self._noise_deviation[i],
                "threshold_deviation": self._threshold_deviation[i],
                "threshold": self._threshold[i],
                "mean": self._scaler.mean_[i],
                "scale": self._scaler.scale_[i]
            }

            export[name] = entry
        return export
