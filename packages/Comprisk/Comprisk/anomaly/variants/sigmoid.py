"""This module contains the SigmoidTransformer, which uses a sigmoid $$f(x)=1/(1+e^-x)$$ to clip the model's
expected/predicted output difference into the range 0  to 1. First, the loss values are standardized using the sklearn
StandardScaler. Then, the median of the standardized loss values is computed and used to transform the sigmoid on the
x-axis."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from Comprisk.anomaly.base_transformer import BaseTransformer, loss_fn


def sigmoid(x: np.ndarray, smooth: np.ndarray, factor: np.ndarray | float = 1.0) -> np.ndarray:
    factor = np.ones_like(smooth) * factor if isinstance(factor, float) else factor
    return 2 / (1 + np.exp(-factor * x - smooth)) - 1


class SigmoidTransformer(BaseTransformer):
    def __init__(self, noise_floor: float = 0.687, upper_bound: float = 0.999) -> None:
        self._scaler = StandardScaler()

        self._param_upper_bound: float = upper_bound
        self._param_noise_floor: float = noise_floor
        self._noise_floor: np.ndarray | None = None
        self._quantile_delta: np.ndarray | None = None

    def fit(self, predicted: pd.DataFrame, target: pd.DataFrame) -> None:
        loss = loss_fn(predicted, target)
        loss = abs(self._scaler.fit_transform(loss))

        # use median and the upper quantile to find the smoothing parameter per feature
        upper_quantiles = np.quantile(loss, self._param_upper_bound, axis=0)
        self._noise_floor = np.quantile(loss, self._param_noise_floor, axis=0)
        self._quantile_delta = (upper_quantiles - self._noise_floor)

    def transform(self, predicted: pd.DataFrame, target: pd.DataFrame, **kwargs) -> pd.DataFrame:
        assert self._noise_floor is not None, "Fit the transformer first."
        assert self._quantile_delta is not None, "Fit the transformer first."
        loss = loss_fn(predicted, target)
        loss = abs(self._scaler.transform(loss))

        loss = sigmoid(loss, self._quantile_delta, 1 / self._quantile_delta ** 2)
        return pd.DataFrame(loss, columns=predicted.columns).clip(0, 1)
