"""This module contains the LinearTransformer which should keep a machine's wear curve consistent."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from Comprisk.anomaly.base_transformer import BaseTransformer, loss_fn


def linear(x: pd.DataFrame, factor: np.ndarray, translate: np.ndarray) -> pd.DataFrame:
    return factor * x - factor * translate # type: ignore


class LinearTransformer(BaseTransformer):
    def __init__(self, noise_floor: float = 0.500, outer_border: float = 0.999) -> None:
        self._scaler = StandardScaler()

        self._param_noise_floor = noise_floor
        self._param_out_border = outer_border

        self._noise_floor: np.ndarray | None = None
        self._outer_std: np.ndarray | None = None

    def fit(self, predicted: pd.DataFrame, target: pd.DataFrame) -> None:
        # get the loss and train the standard scaler, a normal distribution is
        # required for the following math
        loss = loss_fn(predicted, target)
        loss = abs(self._scaler.fit_transform(loss))

        # determine the loss' noise floor
        self._noise_floor = np.quantile(loss, self._param_noise_floor, axis=0)
        self._outer_std = np.quantile(loss, self._param_out_border, axis=0)

    def transform(self, predicted: pd.DataFrame, target: pd.DataFrame,
                  factors: np.ndarray | float = 0.01) -> pd.DataFrame:
        assert self._noise_floor is not None, "Fit the transformer first."
        assert self._outer_std is not None
        loss = loss_fn(predicted, target)
        loss = abs(self._scaler.transform(loss))

        factors = np.ones_like(loss.shape[-1]) * factors if isinstance(factors, float) else factors
        factors = np.ones(shape=(1, loss.shape[-1])) * factors
        factors = self._scaler.transform(factors)

        # calculate the function's gradient by solving the linear equation:
        # f(outer_std) = ax - noise_floor = factors
        factors = (factors + self._noise_floor) / self._outer_std

        loss = factors * loss - self._noise_floor  # type: ignore
        loss = self._scaler.inverse_transform(loss)
        return pd.DataFrame(loss, columns=predicted.columns, index=predicted.index).clip(0, 1)
