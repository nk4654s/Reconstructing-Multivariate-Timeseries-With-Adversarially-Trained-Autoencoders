import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from Comprisk.anomaly.base_transformer import BaseTransformer, loss_fn


def linear(x: pd.DataFrame, factor: np.ndarray, translate: np.ndarray) -> pd.DataFrame:
    return factor * x - factor * translate # type: ignore


class AnaTransformer(BaseTransformer):
    """The ScoreTransformer learns the loss distribution of a trained model over the training data
    to translate a loss value into a score between 0 and 1. Here, 0 means the model is working fine,
    while 1 means something is wrong with the input data.
    """

    def __init__(self, noise_floor: float = 0.500, outer_border: float = 0.999) -> None:
        self._scaler = StandardScaler()

    def fit(self, predicted: pd.DataFrame, target: pd.DataFrame) -> None:
        # get the loss and train the standard scaler, a normal distribution is
        # required for the following math
        loss = loss_fn(predicted, target)
        loss = abs(self._scaler.fit_transform(loss))

    def transform(self, predicted: pd.DataFrame, target: pd.DataFrame,
                  factors: np.ndarray | float = 0.01) -> pd.DataFrame:
        pass
