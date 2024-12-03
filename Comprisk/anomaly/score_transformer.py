"""This module contains the score transformer which maps a loss value into a region of 0 to 1."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Dataflux.visualization import show_dataframe
from Comprisk import logger


def loss_fn(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(np.power(y - y_hat, 2)) # type: ignore


def sigmoid(x: pd.DataFrame, smooth: np.ndarray, factor: np.ndarray | float = 1.0) -> pd.DataFrame:
    factor = np.ones_like(smooth) * factor if isinstance(factor, float) else factor
    x = x.copy()
    for i, name in enumerate(x.columns):
        x[name] = 2 / (1 + np.exp(-factor[i] * (x[name] - smooth[i]))) - 1 # type: ignore
    return x


class ScoreTransformer:
    """The ScoreTransformer learns the loss distribution of a trained model over the training data
    to translate a loss value into a score between 0 and 1. Here, 0 means the model is working fine,
    while 1 means something is wrong with the input data.

    The ScoreTransformer uses a sigmoidf(x)=1/(1+e^-x) to map the loss values into a score between 0 and 1. First,
    the loss values are standardized using sklearn's StandardScaler. Then, the median of
    the standardized loss values is computed and used to transform the sigmoid on the
    x-axis. To controll how sensitive the transformer behaves, the x-value is multiplied with a given factor.

    Args:
        upper_bound (float): The upper quantile in percent, required to calculate a smoothing factor.
            The smoothing factor defines how high a loss can get before the ScoreTransformer outputs 
            a score of 1. Defaults to 0.999.
        sensitivity (float): The factor controls how fast the underlying sigmoid function increases
            from 0 to 1. Hence, it can be called the sensitivity of the transformer and A lower value
            results in less sensitivity, defaults to 0.3.
    """
    def __init__(self, upper_bound: float = 0.999, sensitivity: float = 0.3) -> None:
        self._error_scaler = StandardScaler()
        self._sensitivity = sensitivity

        self._upper_bound = upper_bound
        self._smoothing: np.ndarray | None = None

    def fit(self, predicted: pd.DataFrame, target: pd.DataFrame) -> None:
        """This method fits the parameters of the ScoreTransformer to create a function based on f(x)=e^x
        which transforms loss values in the range of 0 to 1. This method calculates the median to
        translate the internal transformation along the x-axis. Then, the smoothing factor is calculated
        to determine how strict a loss is clipped in the range of 0 to 1.

        Args:
            predicted (np.array): The predicted values of the model.
            target (np.array): The target values of the model.
        """
        assert self._smoothing is None, "Smoothing already fitted."
        predicted = predicted.reindex(sorted(predicted.columns), axis=1)
        target = target.reindex(sorted(target.columns), axis=1)

        # compute loss
        loss = loss_fn(predicted, target)

        # standardize loss
        scores = self._error_scaler.fit_transform(loss)

        # use median and the upper quantile to find the smoothing parameter per feature
        quantiles = np.quantile(scores, self._upper_bound, axis=0)
        medians = np.quantile(scores, 0.5, axis=0)
        self._smoothing = (quantiles - medians)
        
        logger.debug("Fitted ScoreTransformer with smoothing factor %s", self._smoothing)

    def transform(self, predicted: pd.DataFrame, target: pd.DataFrame, visualize: str = "",
                  show: bool = False) -> pd.DataFrame:
        """This method transforms the loss values in the range of 0 to 1 by applying the internal
        transformation function. Make sure to run `self.fit(predicted, target)` first, as the 
        transformation function's parameters must be found first. Only run `.fit()` on training data.

        Args:
            predicted (pd.DataFrame): The predicted values of the model.
            target (pd.DataFrame): The target values of the model.
            visualize (str): Whether to visualize the transformed loss values or not. Parameter defines
                the output path of the visualization. Defaults to "", which does not generate a graph.
            show (bool): Whether to show the graph or not.
            
        Returns:
            (pd.DataFrame): The transformed loss values in the range of 0 to 1. 0 is best
                and 1 is worst.
        """
        assert self._smoothing is not None, "Fit the transformer first."
        predicted = predicted.reindex(sorted(predicted.columns), axis=1)
        target = target.reindex(sorted(target.columns), axis=1)

        # compute anomaly scores
        loss = loss_fn(predicted, target)
        anomaly_scores = self._error_scaler.transform(loss)
        anomaly_scores = pd.DataFrame(anomaly_scores, columns=loss.columns, index=loss.index)
        anomaly_scores = anomaly_scores.reindex(sorted(anomaly_scores.columns), axis=1)

        malfunction = sigmoid(anomaly_scores, self._smoothing, factor=1 / self._smoothing**2)
        malfunction.clip(0, None, inplace=True)

        # transfer all values to dataframes and return them
        if show or visualize:
            show_dataframe([malfunction], ["Malfunction"], y_label="% -  lower is better",
                        grid_size=5, y_limit=(0, 1), save=visualize, show=show)
    
        return malfunction
