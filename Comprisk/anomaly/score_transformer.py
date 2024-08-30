"""This module contains the score transformer which maps a loss value into a region of 0 to 1."""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Dataflux.visualization import show_dataframe
from Comprisk import logger


class ScoreTransformer:
    """The ScoreTransformer learns the loss distribution of a trained model over the training data
    to translate a loss value into a score between 0 and 1. Here, 0 means the model is working fine,
    while 1 means something is wrong with the input data.

    The ScoreTransformer uses f(x)=e^x to map the loss values into a score between 0 and 1. First,
    the loss values are standardized using sklearn's StandardScaler. Then, the median of
    the standardized loss values is computed and used to transform the function f(x)=e^x on the
    x-axis. Hence, a loss value of 0 will be transformed into 1 and a loss values greater than
    0 will result in a lower score. Finally, a computed smoothing factor defines how high a loss can
    get before the ScoreTransformer outputs a score of 1

    Args:
        upper_bound (float): The upper quantile in percent, required to calculate a smoothing factor.
            The smoothing factor defines how high a loss can get before the ScoreTransformer outputs 
            a score of 1. Defaults to 0.999.
        smoothing_amplifier (float): The smoothing amplifier increases the smoothing factor based on
            the base logarithm log(x). So smaller value result in higher amplifications. Defaults to
            0.9.
        critical_threshold (float): The threshold value for a loss value to be considered as critical.
            Defaults to 1.
    """
    def __init__(self, upper_bound: float = 0.999, smoothing_amplifier: float = 0.9, critical_threshold: float = 1) -> None:
        self._error_scaler = StandardScaler()

        self._upper_bound = upper_bound
        self._factor = smoothing_amplifier

        self._features = set()
        self._median_quantile = None
        self._smoothing = None

        self._loss_fn = lambda y, y_hat: np.sqrt(np.power(y - y_hat, 2))
        self._critical_threshold = critical_threshold

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

        # compute anomaly scores
        error = predicted - target
        anomaly_scores = np.abs(self._error_scaler.fit_transform(error.to_numpy()))

        # Compute the thresholds for the given percentages.
        # If zero is given, we do not want to compute the quantile but just use zero as the threshold.
        self._median_quantile = np.quantile(anomaly_scores, 0.5, axis=0)
        smoothing_target = np.quantile(anomaly_scores, self._upper_bound, axis=0)
        smoothing_target -= self._median_quantile

        self._smoothing = -(smoothing_target**2) / np.log(self._factor)
        logger.debug("Fitted ScoreTransformer with smoothing factor %s", self._smoothing)

        self._features = set(predicted.columns.values)

    def transform(self, predicted: pd.DataFrame, target: pd.DataFrame, visualize: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This method transforms the loss values in the range of 0 to 1 by applying the internal
        transformation function. Make sure to run `self.fit(predicted, target)` first, as the 
        transformation function's parameters must be found first. Only run `.fit()` on training data.

        Args:
            predicted (pd.DataFrame): The predicted values of the model.
            target (pd.DataFrame): The target values of the model.
            visualize (str): Whether to visualize the transformed loss values or not. Parameter defines
                the output path of the visualization. Defaults to None, which does not generate a graph.
        
        Returns:
            (Tuple[pd.DataFrame, pd.DataFrame]): (1) The transformed loss values in the range of 0 to 1. 0 is best
                and 1 is worst. (2) The scores which are critical as the loss values exceed the known region.
        """
        assert self._smoothing is not None, "Fit the transformer first."

        # compute anomaly scores
        error = predicted - target
        anomaly_scores = np.abs(self._error_scaler.transform(error.to_numpy()))

        # compute the malfunction probability in the region 0 to 1
        medians = self._median_quantile
        malfunction = 1 - np.exp(
            -((anomaly_scores - medians).clip(0, None) ** 2) / self._smoothing
        )

        # get the loss values to find wether a score is critical or not
        loss = self._loss_fn(predicted.to_numpy(), target.to_numpy())
        critical = malfunction.copy()
        critical[loss < self._critical_threshold] = np.nan

        # apply the same to the loss itself, this will be used when visualizing 
        # the transformed values
        loss[loss < self._critical_threshold] = np.nan

        # transfer all values to dataframes and return them
        names = predicted.columns.values
        malfunction = pd.DataFrame(malfunction, columns=names)
        critical = pd.DataFrame(critical, columns=names)
        loss = pd.DataFrame(loss, columns=names)

        if visualize:
            # TODO: remove show and replace with store given the path
            show_dataframe(names, [malfunction, loss, critical], ["Malfunction", "Critical Evidence", "Critical"],
                           y_label="% -  lower is better", show=True)
    
        return malfunction, critical
