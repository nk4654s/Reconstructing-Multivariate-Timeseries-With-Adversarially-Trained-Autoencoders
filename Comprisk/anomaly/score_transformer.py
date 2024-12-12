"""This module contains the score transformer which maps a loss value into a region of 0 to 1."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Dataflux.visualization import show_dataframe
from Comprisk import logger


def loss_fn(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.DataFrame:
    return y - y_hat


def sigmoid(x: np.ndarray, smooth: np.ndarray, factor: np.ndarray | float = 1.0) -> np.ndarray:
    factor = np.ones_like(smooth) * factor if isinstance(factor, float) else factor
    return 2 / (1 + np.exp(-factor * x - smooth)) - 1


def linear(x: pd.DataFrame, factor: np.ndarray, translate: np.ndarray) -> pd.DataFrame:
    return factor * x - factor * translate # type: ignore


class LinearTransfomer:
    """The ScoreTransformer learns the loss distribution of a trained model over the training data
    to translate a loss value into a score between 0 and 1. Here, 0 means the model is working fine,
    while 1 means something is wrong with the input data.
    """
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

        loss = factors * loss - self._noise_floor # type: ignore
        loss = self._scaler.inverse_transform(loss)
        return pd.DataFrame(loss, columns=predicted.columns, index=predicted.index).clip(0, 1)


class SigmoidTransformer:
    """The ScoreTransformer learns the loss distribution of a trained model over the training data
    to translate a loss value into a score between 0 and 1. Here, 0 means the model is working fine,
    while 1 means something is wrong with the input data.

    The ScoreTransformer uses a sigmoidf(x)=1/(1+e^-x) to map the loss values into a score between 0 and 1. First,
    the loss values are standardized using sklearn's StandardScaler. Then, the median of
    the standardized loss values is computed and used to transform the sigmoid on the
    x-axis.

    Args:
        noise_floor (float): The percentile under which a loss will count as 0 (no issue with the prediction).
        upper_bound (float): The upper quantile in percent, required to stretch the sigmoid.
            The smoothing factor defines how high a loss can get before the ScoreTransformer outputs 
            a score of 1. Defaults to 0.999.
    """
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


class ScoreTransformer:
    """The ScoreTransformer learns the loss distribution of a trained model over the training data
    to translate a loss value into a score between 0 and 1. Here, 0 means the model is working fine,
    while 1 means something is wrong with the input data.

    This transformer wraps other transformers. To get more details on these, look into the documentation
    of `LinearTransformer` and `SigmoidTransformer`.

    Args:
        mode (str): Either `linear` or `sigmoid` to specify which backend to use. Defaults to `linear`.
    """
    def __init__(self, mode: str = "linear", *args, **kwargs) -> None:
        supported_modes = ["linear", "sigmoid"]
        assert mode in supported_modes, f"Supported modes are '{supported_modes}'. Unknown mode '{mode}'"
        self._mode = mode

        self._transformer = None
        if self._mode == "linear":
            self._transformer = LinearTransfomer(*args, **kwargs)
        
        if self._mode == "sigmoid":
            self._transformer = SigmoidTransformer(*args, **kwargs)

    def fit(self, predicted: pd.DataFrame, target: pd.DataFrame) -> None:
        """This method fits the parameters of the ScoreTransformer.

        Args:
            predicted (np.array): The predicted values of the model.
            target (np.array): The target values of the model.
        """
        assert self._transformer
        self._transformer.fit(predicted, target)
        logger.debug("Transformer '%s' fitted successfully", self._mode)

    def transform(self, predicted: pd.DataFrame, target: pd.DataFrame, visualize: str = "",
                  show: bool = False, **kwargs) -> pd.DataFrame:
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
        assert self._transformer
        scores = self._transformer.transform(predicted, target, **kwargs)

        # transfer all values to dataframes and return them
        if show or visualize:
            show_dataframe([scores], ["Malfunction"], y_label="% -  lower is better",
                        grid_size=5, y_limit=(0, 1), save=visualize, show=show)
    
        return scores
