"""This module contains the score transformer which maps a loss value into a region of 0 to 1."""
import pandas as pd

from Comprisk.anomaly.base_transformer import BaseTransformer
from Comprisk.anomaly.linear import LinearTransformer
from Comprisk.anomaly.sigmoid import SigmoidTransformer
from Comprisk.anomaly.ana import AnaTransformer
from Dataflux.visualization import show_dataframe
from Comprisk import logger


class ScoreTransformer(BaseTransformer):
    """The ScoreTransformer learns the loss distribution of a trained model over the training data
    to translate a loss value into a score between 0 and 1. Here, 0 means the model is working fine,
    while 1 means something is wrong with the input data.

    This transformer wraps other transformers. To get more details on these, look into the documentation
    of `LinearTransformer` and `SigmoidTransformer`.

    Args:
        mode (str): Either `linear`, `sigmoid`, or `anacision` to specify which backend to use. Defaults to `linear`.
    """
    def __init__(self, mode: str = "linear", *args, **kwargs) -> None:
        supported_modes = ["linear", "sigmoid", "anacision"]
        assert mode in supported_modes, f"Supported modes are '{supported_modes}'. Unknown mode '{mode}'"
        self._mode = mode

        self._transformer = None
        if self._mode == "linear":
            self._transformer = LinearTransformer(*args, **kwargs)
        
        if self._mode == "sigmoid":
            self._transformer = SigmoidTransformer(*args, **kwargs)

        if self._mode == "anacision":
            self._transformer = AnaTransformer(*args, **kwargs)

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
