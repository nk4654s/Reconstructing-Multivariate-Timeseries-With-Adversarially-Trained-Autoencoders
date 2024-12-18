"""This module contains the BaseTransformer class which describes each ScoreTransformer to be used by a model. The
transformer uses the difference of the predicted and expected model output to calculate a score between 0 and 1."""
import pandas as pd


def loss_fn(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.DataFrame:
    return y - y_hat


class BaseTransformer:
    """The ScoreTransformer learns the loss distribution of a trained model over the training data
    to translate a loss value into a score between 0 and 1. Here, 0 means the model is working fine,
    while 1 means something is wrong with the input data.
    """

    def fit(self, predicted: pd.DataFrame, target: pd.DataFrame) -> "BaseTransformer":
        """This method fits the transformer to the model's output. As a result, the `transform()` method will
        return zeros on the same dataset.

        Args:
            predicted (pd.DataFrame): The model's output.
            target (pd.DataFrame): The model's expected output.

        Returns:
            BaseTransformer: Self, but fitted.
        """
        return self

    def transform(self, predicted: pd.DataFrame, target: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """This method transforms the model's (expected) output to a score between 0 and 1 where 1 is the best case.
        First, the method calculates the difference between the data and then it transforms each feature individually.

        Args:
            predicted (pd.DataFrame): The model's output.
            target (pd.DataFrame): The expected model's output.

        Returns:
            pd.DataFrame: The scores given the input format.
        """
        return pd.DataFrame()
