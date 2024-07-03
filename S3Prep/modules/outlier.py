"""This module removes outliers from the dataframe."""
from pandas import DataFrame
import numpy as np
from . import BasePreProcessor


class OutlierPreProcessor(BasePreProcessor):
    """This class removes outliers from the dataset using IQR-distance from Median."""
    def __init__(self, lower_quantile: float =0.25, upper_quantile: float = 0.75,
                 threshold: float = 1.35):
        """The constructor of the class. This will be automatically executed by
        the CLIArgumentParser.

        Args:
            lower_quantile (float): The lower quantile to be used for outlier removal
            upper_quantile (float): The upper quantile to be used for outlier removal
            threshold (float): The threshold to be used for outlier removal
        """
        super().__init__()
        self._lower = lower_quantile
        self._upper = upper_quantile
        self._threshold = threshold

    def __call__(self, df: DataFrame) -> DataFrame:
        # Calculate the interquartile range (IQR)
        q1 = df.quantile(self._lower)
        q3 = df.quantile(self._upper)
        iqr = q3 - q1

        # Define a threshold for outliers
        threshold = self._threshold * iqr

        # Identify and remove the outliers
        df[(df < (q1 - threshold)) | (df > (q3 + threshold))] = np.nan

        # Remove the outliers from your DataFrame
        df = df.dropna(how="any")

        return df
