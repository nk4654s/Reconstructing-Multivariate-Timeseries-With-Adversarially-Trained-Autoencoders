"""This module removes outliers from the dataframe."""
from pandas import DataFrame
import numpy as np
from S3Prep.modules import BasePreProcessor
from S3Prep.helper import Metadata, get_logger

logger = get_logger(__name__)


class OutlierPreProcessor(BasePreProcessor):
    """This class removes outliers from the dataset using IQR-distance from Median."""
    def __init__(self, lower_quantile: float =0.02, upper_quantile: float = 0.98,
                 threshold: float = 1.5):
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

        # execute after all df are gathered
        self._intermediate_run = False

    def on_download_finished(self, df: DataFrame, _: Metadata = None) -> DataFrame:
        original_len = df.shape[0]
        if df.shape[0] == 0:
            return df

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

        removed_len = df.shape[0] - original_len
        logger.debug("Outlier detection removed %d%% of the DataFrame",
                     round(removed_len / original_len * 100, 3))
        return df
