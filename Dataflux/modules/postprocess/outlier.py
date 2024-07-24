"""This module removes outliers from the dataframe."""

from pandas import DataFrame
import numpy as np

from Dataflux.modules import BasePreProcessor
from Dataflux.helper import Meta, Metadata, get_logger, simple_graph

logger = get_logger(__name__)

GRAPH_FILE_NAME = "outliers_statistics.png"
GRAPH_TITLE = "Outliers Removed Per Day (%)"
GRAPH_XLABEL = "Date"
GRAPH_YLABEL = "% Outliers Removed"


class OutlierPreProcessor(BasePreProcessor):
    """This class removes outliers from the dataset using IQR-distance from Median."""

    def __init__(
        self,
        lower_quantile: float = 0.02,
        upper_quantile: float = 0.98,
        threshold: float = 1.5,
    ):
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

        # collect statistics
        self._stats = {}

    def on_download_finished(
        self, df: DataFrame, metadata: Metadata = None
    ) -> DataFrame:
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

        removed_len = original_len - df.shape[0]
        removed_perc = round(removed_len / original_len * 100, 3)
        logger.debug("Outlier detection removed %d%% of the DataFrame", removed_perc)

        # add outlier statistics to metadata
        self._stats[metadata[Meta.DAY]] = removed_perc
        return df

    def on_processing_finished(self, metadata: Metadata) -> None:
        days = list(self._stats.keys())
        vals = list(self._stats.values())

        # generate graph
        ax = simple_graph(days, vals)(GRAPH_TITLE, GRAPH_XLABEL, GRAPH_YLABEL)

        # add graph to metadata
        metadata.put_list((ax, GRAPH_FILE_NAME), Meta.ADDITIONAL_REPORT_GRAPHS)
