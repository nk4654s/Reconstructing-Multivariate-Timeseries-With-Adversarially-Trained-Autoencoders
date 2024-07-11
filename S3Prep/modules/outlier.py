"""This module removes outliers from the dataframe."""
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from S3Prep.modules import BasePreProcessor
from S3Prep.helper import Meta, Metadata, get_logger

logger = get_logger(__name__)
rcParams.update({"figure.autolayout": True})

GRAPH_FILE_NAME = "outliers_statistics.png"
GRAPH_TITLE = "Outliers Removed Per Day (%)"
GRAPH_XLABEL = "Date"
GRAPH_YLABEL = "% Outliers Removed"


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

        # collect statistics
        self._stats = {}

    def on_download_finished(self, df: DataFrame, metadata: Metadata = None) -> DataFrame:
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
        logger.debug("Outlier detection removed %d%% of the DataFrame",
                     removed_perc)

        # add outlier statistics to metadata
        self._stats[metadata[Meta.DAY]] = removed_perc
        return df

    def on_processing_finished(self, metadata: Metadata) -> None:
        days = list(self._stats.keys())
        vals = list(self._stats.values())

        # generate graph
        _, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(GRAPH_TITLE)
        ax.plot(days, vals)
        ax.set_xlabel(GRAPH_XLABEL)
        ax.set_ylabel(GRAPH_YLABEL)

        # limit label ticks
        x_locs = np.arange(0, len(days), max(len(days) // 2, len(days) // 50))
        x_labels = [days[i] for i in x_locs]
        ax.set_xticks(x_locs)
        ax.set_xticklabels(x_labels)

        # add graph to metadata
        if metadata.has(Meta.ADDITIONAL_REPORT_GRAPHS):
            container = metadata[Meta.ADDITIONAL_REPORT_GRAPHS]
            container.append((ax, GRAPH_FILE_NAME))
        else:
            metadata.put([(ax, GRAPH_FILE_NAME)], Meta.ADDITIONAL_REPORT_GRAPHS)
