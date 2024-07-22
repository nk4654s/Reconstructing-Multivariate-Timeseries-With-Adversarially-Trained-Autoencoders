"""This module contains several classes to ensure homogeneous value
distribution across the dataset."""

from typing import List
from pandas import DataFrame, Index
import numpy as np

from S3Prep.data.header import Header
from S3Prep.helper import Meta, Metadata, get_logger, simple_graph
from S3Prep.modules import BasePreProcessor

logger = get_logger(__name__)

GRAPH_HTS_TITLE = "Time Steps Addded (%)"
GRAPH_HTS_XLABEL = "Date"
GRAPH_HTS_YLABEL = "Time Steps Added (%)"
GRAPH_HTS_FILE_NAME = "homogeneous_timesteps.png"

GRAPH_HC_TITLE = "Columns Removed"
GRAPH_HC_XLABEL = "Date"
GRAPH_HC_YLABEL = "# Removed"
GRAPH_HC_FILE_NAME = "homogeneous_columns.png"


class HomogeneousTimeSteps(BasePreProcessor):
    """Ensures homogeneous time steps. Requires index column to be timestamps."""

    def __init__(self, index_key: str = "ts", interpolate: str = "linear"):
        """Initializes the pre-processing module.

        Args:
            index_key (str): Column name of index. Defaults to "ts".
            interpolate (str): Interpolate inserted rows with provided method. None
            means no interpolation. Defaults to 'linear'.
        """
        self._index_key = index_key
        self._interpolate = interpolate
        self._stats = {}

    def on_download_finished(self, df: DataFrame, metadata: Metadata) \
        -> DataFrame | List[DataFrame]:
        # get time start and end
        range_start = df[self._index_key].iloc[0]
        range_end = df[self._index_key].iloc[-1]

        # create a new index based on the start and end range
        new_index = Index(np.arange(range_start, range_end, 1), name=self._index_key)

        # reindex the dataframe, inserted rows will be filled with NaNs
        old_len = df.shape[0]
        df = df.set_index(self._index_key).reindex(new_index).reset_index(self._index_key)
        logger.debug(
            "Reindexed dataframe increased size %s to %s",
            old_len,
            df.shape[0],
        )

        # interpolate added rows
        if self._interpolate:
            df.interpolate(method=self._interpolate, inplace=True)

        self._stats[metadata[Meta.DAY]] = round((df.shape[0] - old_len) / df.shape[0] * 100, 3)
        return df

    def on_processing_finished(self, metadata: Metadata) -> None:
        days = list(self._stats.keys())
        vals = list(self._stats.values())

        # generate graph
        ax = simple_graph(days, vals)(GRAPH_HTS_TITLE, GRAPH_HTS_XLABEL, GRAPH_HTS_YLABEL)

        # add graph to metadata
        metadata.put_list((ax, GRAPH_HTS_FILE_NAME), Meta.ADDITIONAL_REPORT_GRAPHS)


class HomogeneousColumns(BasePreProcessor):
    """Ensures homogeneous columns."""

    def __init__(self, enable: bool = True):
        self._header: Header = None
        self._enable = enable
        self._stats = {}

    def on_download_finished(self, df: DataFrame, metadata: Metadata) \
          -> DataFrame | List[DataFrame]:
        self._stats[metadata[Meta.DAY]] = 0
        if not self._enable:
            return df

        incoming = Header()
        incoming.values = df.columns.values

        # initial dataframe is always homogeneous
        if self._header is None:
            logger.debug("Initial headers: %s", incoming.values)
            self._header = incoming
            return df

        logger.debug("Using headers %s", self._header.values)

        # initial and current dataframe are homogeneous
        if self._header.length == incoming.length:
            return df

        # too many columns, remove the relative complement
        assert self._header.length < incoming.length, (
            "Incoming dataframe has less columns than expected: "
            + f"{incoming.length} to {self._header.length}"
        )

        incoming.relative_complement(self._header)
        if incoming.drop:
            df.drop(columns=incoming.drop, axis=1, inplace=True)
            logger.debug("Removed %s columns", len(incoming.drop))

        self._stats[metadata[Meta.DAY]] = len(incoming.drop)
        return df

    def on_processing_finished(self, metadata: Metadata) -> None:
        days = list(self._stats.keys())
        vals = list(self._stats.values())

        # generate graph
        ax = simple_graph(days, vals)(GRAPH_HC_TITLE, GRAPH_HC_XLABEL, GRAPH_HC_YLABEL)

        # add graph to metadata
        metadata.put_list((ax, GRAPH_HC_FILE_NAME), Meta.ADDITIONAL_REPORT_GRAPHS)
