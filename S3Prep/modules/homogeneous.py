"""This module contains several classes to ensure homogeneous value
distribution across the dataset."""

from typing import List
from pandas import DataFrame, Index
import numpy as np
from S3Prep.data.header import Header
from S3Prep.helper import Metadata, get_logger
from S3Prep.modules import BasePreProcessor

logger = get_logger(__name__)


class HomogeneousTimeSteps(BasePreProcessor):
    """Ensures homogeneous time steps. Requires index column to be timestamps."""

    def __init__(self, index_key: str = "ts"):
        """Initializes the pre-processing module.

        Args:
            index_key (str): Column name of index. Defaults to "ts".
        """
        self._index_key = index_key

    def on_processing_started(self, df: DataFrame, _: Metadata) \
        -> DataFrame | List[DataFrame]:
        # get time start and end
        range_start = df[self._index_key].iloc[0]
        range_end = df[self._index_key].iloc[-1]

        # create a new index based on the start and end range
        new_index = Index(np.arange(range_start, range_end, 1), name=self._index_key)

        # reindex the dataframe, inserted rows will be filled with NaNs
        old_len = df.shape[0]
        df = df.set_index(self._index_key).reindex(new_index).reset_index()
        logger.debug(
            "Reindexed dataframe increased size %s to %s",
            old_len,
            df.shape[0],
        )
        return df


class HomogeneousColumns(BasePreProcessor):
    """Ensures homogeneous columns."""

    def __init__(self, enable: bool = True):
        self._header: Header = None
        self._enable = enable

    def on_download_finished(self, df: DataFrame, _: Metadata) \
          -> DataFrame | List[DataFrame]:
        if not self._enable:
            return df

        incoming = Header()
        incoming.values = df.columns.values

        # initial dataframe is always homogeneous
        if self._header is None:
            logger.debug("Initial headers: %s", incoming.length)
            self._header = incoming
            return df

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
        return df
