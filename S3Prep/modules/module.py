"""This module defines the interface for a pre-processor module."""
from enum import Enum
from typing import Callable, List
from pandas import DataFrame
from S3Prep.helper import Metadata, get_logger

# create logger object
logger = get_logger(__name__)


class EntryPoint(Enum):
    """The entry point enum."""

    ON_DOWNLOAD_FINISHED = "on download finished"
    ON_PROCESSING_FINISHED = "on processing finished"


class BasePreProcessor(object):
    """The base pre-processor class."""

    def on_download_finished(
        self, df: DataFrame, metadata: Metadata
    ) -> DataFrame | List[DataFrame]:
        """Executes the pre-processor on download finished.

        Args:
            df (pd.DataFrame): The dataframe to be processed
            metadata (Metadata): The metadata dictionary

        Returns:
            The processed dataframe or a list of dataframes
        """
        return df

    def on_processing_finished(self, metadata: Metadata) -> None:
        """Executes the pre-processor after all dataframes were downloaded
        and processed. Useful for modules which require the whole dataset
        to be available.

        Args:
            metadata (Metadata): The metadata dictionary
        """
        return None

    def _process(
        self, df: DataFrame, func: Callable, metadata: Metadata
    ) -> DataFrame | List[DataFrame]:
        if df is None:
            func(metadata)
            return None

        # set up the container of processed dataframes, remember the length of the container
        # as this will change during processing
        processed = [df] if isinstance(df, DataFrame) else df
        pointer = len(processed)

        # process all dataframes in the container
        for i in range(pointer):
            df = func(processed[i], metadata)
            processed += [df] if isinstance(df, DataFrame) else df

        # remove old dataframes from the container
        processed = processed[pointer:]
        return processed

    def __call__(
        self,
        df: DataFrame = None,
        entry_point: EntryPoint = EntryPoint.ON_PROCESSING_FINISHED,
        metadata: Metadata = None,
    ) -> List[DataFrame]:
        if entry_point == EntryPoint.ON_DOWNLOAD_FINISHED:
            df = self._process(df, self.on_download_finished, metadata)

        elif entry_point == EntryPoint.ON_PROCESSING_FINISHED:
            self.on_processing_finished(metadata)

        return [df] if isinstance(df, DataFrame) else df
