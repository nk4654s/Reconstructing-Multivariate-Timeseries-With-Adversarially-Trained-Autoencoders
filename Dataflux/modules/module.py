"""This module defines the interface for a pre-processor module."""
from enum import Enum
from typing import Callable, List
from pandas import DataFrame
from Dataflux.helper import Metadata


class EntryPoint(Enum):
    """The entry point enum."""

    ON_DOWNLOAD_FINISHED = "on download finished"
    ON_PROCESSING_FINISHED = "on processing finished"


def parse(df: DataFrame | List[DataFrame] | None) -> List[DataFrame]:
    """Ensures that the given dataframe is in a list. If the provided argument
    was a list of dataframes, it will be returned. If the argument is None, an empty list
    will be returned.
    
    Args:
        df (DataFrame | List[DataFrame] | None): A representation of one or multiple dataframes.

    Returns:
        Provided dataframe in a list, or empty.
    """
    if isinstance(df, DataFrame):
        return [df]
    
    if df is None:
        return []
    
    return df


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
        assert metadata
        return df

    def on_processing_finished(self, metadata: Metadata) -> None:
        """Executes the pre-processor after all dataframes were downloaded
        and processed. Useful for modules which require the whole dataset
        to be available.

        Args:
            metadata (Metadata): The metadata dictionary
        """
        assert metadata
        return None

    def _process(
        self, df: DataFrame | List[DataFrame] | None, func: Callable, metadata: Metadata
    ) -> List[DataFrame] | None:
        if df is None:
            func(metadata)
            return None

        # set up the container of processed dataframes, remember the length of the container
        # as this will change during processing
        processed: List[DataFrame] = parse(df)
        pointer = len(processed)

        # process all dataframes in the container
        for i in range(pointer):
            result: DataFrame | None = func(processed[i], metadata)
            if df is not None:
                processed += parse(result)

        # remove old dataframes from the container
        processed = processed[pointer:]
        return processed

    def __call__(
        self,
        df: DataFrame | List[DataFrame] | None,
        metadata: Metadata,
        entry_point: EntryPoint = EntryPoint.ON_PROCESSING_FINISHED,
    ) -> List[DataFrame] | None:
        if entry_point == EntryPoint.ON_DOWNLOAD_FINISHED:
            df = self._process(df, self.on_download_finished, metadata)
            return parse(df)

        self.on_processing_finished(metadata)
        return None
