"""This module defines the interface for a pre-processor module."""
from enum import Enum
from typing import Callable, List
from Dataflux.helper import Metadata, DataWrapper


class EntryPoint(Enum):
    """The entry point enum."""

    ON_DOWNLOAD_FINISHED = "on download finished"
    ON_PROCESSING_FINISHED = "on processing finished"


def parse(df: DataWrapper | List[DataWrapper] | None) -> List[DataWrapper]:
    """Ensures that the given dataframe is in a list. If the provided argument
    was a list of dataframes, it will be returned. If the argument is None, an empty list
    will be returned.
    
    Args:
        df (DataFrame | List[DataFrame] | None): A representation of one or multiple dataframes.

    Returns:
        Provided dataframe in a list, or empty.
    """
    if isinstance(df, DataWrapper):
        return [df]
    
    if df is None:
        return []
    
    return df


class BasePreProcessor(object):
    """The base pre-processor class."""

    def on_download_finished(
        self, wrapper: DataWrapper, metadata: Metadata
    ) -> DataWrapper | List[DataWrapper]:
        """Executes the pre-processor on download finished.

        Args:
            wrapper (pd.DataFrame): The dataframe to be processed
            metadata (Metadata): The metadata dictionary

        Returns:
            The processed dataframe or a list of dataframes
        """
        assert metadata
        return wrapper

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
        self, df: DataWrapper | List[DataWrapper] | None, func: Callable, metadata: Metadata
    ) -> List[DataWrapper] | None:
        if df is None:
            func(metadata)
            return None

        # set up the container of processed dataframes, remember the length of the container
        # as this will change during processing
        processed: List[DataWrapper] = parse(df)
        pointer = len(processed)

        # process all dataframes in the container
        for i in range(pointer):
            result: DataWrapper | None = func(processed[i], metadata)
            if df is not None:
                processed += parse(result)

        # remove old dataframes from the container
        processed = processed[pointer:]
        return processed

    def __call__(
        self,
        df: DataWrapper | List[DataWrapper] | None,
        metadata: Metadata,
        entry_point: EntryPoint = EntryPoint.ON_PROCESSING_FINISHED,
    ) -> List[DataWrapper]:
        if entry_point == EntryPoint.ON_DOWNLOAD_FINISHED:
            df = self._process(df, self.on_download_finished, metadata)
            return parse(df)

        self.on_processing_finished(metadata)
        return []
