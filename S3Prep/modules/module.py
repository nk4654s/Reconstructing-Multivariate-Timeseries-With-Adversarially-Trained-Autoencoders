"""This module defines the interface for a pre-processor module."""
import logging
from enum import Enum
from typing import Callable, List, Dict
from pandas import DataFrame

# create logger object
logger = logging.getLogger(__name__)
logging.basicConfig(filename="s3prep.log", encoding="utf-8", level=logging.DEBUG)


class EntryPoint(Enum):
    """The entry point enum.
    """
    ON_DOWNLOAD_FINISHED = 'on download finished'
    ON_PROCESSING_STARTED = 'on processing started'
    ON_PROCESSING_FINISHED = 'on processing finished'


class BasePreProcessor(object):
    """The base pre-processor class.
    """
    def on_download_finished(self, df: DataFrame, metadata: Dict) -> DataFrame | List[DataFrame]:
        """Executes the pre-processor on download finished.

        Args:
            df (pd.DataFrame): The dataframe to be processed
            metadata (dict): The metadata dictionary

        Returns:
            The processed dataframe or a list of dataframes
        """
        return df

    def on_processing_started(self, df: DataFrame, metadata: Dict) -> DataFrame | List[DataFrame]:
        """Executes the pre-processor before downloading a dataframe.

        Args:
            df (pd.DataFrame): The dataframe to be processed
            metadata (dict): The metadata dictionary

        Returns:
            The processed dataframe or a list of dataframes
        """
        return df

    def on_processing_finished(self, metadata: Dict) -> None:
        """Executes the pre-processor after all dataframes were downloaded and processed.

        Args:
            metadata (dict): The metadata dictionary
        """
        return None

    def _process(self, df: DataFrame, func: Callable,
                 metadata: Dict) -> DataFrame | List[DataFrame]:
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

    def __call__(self, df: DataFrame, entry_point: EntryPoint,
                 metadata: Dict = None) -> List[DataFrame]:
        if entry_point == EntryPoint.ON_DOWNLOAD_FINISHED:
            df = self._process(df, self.on_download_finished, metadata)

        elif entry_point == EntryPoint.ON_PROCESSING_STARTED:
            df = self._process(df, self.on_processing_started, metadata)

        elif entry_point == EntryPoint.ON_PROCESSING_FINISHED:
            self.on_processing_finished(metadata)

        return [df] if isinstance(df, DataFrame) else df
