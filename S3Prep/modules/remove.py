from typing import List

from pandas import DataFrame
from S3Prep.modules import BasePreProcessor
from S3Prep.helper import Metadata, get_logger

logger = get_logger(__name__)


class RemoveColumns(BasePreProcessor):
    """This module removes specified columns from the dataframe
    """
    def __init__(self, columns_to_remove: List[str] = ["ts"]) -> None:
        self._columns = columns_to_remove

    def on_download_finished(self, df: DataFrame, _: Metadata) -> DataFrame | List[DataFrame]:
        old_len = df.shape[1]
        df = df.drop(columns=self._columns)
        logger.debug("Removed %s columns", old_len - df.shape[1])
        return df
