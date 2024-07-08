from typing import Dict, List

from pandas import DataFrame
from S3Prep.modules import BasePreProcessor


class RemoveColumns(BasePreProcessor):
    """This module removes specified columns from the dataframe
    """
    def __init__(self, columns_to_remove: List[str] = ["ts"]) -> None:
        self._columns = columns_to_remove

    def on_processing_started(self, df: DataFrame, metadata: Dict) -> DataFrame | List[DataFrame]:
        df = df.drop(columns=self._columns)
        return df
