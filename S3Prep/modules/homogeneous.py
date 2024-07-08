from typing import Dict, List
from pandas import DataFrame, Index
import numpy as np
from S3Prep.modules import BasePreProcessor


class HomogeneousTimeSteps(BasePreProcessor):
    """Ensures homogeneous time steps. Requires index column to be timestamps.
    """
    def __init__(self, index_key: str = "ts"):
        """Initializes the pre-processing module.

        Args:
            index_key (str): Column name of index. Defaults to "ts".
        """
        self._index_key = index_key

    def on_processing_started(self, df: DataFrame, metadata: Dict) -> DataFrame | List[DataFrame]:
        # get time start and end
        range_start = df[self._index_key].iloc[0]
        range_end = df[self._index_key].iloc[-1]

        # create a new index based on the start and end range
        new_index = Index(np.arange(range_start, range_end, 1), name=self._index_key)

        # reindex the dataframe, inserted rows will be filled with NaNs
        df = df.reindex(new_index)
        return df
