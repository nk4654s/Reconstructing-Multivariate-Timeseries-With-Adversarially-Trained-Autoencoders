"""This module removes rows from the dataframe if the machine is off."""
from typing import List, Tuple
from pandas import DataFrame
import numpy as np
from . import BasePreProcessor


class SplitByMachineOff(BasePreProcessor):
    """This class removes rows from the dataframe if the machine is off."""
    def __init__(self, on_key: str = "Laeuft"):
        """The constructor of the class. This will be automatically executed by
        the CLIArgumentParser.

        Args:
            on_key (str): The key of the column that contains the machine state.
            Defaults to "Laeuft".
        """
        super().__init__()
        self._column_key = on_key

    def _trim_off(self, df: DataFrame) -> DataFrame:
        running = df[self._column_key].to_numpy()
        start_idx = np.argmax(running > 0)
        return df.iloc[start_idx:]

    def _split_off(self, df: DataFrame) -> Tuple[DataFrame]:
        running = df[self._column_key].to_numpy()
        end_idx = np.argmax(running < 1)
        return df.iloc[:end_idx], df.iloc[end_idx:]

    def process_df(self, df: DataFrame) -> List[DataFrame]:
        df_splits = []
        to_split = df.copy()
        trimmed = to_split
        while not trimmed.empty:
            trimmed = self._trim_off(to_split)
            trimmed, to_split = self._split_off(trimmed)
            if not trimmed.empty:
                df_splits.append(trimmed)

        df_splits.append(to_split)
        return df_splits
