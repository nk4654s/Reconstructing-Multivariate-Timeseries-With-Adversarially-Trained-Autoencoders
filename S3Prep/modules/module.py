"""This module defines the interface for a pre-processor module."""
from typing import List
import pandas as pd


class BasePreProcessor(object):
    """The base pre-processor class.
    """
    def __init__(self):
        self._intermediate_run = True

    @property
    def intermediate_run(self) -> bool:
        """Whether the pre-processor is allowed to run intermediately."""
        return self._intermediate_run

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame | List[pd.DataFrame]:
        raise NotImplementedError()

    def __call__(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        df = self.process_df(df)
        return [df] if isinstance(df, pd.DataFrame) else df
