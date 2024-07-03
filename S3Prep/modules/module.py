"""This module defines the interface for a pre-processor module."""
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

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
