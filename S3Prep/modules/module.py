"""This module defines the interface for a pre-processor module."""
import pandas as pd


class BasePreProcessor(object):
    """The base pre-processor class.
    """
    def __init__(self):
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
