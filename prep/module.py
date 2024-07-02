"""This module defines the interface for a pre-processor module."""
from typing import Any
import pandas as pd


class BasePreProcessor(object):
    def __init__(self):
        pass

    def __call__(self, df: pd.DataFrame) -> Any:
        raise NotImplementedError()
