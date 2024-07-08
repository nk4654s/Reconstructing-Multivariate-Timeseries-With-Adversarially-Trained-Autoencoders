"""This module pre-processes a dataframe and insters values to NAN."""
from typing import Dict
from pandas import DataFrame
from . import BasePreProcessor
from S3Prep.modules.module import logger


class InterpolatePreProcessor(BasePreProcessor):
    """This class replaces missing values in a dataframe using interpolation methods.

    Possible methods are:
        - "ffill",
        - "bfill",
        - "linear",
        - "quadratic",
        - "cubic",
        - "polynomial".
    """
    METHODS = ["ffill", "bfill", "linear", "quadratic", "cubic", "polynomial"]
    def __init__(self, method: str = "ffill"):
        super().__init__()
        assert method in InterpolatePreProcessor.METHODS, f"{method} is not a valid method."
        self._method = method

    def on_processing_started(self, df: DataFrame, metadata: Dict = None) -> DataFrame:
        logger.debug("Processing ON_PROCESSING_STARTED in %s", self.__class__.__name__)
        logger.debug("Interpolation method: %s", self._method)

        if self._method == "ffill":
            df.ffill(inplace=True)

        elif self._method == "bfill":
            df.bfill(inplace=True)

        elif self._method in InterpolatePreProcessor.METHODS:
            df.interpolate(method=self._method, inplace=True)

        return df.dropna(how="any")
