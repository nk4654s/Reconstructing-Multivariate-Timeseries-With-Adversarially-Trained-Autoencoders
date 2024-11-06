"""This module pre-processes a dataframe and insters values to NAN."""

from Dataflux.modules import BasePreProcessor
from Dataflux.helper import Metadata, DataWrapper
from Dataflux import logger


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
        assert (
            method in InterpolatePreProcessor.METHODS
        ), f"{method} is not a valid method."
        self._method = method

    def on_download_finished(self, wrapper: DataWrapper, metadata: Metadata) -> DataWrapper:
        logger.debug("Interpolation method %s", self._method)

        if self._method == "ffill":
            wrapper.df.ffill(inplace=True)

        elif self._method == "bfill":
            wrapper.df.bfill(inplace=True)

        elif self._method in InterpolatePreProcessor.METHODS:
            wrapper.df.interpolate(method=self._method, inplace=True) # type: ignore

        wrapper.df.dropna(how="any", inplace=True)
        return wrapper
