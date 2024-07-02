from typing import Any
from pandas import DataFrame
from .module import BasePreProcessor


class InterpolatePreProcessor(BasePreProcessor):
    METHODS = ["ffill", "bfill", "linear", "quadratic", "cubic", "polynomial"]
    def __init__(self, method: str = "ffill"):
        assert method in InterpolatePreProcessor.METHODS, f"{method} is not a valid method."
        self._method = method

    def __call__(self, df: DataFrame) -> Any:
        if self._method == "ffill":
            df.ffill(inplace=True)
        elif self._method == "bfill":
            df.bfill(inplace=True)
        elif self._method in InterpolatePreProcessor.METHODS:
            df.interpolate(method=self._method, inplace=True)
