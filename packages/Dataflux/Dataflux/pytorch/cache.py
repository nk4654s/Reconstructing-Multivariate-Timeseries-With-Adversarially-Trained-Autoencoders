"""This module contains the DataFrameCache class, for faster and memory optimized
DataFrame loading."""
from collections import OrderedDict
from typing import List, Dict
import numpy as np
import pandas as pd


class DataFrameCache:
    """
    A cache for storing and retrieving pandas DataFrames. This class is designed to efficiently
    store and retrieve dataframes from memory. The cache has a limited capacity which defaults
    to #32 Bit frames, but this can be adjusted when initializing the cache.

    Args:
        num_frames (int): The maximum number of dataframes to store in memory.
            If not provided, defaults to 2^31 - 1, which is the maximum value for a 32-bit
            signed integer.
        window_size (int): The window size of one sample. Defaults to 1.
        skip (int): The number of samples to skip between each sample. Defaults to 1.
    """
    def __init__(self, num_frames: int = -1, window_size: int = 1, skip: int = 1) -> None:
        self._num_frames = 2**31 - 1 if num_frames < 0 else num_frames
        self._window_size = window_size
        self._skip = skip
        self._loaded: Dict[str, OrderedDict[str, pd.Series]] = {}
        self._samples_read = {}
        self._add_history = []
        self._column_names = None

    def _register(self, key: str, matrix: OrderedDict[str, pd.Series]) -> None:
        self._loaded[key] = matrix
        self._samples_read[key] = self._samples_read.get(key, 0)
        self._add_history.append(key)

        # check if the cache is full
        if len(self._loaded) >= self._num_frames:
            # if the cache is full, remove the dataframes with the highest read count
            # those are less likely to be used again
            del self._loaded[self._add_history.pop(0)]

    @property
    def columns(self) -> List[str]:
        """The column names of the dataframe (sorted) as list of strings.
        """
        assert self._column_names, "Column names available after calling .get() for the first time."
        return self._column_names

    def get(self, path: str, key: str = "norm") -> OrderedDict[str, pd.Series]:
        """
        Retrieves a DataFrame from the cache.

        Args:
            path (str): The path to the HDF5 file containing the data.
            key (str): The key for the data in the HDF5 file. Defaults to 'norm'.

        Returns:
            A tuple of two numpy arrays: sensors data as numpy matrix, but the columns were sorted by their
                header name before.
        """
        data = self._loaded.get(path, None)

        # load and register a new dataframe so it won't load again if the cache
        # is not full
        if data is None:
            # read dataframe and sort columns by name for consistency
            df = pd.read_hdf(path, key=key, mode="r")
            df = df.reindex(sorted(df.columns), axis=1) # type: ignore

            # initialize the column names and ts-index value
            if self._column_names is None:
                self._column_names = list(df.columns.values)
                assert "ts" in self._column_names, "'ts' not a column in dataset"
                self._column_names.remove("ts")

            # get every nth value for weather and sensor data, this reduces the amount of samples
            # held back in memory
            sensors = df.to_dict(orient="series", into=OrderedDict)
            for name in self.columns:
                sensors[name] = sensors[name][::self._skip].astype(np.float32)

            self._register(path, sensors)
        else:
            sensors = data

        # remove a dataframe from memory, which was fully read
        self._samples_read[path] += 1
        if self._samples_read[path] == len(sensors["ts"]) - self._window_size:
            self._add_history.remove(path)
            del self._loaded[path]
        
        return sensors
