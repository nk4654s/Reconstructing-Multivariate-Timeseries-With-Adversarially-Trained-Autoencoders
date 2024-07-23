"""This module contains the DataFrameCache class, for faster and memory optimized
DataFrame loading."""
from typing import Tuple
import numpy as np
import pandas as pd

WEATHER_KEYS = ["Aussentemperatur (°C)"]


class DataFrameCache:
    """
    A cache for storing and retrieving pandas DataFrames. This class is designed to efficiently
    store and retrieve dataframes from memory. The cache has a limited capacity which defaults
    to #32 Bit frames, but this can be adjusted when initializing the cache.
    """
    def __init__(self, num_frames: int = None) -> None:
        """
        Initializes a DataFrame cache.

        Args:
            num_frames (int): The maximum number of dataframes to store in memory.
                If not provided, defaults to 2^31 - 1, which is the maximum value for a 32-bit
                signed integer.
        
        Returns:
            None
        """
        self._num_frames = 2**31 - 1 if num_frames is None else num_frames
        self._loaded = {}
        self._samples_read = {}

    def _register(self, key: str, matrix: np.array) -> None:
        # check if the chache is full, if not, then load a new dataframe
        if len(self._loaded) < self._num_frames:
            self._loaded[key] = matrix
            self._samples_read[key] = self._samples_read.get(key, 0)
            return None

        # if the cache is full, remove the dataframes with the highest read count
        # those are less likeley to be used again
        max_val = max(self._samples_read.values())
        res = list(
            filter(lambda x: self._samples_read[x] == max_val, self._samples_read)
        )
        del self._loaded[res[0]]

    def get(self, path: str, key: str = "norm") -> Tuple[np.array, np.array]:
        """
        Retrieves a DataFrame from the cache.

        Args:
            path (str): The path to the HDF5 file containing the data.
            key (str): The key for the data in the HDF5 file. Defaults to 'norm'.

        Returns:
            A tuple of two numpy arrays: sensors and weather data.
        """
        data = self._loaded.get(path, None)

        # load and register a new dataframe
        if data is None:
            df = pd.read_hdf(path, key=key, mode="r")
            weather = df[WEATHER_KEYS].to_numpy()
            sensors = df.drop(WEATHER_KEYS, axis=1).to_numpy()

            self._register(path, (sensors, weather))
            self._samples_read[path] += 1
        else:
            sensors, weather = data

        # remove a dataframe, which was fully read
        if self._samples_read[path] == sensors.shape[0]:
            del self._loaded[path]

        return sensors, weather
