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
    def __init__(self, num_frames: int = None, window_size: int = 1) -> None:
        """
        Initializes a DataFrame cache.

        Args:
            num_frames (int): The maximum number of dataframes to store in memory.
                If not provided, defaults to 2^31 - 1, which is the maximum value for a 32-bit
                signed integer.
            window_size (int): The window size of one sample. Defaults to 1.
        
        Returns:
            None
        """
        self._num_frames = 2**31 - 1 if num_frames is None else num_frames
        self._window_size = window_size
        self._loaded = {}
        self._samples_read = {}
        self._add_history = []

    def _register(self, key: str, matrix: np.array) -> None:
        self._loaded[key] = matrix
        self._samples_read[key] = self._samples_read.get(key, 0)
        self._add_history.append(key)

        # check if the chache is full
        if len(self._loaded) >= self._num_frames:
            # if the cache is full, remove the dataframes with the highest read count
            # those are less likeley to be used again
            del self._loaded[self._add_history.pop(0)]

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
        else:
            sensors, weather = data

        # remove a dataframe, which was fully read
        self._samples_read[path] += 1
        if self._samples_read[path] == sensors.shape[0] - self._window_size:
            self._add_history.remove(path)
            del self._loaded[path]

        return sensors, weather
