"""This module contains the DataFrameCache class, for faster and memory optimized
DataFrame loading."""
from typing import List, Tuple
import numpy as np
import pandas as pd

WEATHER_KEYS = ["Aussentemperatur (°C)"]


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
    def __init__(self, num_frames: int = None, window_size: int = 1, skip: int = 1) -> None:
        self._num_frames = 2**31 - 1 if num_frames is None else num_frames
        self._window_size = window_size
        self._skip = skip
        self._loaded = {}
        self._samples_read = {}
        self._add_history = []
        self._column_names = None
        self._ts_index = None

    def _register(self, key: str, matrix: np.array) -> None:
        self._loaded[key] = matrix
        self._samples_read[key] = self._samples_read.get(key, 0)
        self._add_history.append(key)

        # check if the chache is full
        if len(self._loaded) >= self._num_frames:
            # if the cache is full, remove the dataframes with the highest read count
            # those are less likeley to be used again
            del self._loaded[self._add_history.pop(0)]

    @property
    def columns(self) -> List[str]:
        """The column names of the dataframe (sorted) as list of strings.
        """
        return self._column_names

    @property
    def columns_sensor(self) -> List[str]:
        """The column names of the dataframe's sensors (sorted) as list of strings.
        """
        return [c for c in self._column_names  if c not in WEATHER_KEYS]

    @property
    def ts_index(self) -> int:
        """The time step index of the dataframe' time series.
        """
        if self._ts_index is None and "ts" in self.columns:
            self._ts_index = np.argwhere(np.array(self.columns) == "ts")[0][0]

        return self._ts_index

    def get(self, path: str, key: str = "norm", include_ts: bool = False) -> Tuple[np.array, np.array]:
        """
        Retrieves a DataFrame from the cache.

        Args:
            path (str): The path to the HDF5 file containing the data.
            key (str): The key for the data in the HDF5 file. Defaults to 'norm'.
            include_ts (bool): Whether to include the timestamp column in the returned array. 
                Defaults to False.

        Returns:
            A tuple of two numpy arrays: sensors and weather data.
        """
        data = self._loaded.get(path, None)

        # load and register a new dataframe so it won't loaded again if the cache
        # is not full
        if data is None:
            # read dataframe and sort columns by name for consistency
            df = pd.read_hdf(path, key=key, mode="r")
            df = df.reindex(sorted(df.columns), axis=1)

            # initialize the column names and ts-index value
            if self.columns is None:
                self._column_names = list(df.columns.values)

                # initialize the ts-index value then remove it from the list of
                # feature names, as it is not a feature
                if self.ts_index:
                    self._column_names.remove("ts")

            # get every nth value for weather and sensor data, this reduces the amount of samples
            # held back in memory
            weather = None
            weather_keys = [wk for wk in WEATHER_KEYS if wk in df.columns.values]
            if len(weather_keys) > 0:
                weather = df[weather_keys].to_numpy()[::self._skip]
                sensors = df.drop(WEATHER_KEYS, axis=1)
            
            sensors = df.to_numpy()[::self._skip]
            self._register(path, (sensors, weather))
        else:
            sensors, weather = data

        # remove a dataframe from memory, which was fully read
        self._samples_read[path] += 1
        if self._samples_read[path] == sensors.shape[0] - self._window_size:
            self._add_history.remove(path)
            del self._loaded[path]

        # remove the ts-column from the data if configured
        # make sure to not modify original data, when removing the column
        if not include_ts and self.ts_index:
            sensors = sensors.copy()
            np.delete(sensors, self.ts_index, axis=1)
            return np.delete(sensors, self.ts_index, axis=1), weather
        
        return sensors, weather
