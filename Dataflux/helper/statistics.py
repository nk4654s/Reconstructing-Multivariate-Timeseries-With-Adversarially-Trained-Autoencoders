from typing import Union
import numpy as np
import pandas as pd


class StatisticsRecorder:
    """This class keeps track of the minimum and maximum values of a given
    dataset over time. Then, the recorded values can be accessed by
    calling the `min` and `max` attributes. Also, this class can normalize a
    DataFrame to a range between [-1, 1].
    """
    def __init__(self):
        self._min = None
        self._max = None
    
    def data(self, data: Union[np.array, pd.DataFrame]) -> None:
        """This method records the minimum and maximum values of a given
        dataset.

        Args:
            data (np.array or pd.DataFrame): The dataset to record.
        """
        if isinstance(data, pd.DataFrame):
            data = data.reindex(sorted(data.columns), axis=1)
            data = data.to_numpy()

        if self._max is None:
            self._min = data.min(0)
            self._max = data.max(0)
        else:
            self._min[self._min > data.min(0)] = data.min(0)[self._min > data.min(0)]
            self._max[self._max < data.max(0)] = data.max(0)[self._max < data.max(0)]

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """This method normalizes a dataframe to a range between [-1, 1].

        Args:
            data (pd.DataFrame): The dataframe to normalize.
        
        Returns:
            pd.DataFrame: The normalized dataframe.
        """
        return (((data - self._min) / (self._max - self._min) - 0.5) / 0.5).fillna(0)

    @property
    def min(self) -> np.array:
        """The minimum values of the dataset recorded by this class."""
        return self._min
    
    @property
    def max(self) -> np.array:
        """The maxium values of the dataset recorded by this class."""
        return self._max
