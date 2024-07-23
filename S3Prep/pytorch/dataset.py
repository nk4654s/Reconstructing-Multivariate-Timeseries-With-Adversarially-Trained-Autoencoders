"""This module contains the Dataset class, required to train a NN."""

from typing import List, Tuple
import numpy as np
import torch
import pandas as pd

from S3Prep.helper import Metadata, Meta
from S3Prep.pytorch.index_tree import IndexTree
from S3Prep.pytorch.cache import DataFrameCache


class Dataset(torch.utils.data.Dataset):
    """
    A custom PyTorch dataset class that loads and processes data from HDF5 files. This class uses
    the DataFrameCache to load dataframes from memory, which can improve performance.

    The dataset outputs sensor and weather data separately, allowing for separate interpretation
    during training. The target value `y` only includes future sensor data, excluding weather data.

    Args:
        path (str): The path to the HDF5 file containing the data.
        input_sequence (int): The number of time steps for the input sequence.
        output_sequence (int): The number of time steps for the output sequence.
        chache_size (int, optional): The size of the cache. Defaults to None.

    Returns:
        None
    """

    def __init__(
        self,
        path: str,
        input_sequence: int,
        output_sequence: int,
        chache_size: int = None,
    ) -> None:
        self._metadata = Metadata()
        self._metadata.load(path)
        self._tree = IndexTree(
            self._metadata, window_size=input_sequence + output_sequence
        )

        self._num_input = input_sequence
        self._num_output = output_sequence
        self._num_features = self._metadata[Meta.FILES][0]["features"]

        self._id = self._metadata[Meta.MACHINE_UUID]
        self._base_dir = path
        self._df = pd.read_hdf(f"{path}/full.h5", key="norm", mode="r")
        self._cache = DataFrameCache(chache_size, self._num_input + self._num_output)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Outputs the shape of this dataset as `number_of_samples`,
        `input_window_size`, `number_of_features`.
        """
        return (len(self._tree), self._num_input, self._num_features)

    def scale_back(self, data: List | np.ndarray | pd.DataFrame) -> pd.DataFrame:
        """Method scales back the given data using the normalizer of the dataset.

        Args:
            data (List | np.array | pd.DataFrame): The data to scale back.

        Returns:
            pd.DataFrame: The scaled back data.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self):
        index = 0
        while index < len(self):
            yield self[index]
            index += 1

    def __getitem__(self, index: int) -> np.array:
        # find the day and the correct index for that file
        # start of the sequence will be: index + self._num_input ...
        day, index = self._tree.search(index)
        file_path = f"{self._base_dir}/{self._id}_{day}.h5"
        start_idx = index + self._num_input
        end_idx = index + self._num_input + self._num_output

        sensors, weather = self._cache.get(file_path, key="norm")
        assert end_idx < len(sensors)

        x_1 = sensors[index:start_idx, :]
        x_2 = weather[index:start_idx, :]
        y = sensors[start_idx:end_idx, :]

        return x_1, x_2, y


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from S3Prep.pytorch.utils import collate_fn

    # end of file one: index=22121

    DAY = 24 * 60 * 60
    dataset = Dataset("dataset-2021-22", DAY, DAY // 4, chache_size=None)
    # for _ in tqdm(dataset):
    #     pass

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1, collate_fn=collate_fn)
    for _ in tqdm(loader):
        pass
