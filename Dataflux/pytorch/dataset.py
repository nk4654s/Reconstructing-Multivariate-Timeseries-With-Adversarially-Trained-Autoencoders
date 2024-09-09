"""This module contains the Dataset class, required to train a NN."""

import pickle
from typing import List, Tuple
import numpy as np
import torch
import pandas as pd

from Dataflux.pytorch.metadata import MetadataFactory, MetadataTypes, Metadata
from Dataflux.helper import Meta, Metadata as FileMetadata
from Dataflux.pytorch.index_tree import IndexTree
from Dataflux.pytorch.cache import DataFrameCache


class Dataset(torch.utils.data.Dataset):
    """
    A custom PyTorch dataset class that loads and processes data from HDF5 files. This class uses
    the DataFrameCache to load dataframes from memory, which can improve performance.

    The dataset outputs sensor and weather data separately, allowing for separate interpretation
    during training. The target value `y` only includes future sensor data, excluding weather data.

    Args:
        path (str): The path to the root folder of the dataset. The dataset folder requires a sub
        folder `metadata` containing the `info.json` file.
        input_sequence (int): The number of time steps for the input sequence.
        output_sequence (int): The number of time steps for the output sequence.
        cache_size (int, optional): The size of the cache. Defaults to None.

    Returns:
        None
    """

    def __init__(
        self,
        path: str,
        input_sequence: int,
        output_sequence: int,
        skip: int = 1,
        cache_size: int = None,
        key: str = "norm"
    ) -> None:
        self._metadata = FileMetadata()
        self._metadata.load(path)
        self._skip = skip
        self._key = key

        self._num_input = input_sequence
        self._num_output = output_sequence
        self._num_features = self._metadata[Meta.FILES][0]["features"]

        self._id = self._metadata[Meta.MACHINE_UUID]
        self._base_dir = path

        data = {"window_size": input_sequence + output_sequence, "skip": skip}
        self._cache = DataFrameCache(cache_size, **data)
        self._tree = IndexTree(self._metadata, **data)

        # build metadata describing the dataset's configuration
        self._meta_factory = MetadataFactory()
        self._meta_factory.set(MetadataTypes.IN_SEQUENCE, self._num_input)
        self._meta_factory.set(MetadataTypes.OUT_SEQUENCE, self._num_output)

        sensors, weather, targets = self[0]
        self._meta_factory.set(MetadataTypes.NAMES, self._cache.columns)
        self._meta_factory.set(MetadataTypes.NAMES_SENSOR, self._cache.columns_sensor)
        self._meta_factory.set(MetadataTypes.SKIP, skip)
        self._meta_factory.set(MetadataTypes.SENSOR, sensors.shape[-1])
        self._meta_factory.set(MetadataTypes.TARGET, targets.shape[-1])
        self._meta_factory.set(MetadataTypes.WEATHER, weather.shape[-1])

    @property
    def UUID(self) -> str:
        """Returns the unique ID of this.
        """
        return self._id

    @property
    def directory(self) -> str:
        """Returns the base directory of this dataset.
        """
        return self._base_dir

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Outputs the shape of this dataset as `number_of_samples`,
        `input_window_size`, `number_of_features`.
        """
        return (len(self._tree), self._num_input, self._num_features)

    def get_meta(self) -> Metadata:
        """Outputs the metadata of the dataset.

        Returns:
            Metadata: The metadata of the dataset.
        """
        return self._meta_factory.build()

    def to_dataframe(self, percentage: float = 1.0) -> pd.DataFrame:
        """This method transforms the dataset into a single dataframe. To avoid
        memory issues, set the percentage parameter accordingly.
        
        Args:
            precentage (float): The percentage of the dataset to put into the
                dataframe. Defaults to 1.0 (100%).
        
        Returns:
            DataFrame: The dataset as dataframe.
        """
        names = self.get_meta().column_names
        data = {key: [] for key in names}

        for i in range(0, int(len(self) * percentage), self._num_input):
            sensors, weather, _ = self[i]

            num_weather = weather.shape[-1]
            for j in range(num_weather):
                data[names[j]] += list(weather[:, j])
            for j in range(num_weather, num_weather + sensors.shape[-1]):
                data[names[j]] += list(sensors[:, j - num_weather])

        return pd.DataFrame(data)

    def scale(self, data: List | np.ndarray | pd.DataFrame) -> pd.DataFrame:
        """Method scales the given data using the normalizer of the dataset.
        The input must match the feature list provided by `self.get_meta().column_names`.

        Args:
            data (List | np.array | pd.DataFrame): The data to scale back.

        Returns:
            pd.DataFrame: The scaled back data.
        """
        m = self._metadata[Meta.NORMALIZER]
        path = m[Meta.NORM_PATH.name]
        with open(path, "rb") as f:
            scaler = pickle.load(f)
            scaler.set_output(transform="pandas")

        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=-1)

        if not isinstance(data, pd.DataFrame):
            data = {key: data[:, i] for i, key in enumerate(self.get_meta().column_names)}
            data = pd.DataFrame(data)

        return scaler.transform(data)

    def scale_back(self, data: List | np.ndarray | pd.DataFrame) -> pd.DataFrame:
        """Method scales back the given data using the normalizer of the dataset.
        The input must match the feature list provided by `self.get_meta().column_names`.

        Args:
            data (List | np.array | pd.DataFrame): The data to scale back.

        Returns:
            pd.DataFrame: The scaled back data.
        """
        m = self._metadata[Meta.NORMALIZER]
        path = m[Meta.NORM_PATH.name]
        with open(path, "rb") as f:
            scaler = pickle.load(f)

        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=-1)

        if not isinstance(data, pd.DataFrame):
            data = {key: data[:, i] for i, key in enumerate(self.get_meta().column_names)}
            data = pd.DataFrame(data)

        scaled = scaler.inverse_transform(data)
        return pd.DataFrame(scaled, columns=self.get_meta().column_names)

    def get_day(self, index: int) -> Tuple[str, float]:
        """Returns the day and hours the machine is on for the provided index
        value.

        Args:
            index (int): The index of the dataset.
        
        Returns:
            Tuple: Day as string and hours the machine is on as float.
        """
        day, index = self._tree.search(index)
        seconds = index * self._skip
        minutes = seconds / 60 
        hour = minutes / 60
        return day, hour

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

        # get sensor/weather data, the cache returns values sorted by the dataframe's
        # column names for consistency
        sensors, weather = self._cache.get(file_path, key=self._key)

        x_1 = sensors[index:start_idx, :]
        x_2 = np.array([], dtype=np.float32) if weather is None else weather[index:start_idx, :]
        y = sensors[start_idx:end_idx, :]

        return x_1, x_2, y


if __name__ == "__main__":
    from tqdm import tqdm, trange
    from torch.utils.data import DataLoader
    from Dataflux.pytorch.utils import collate_fn
    from Dataflux.pytorch.sampler import RandomBatchSampler

    # end of file one: index=22121

    DAY = 24 * 60 * 60
    dataset = Dataset("/Users/jonas/Dataset/dataset-2021-22", DAY, DAY // 4, cache_size=None, skip=4)
    for _ in tqdm(dataset):
        pass

    batch_sampler = RandomBatchSampler(len(dataset), batch_size=32)
    loader = DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=collate_fn
    )

    prev = None
    for _ in trange(0, 5_000, 1, leave=True):
        current = next(loader.__iter__())
        if prev is None:
            prev = current
            continue

        assert prev is not current
        prev = current
