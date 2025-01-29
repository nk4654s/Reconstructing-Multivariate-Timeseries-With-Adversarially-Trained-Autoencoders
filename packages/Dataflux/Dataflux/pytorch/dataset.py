"""This module contains the Dataset class, required to train a NN."""

from typing import List, Tuple
import numpy as np
import torch

from Dataflux.pytorch.args import DatasetArgs, DatasetTypes, dataset_type_from_string
from Dataflux.helper import Meta, Metadata as FileMetadata
from Dataflux.pytorch.index_tree import IndexTree
from Dataflux.pytorch.cache import DataFrameCache

from Dataflux import logger


class EmptyDatasetError(Exception):
    pass


class Dataset(torch.utils.data.Dataset):  # type: ignore
    """
    A custom PyTorch dataset class that loads and processes data from Feather files. This class uses
    the DataFrameCache to load dataframes from memory, which can improve performance.

    Args:
        path (str): The path to the root folder of the dataset. The dataset folder requires a sub
        folder `metadata` containing the `info.json` file.
        sequence_in (int): The number of time steps for the input sequence.
        sequence_out (int): The number of time steps for the output sequence.
        cache_size (int, optional): The size of the cache. Defaults to -1.

    Returns:
        None
    """

    def __init__(
        self,
        path: str,
        sequence_in: int,
        sequence_out: int,
        feature_names_in: List[str] = None,
        feature_names_out: List[str] = None,
        skip: int = 1,
        stride: int = 1,
        cache_size: int = -1,
        behavior: str = DatasetTypes.AUTOENCODER.value,
    ) -> None:
        self._metadata = FileMetadata()
        self._metadata.load(path)
        self._skip = skip
        self._stride = stride
        self._behavior = dataset_type_from_string(behavior)

        self._num_input = sequence_in
        self._num_output = sequence_out
        if len(self._metadata[Meta.FILES]) == 0:
            raise EmptyDatasetError()

        self._features_in = feature_names_in
        self._features_out = feature_names_out
        self._num_features = self._metadata[Meta.FILES][0]["features"]

        if behavior == DatasetTypes.AUTOENCODER:
            self._num_output = 0
            logger.info(
                "The output sequence does not exist when loading the dataset in 'AUTOENCODER' mode. "
                + "Overwriting parameter to have the value '0'"
            )

        self._id = self._metadata[Meta.MACHINE_UUID]
        self._base_dir = path

        # create the custom iterators to navigate through the many dataset files
        # the cache is responsible for loading and storing dataframes in memory,
        # the tree is used to find the correct file based on a given index.
        data = {"window_size": sequence_in + sequence_out, "skip": skip}
        self._cache = DataFrameCache(cache_size, **data)
        self._tree = IndexTree(self._metadata, **data)

        # build metadata describing the dataset's configuration
        # first, get an element from the cache to make sure it is fully initialized
        sensors, targets = self[0]

        self._args = DatasetArgs(
            sequence_in=self._num_input,
            sequence_out=self._num_output,
            features_in=sensors.shape[-1],
            features_out=targets.shape[-1],
            skip=self._skip,
            stride=self._stride,
            behavior_=behavior,
            names=self._cache.columns,
            names_in=feature_names_in,
            names_out=feature_names_out,
        )

    @property
    def UUID(self) -> str:
        """Returns the unique ID of this."""
        return self._id

    @property
    def directory(self) -> str:
        """Returns the base directory of this dataset."""
        return self._base_dir

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Outputs the shape of this dataset as `number_of_samples`,
        `input_window_size`, `number_of_features`.
        """
        return len(self._tree) // self._stride, self._num_input, self._num_features

    def get_args(self) -> DatasetArgs:
        """Outputs the metadata of the dataset.

        Returns:
            Metadata: The metadata of the dataset.
        """
        return self._args.copy()

    def set_stride(self, stride: int) -> None:
        """Sets the stride value for this dataset."""
        assert 0 < stride <= self._num_input
        self._args.stride = stride
        self._stride = stride

    def set_behavior(self, behavior: DatasetTypes | str) -> None:
        """Sets the behavior of this dataset.

        Args:
            behavior (DatasetType | str): The new behavior of this dataset.
        """
        behavior = (
            dataset_type_from_string(behavior)
            if isinstance(behavior, str)
            else behavior
        )
        self._args.behavior = behavior
        self._behavior = behavior

    def get_day(self, index: int) -> Tuple[str, float]:
        """Returns the day and hours the machine is on for the provided index
        value.

        Args:
            index (int): The index of the dataset.

        Returns:
            Tuple: Day as string and hours the machine is on as float.
        """
        if index == -1:
            index = len(self) - 1

        day, index = self._tree.search(index * self._stride)
        seconds = index * self._skip * self._stride
        minutes = seconds / 60
        hour = minutes / 60
        return day, hour

    def get_timestamp(self, index: int) -> int:
        """Returns the timestamp of the provided index value.

        Args:
            index (int): The index of the timestamp.

        Returns:
            int: The timestamp of the provided index value.
        """
        if index == -1:
            index = len(self) - 1

        index *= self._stride

        # find the day and the correct index for that file
        # start of the sequence will be: index + self._num_input ...
        day, index = self._tree.search(index)
        file_path = f"{self._base_dir}/{self._id}_{day}.feather"
        ts = self._cache.get(file_path)["ts"].to_numpy()
        return int(ts[index])

    def _get(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        index *= self._stride

        # find the day and the correct index for that file
        # start of the sequence will be: index + self._num_input ...
        day, index = self._tree.search(index)
        file_path = f"{self._base_dir}/{self._id}_{day}.feather"
        start_idx = index + self._num_input
        end_idx = index + self._num_input + self._num_output

        # get sensor data, the cache returns values sorted by the dataframe's
        # column names for consistency
        sensors = self._cache.get(file_path)

        # create containers for the sensor data
        features_in = self._features_in if self._features_in else self._cache.columns
        features_out = self._features_out if self._features_out else self._cache.columns

        x = np.zeros((self._num_input, len(features_in)), dtype=np.float32)

        for i, name in enumerate(features_in):
            x[:, i] = sensors[name].to_numpy()[index:start_idx]

        # if the dataset will be used to train an autoencoder, then the input should match the output
        if self._behavior == DatasetTypes.AUTOENCODER:
            y = np.zeros((self._num_input, len(features_out)), dtype=np.float32)
            for i, name in enumerate(features_out):
                y[:, i] = sensors[name].to_numpy()[index:start_idx]

            return x, y

        # get only the output features
        y = np.zeros((self._num_output, len(features_out)), dtype=np.float32)
        for i, name in enumerate(features_out):
            y[:, i] = sensors[name].to_numpy()[start_idx:end_idx]

        return x, y

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self):
        index = 0
        while index < len(self):
            yield self._get(index)
            index += 1

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._get(index)


if __name__ == "__main__":
    dataset = Dataset(
        "../../../../environment/dev/datasets/thyssen-vd2-trainset-phase-1",
        50,
        25,
        feature_names_in=[
            "SaugtemperaturErsteStufeTETIT74328 (°C)",
            "Biomethandurchfluss (m^3/min)",
        ],
        feature_names_out=["Biomethandurchfluss (m^3/min)"],
        cache_size=-1,
        skip=1,
    )
    print(dataset.get_args())

    for i in range(len(dataset)):
        dataset[i]
