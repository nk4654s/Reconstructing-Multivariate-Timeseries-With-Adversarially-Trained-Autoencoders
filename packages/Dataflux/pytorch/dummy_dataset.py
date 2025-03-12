from enum import Enum, auto
from typing import Tuple

import numpy as np
import torch

from Dataflux.pytorch import DatasetArgs, DatasetTypes


class DummyMode(Enum):
    """This enum helps to choose the initialization of the DummyDataset."""
    RANDOM = auto()
    ZEROS = auto()
    ONES = auto()


class DummyDataset(torch.utils.data.Dataset):
    """The dummy dataset comes in handy when testing other packages which rely on a dataset with DatasetArgs.

    Args:
        init_mode (DummyMode): The initialization of the dataset's values.
        n_samples (int): The dataset's size.
        n_features (int): The amount of features supported by the dataset.
        n_sequence (int): The window or sequence size returned.
        stride (int): The stride factor of the window/sequence.
    """
    def __init__(
        self,
        init_mode: DummyMode,
        n_samples: int,
        n_features: int,
        n_sequence: int,
        n_future: int = 0,
        stride: int = 1,
    ) -> None:
        # init data
        self._data = np.random.randn(n_samples, n_features)
        if init_mode == DummyMode.ZEROS:
            self._data *= 0

        elif init_mode == DummyMode.ONES:
            self._data = np.ones_like(self._data, dtype=np.float32)

        # init args
        self._args = DatasetArgs(
            sequence_in=n_sequence,
            sequence_out=n_future,
            features_in=n_features,
            features_out=n_features,
            skip=1,
            stride=stride,
            behavior_=DatasetTypes.AUTOENCODER if n_future == 0 else DatasetTypes.PREDICTIVE,
            names_in=[str(i) for i in range(n_features)],
            names_out=[str(i) for i in range(n_features)],
            names=[str(i) for i in range(n_features)],
        )

    def get_args(self) -> DatasetArgs:
        return self._args.copy()

    def get_timestamp(self, index: int) -> int:
        return index + 1

    def __len__(self) -> int:
        return (len(self._data) - self._args.sequence_in) // self._args.stride

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        i_0 = index * self._args.stride
        i_t = index * self._args.stride + self._args.sequence_in
        i_n = index * self._args.stride + self._args.sequence_in + self._args.sequence_out

        if self._args.sequence_out > 0:
            return self._data[i_t:i_n], self._data[i_t:i_n]

        return self._data[i_0:i_n], self._data[i_0:i_n]
