from enum import Enum, auto
from typing import Tuple
import uuid
import numpy as np
import torch

from Dataflux.diskio.meta import Metadata
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
        n_sequence (int): The window or sequence size returned.
        stride (int): The stride factor of the window/sequence.
    """

    def __init__(
        self,
        init_mode: DummyMode,
        n_samples: int,
        features_in: list[str],
        features_out: list[str],
        n_sequence: int,
        n_future: int = 0,
        stride: int = 1,
    ) -> None:
        features_out = features_out or features_in

        # init data
        self._x = np.random.rand(n_samples, len(features_in)).astype(np.float32)
        self._y = np.random.rand(n_samples, len(features_out)).astype(np.float32)

        if init_mode == DummyMode.ZEROS:
            self._x *= 0
            self._y *= 0

        elif init_mode == DummyMode.ONES:
            self._x = np.ones_like(self._x, dtype=np.float32)
            self._y = np.ones_like(self._y, dtype=np.float32)

        # init args
        self._uuid = uuid.uuid4()
        self._args = DatasetArgs(
            sequence_in=n_sequence,
            sequence_out=n_future,
            features_in=len(features_in),
            features_out=len(features_in),
            skip=1,
            stride=stride,
            behavior_=(
                DatasetTypes.AUTOENCODER if n_future == 0 else DatasetTypes.PREDICTIVE
            ),
            names_in=features_in,
            names_out=features_out,
            names=list(set(features_in + features_out)),
        )

    def set_stride(self, stride: int) -> None:
        self._args.stride = stride

    @property
    def UUID(self) -> str:
        """Returns the unique ID of this."""
        return self._uuid

    @property
    def metadata(self) -> Metadata:
        state = self._args.names_in[0]
        return Metadata(
            path="dummy",
            uuid=self.UUID,
            files=[],
            total_recordings=len(self),
            state_alpha={state: {"delay": 1_000, "lower": 0, "upper": 1}},
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self), self._args.sequence_in, self._args.features_in)

    def get_args(self) -> DatasetArgs:
        return self._args.copy()

    def get_timestamp(self, index: int) -> int:
        return index + 1

    def __len__(self) -> int:
        return (len(self._x) - self._args.sequence_in) // self._args.stride

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        i_0 = index * self._args.stride
        i_t = index * self._args.stride + self._args.sequence_in
        i_n = i_t + self._args.sequence_out

        if self._args.sequence_out > 0:
            return self._x[i_0:i_t], self._y[i_t:i_n]

        return self._x[i_0:i_n], self._y[i_0:i_n]


if __name__ == "__main__":
    dataset = DummyDataset(
        DummyMode.RANDOM, n_samples=100, n_features_in=3, n_sequence=10, n_future=1
    )
    print(dataset.metadata)
