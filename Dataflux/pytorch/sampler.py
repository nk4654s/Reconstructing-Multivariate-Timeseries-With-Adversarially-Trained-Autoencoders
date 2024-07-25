"""This module contains a custom sampler, which returns indicies concerning the
split: train 70%, validation 20% and test 10%."""
from torch.utils.data import Sampler
import numpy as np


class RandomBatchSampler(Sampler):
    """
    A custom sampler that generates random batches of indices from a dataset.
    
    Attributes:
        TRAIN (int): Index for training data.
        VALIDATION (int): Index for validation data.
        TEST (int): Index for testing data.
    """
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

    def __init__(
        self,
        data_length: int,
        batch_size: int,
        shuffle: bool = True,
        train_size: float = 0.7,
        validation_size: float = 0.2,
    ):
        assert train_size + validation_size <= 1
        super().__init__()
        self._train = train_size
        self._validation = validation_size
        self._test = 1 - (train_size + validation_size)

        self._data_length = data_length
        self._indicies = np.random.permutation(len(self._data_length)).tolist()
        self._mode_indicies = self._indicies[0 : int(self._data_length * self._train)]

        self._batch_size = batch_size
        self._shuffle = shuffle

    def set_mode(self, mode: int) -> None:
        """
        Sets the mode of the sampler to either TRAIN, VALIDATION, or TEST.
        
        Args:
            mode (int): The mode to set (0 for TRAIN, 1 for VALIDATION, or 2 for TEST).
            
        Raises:
            AssertionError: If the mode is not one of TRAIN, VALIDATION, or TEST.
            AssertionError: If the indices have already been set.
        """
        assert mode in [0, 1, 2], "Mode needs to be TRAIN, VALIDATION, TEST."
        assert self._indicies is not None, "Mode already set."

        if mode == RandomBatchSampler.TRAIN:
            end = int(self._data_length * self._train)
            self._mode_indicies = self._indicies[0:end]
        elif mode == RandomBatchSampler.VALIDATION:
            start = int(self._data_length * self._train)
            end = int(self._data_length * (self._train + self._validation))
            self._mode_indicies = self._indicies[start:end]
        else:
            start = int(self._data_length * (self._train + self._validation))
            self._mode_indicies = self._indicies[start:]

        self._indicies = None

    def __iter__(self):
        for i in range(0, len(self._mode_indicies), self._batch_size):
            yield self._mode_indicies[i : i + self._batch_size]

    def __len__(self):
        return len(self._mode_indicies) // self._batch_size
