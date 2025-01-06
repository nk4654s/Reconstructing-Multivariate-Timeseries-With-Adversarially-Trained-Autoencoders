from typing import List, Tuple

import numpy as np
import torch


BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]


class MinMaxScaler:
    def __init__(self) -> None:
        self._min = None
        self._max = None
    
    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        self._min = x.min(axis = 0)
        self._max = x.max(axis = 0)
        return self
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        b = self._max - self._min
        return (np.divide((x - self._min), b, np.zeros_like(x), where=b!=0) - 0.5) * 2 
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
    
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x / 2 + 0.5) * (self._max - self._min) + self._min


def online_collate_inference(batch: List[np.ndarray]) -> Tuple[torch.Tensor, MinMaxScaler]:
    """To be used in a torch DataLoader.

    Args:
        batch (List[np.array]): List of numpy arrays.
    
    Returns:
        Tuple[torch.Tensor, Scaler]: Scaled model input and scaler.
    """
    x = np.array([x[0] for x in batch])

    s = MinMaxScaler()
    return (torch.tensor(s.fit_transform(x)), s)


def online_collate_train(batch: List[np.ndarray]) -> BATCH_TYPE:
    """To be used in a torch DataLoader.

    Args:
        batch (List[np.array]): List of numpy arrays.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of tensors as batch.
    """
    x = np.array([x[0] for x in batch])
    y = np.array([x[1] for x in batch])

    s = MinMaxScaler()
    s = s.fit(np.concatenate([x, y]))
    return (torch.tensor(s.transform(x)), torch.tensor(s.transform(y)))


def collate_fn(batch: List[np.ndarray | str]) -> BATCH_TYPE:
    """To be used in a torch DataLoader.

    Args:
        batch (List[np.array]): List of numpy arrays.
    
    Returns:
        Dataflux.pytorch.BATCH_TYPES: Tuple of tensors as batch.
    """
    return (torch.tensor(np.array([x[0] for x in batch])),
            torch.tensor(np.array([x[1] for x in batch])))