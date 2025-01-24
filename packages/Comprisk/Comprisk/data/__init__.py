"""The data module combines multiple classes to load the dataset defined by Dataflux.pytorch.Dataset efficiently.
The Comprisk.data.DataModule class uses PyTorch's dataloaders to enable multithreading of loading a dataset."""
from Comprisk.data.module import DataModule

__all__ = ["DataModule"]
