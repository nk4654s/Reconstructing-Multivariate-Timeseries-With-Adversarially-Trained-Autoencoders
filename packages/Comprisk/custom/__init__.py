"""The data module combines multiple classes to load the dataset defined by Dataflux.pytorch.Dataset efficiently.
The Comprisk.data.DataModule class uses PyTorch's dataloaders to enable multithreading of loading a dataset.
"""

from Comprisk.custom.components import DataModule, CASTrainer
from Comprisk.custom.cli import CompriskCLI


__all__ = ["CASTrainer", "CompriskCLI", "DataModule"]
