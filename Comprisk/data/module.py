"""This module manages multiple datasets and loads them in the required
training stages. Moreover, the module relies on the `Dataflux` package
which downloads and pre-processes a dataset. Also, the `Dataflux`
package provides classes and methods to load a processed dataset and
to sample as efficient as possible."""

from copy import deepcopy
import lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

from Dataflux.pytorch import Dataset, collate_fn
from Dataflux.pytorch import RandomBatchSampler as RBS

from Comprisk import logger


class DataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading and processing data. This class
    uses the `Dataflux.pytorch` module to load a dataset and the `torch.utils.data`
    DataLoader to sample data using multiple threads.

    Args:
        path (str): Path to the dataset root folder. The folder must contain
            many `pandas.DataFrames` stored as `HDF5` files next to a sub-folder
            `metadata` which contains the dataset's information.
        input_sequence (int): The input sequence length of one sample.
        output_sequence (int): The output sequence length of one sample.
        skip (int): Skip every n-th sample in the dataset.
        batch_size (int): Batch size of the model's input.
        shuffle (bool): Whether to shuffle the dataset or not.
        train_size (float): Percentage of the dataset to be used for training.
        validation_size (float): Percentage of the dataset to be used for validation.
        num_workers (int): Number of workers to use in data loading.
        persist (bool): Whether to persist the dataset or not.
        pin_memory (bool): Whether to pin memory or not in data loading.
    """

    def __init__(
        self,
        path: str = None,
        input_sequence: int = 1,
        output_sequence: int = 1,
        skip: int = 1,
        batch_size: int = 1,
        shuffle: bool = True,
        train_size: float = 0.7,
        validation_size: float = 0.2,
        num_workers: int = 1,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self._loader_kwargs = {
            "num_workers": num_workers,
            "persistent_workers": persistent_workers,
            "pin_memory": pin_memory,
        }

        self._sampler_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "train_size": train_size,
            "validation_size": validation_size,
        }

        self._dataset = Dataset(
            path=path,
            input_sequence=input_sequence,
            output_sequence=output_sequence,
            skip=skip,
        )

        logger.debug("Dataset consists of %s samples", len(self._dataset))
        logger.debug("Dataset skip is set to %s", skip)
        logger.debug("Dataset input sequence: %s", input_sequence)
        logger.debug("Dataset output sequence: %s", output_sequence)

        self._sampler_kwargs["data_length"] = len(self._dataset)

        self._train = None
        self._val = None
        self._test = None

    @property
    def dataset(self) -> Dataset:
        """Returns the loaded dataset of this data module.

        Returns:
            Dataset: the loaded dataset.
        """
        return self._dataset

    def _free_memory(self) -> None:
        self._train = None
        self._val = None
        self._test = None

    def setup(self, stage: str, train_size: float = None, val_size: float = None) -> None:
        # assert stage in ["fit", "test"], "Unsopported stage, supported are [fit, test]."
        self._free_memory()

        # prevent changes to the original kwargs
        kwargs = deepcopy(self._sampler_kwargs)
        if train_size:
            kwargs["train_size"] = train_size

        if val_size:
            kwargs["validation_size"] = val_size

        # load the datasets
        if stage == "fit":
            self._train = RBS(**kwargs)
            self._train.set_mode(RBS.TRAIN)

            self._val = RBS(**kwargs)
            self._val.set_mode(RBS.VALIDATION)

        if stage == "test":
            self._val = RBS(**kwargs)
            self._val.set_mode(RBS.TEST)

    def train_dataloader(self) -> CombinedLoader:
        return DataLoader(
            self._dataset,
            batch_sampler=self._train,
            collate_fn=collate_fn,
            **self._loader_kwargs
        )

    def val_dataloader(self) -> CombinedLoader:
        return DataLoader(
            self._dataset,
            batch_sampler=self._val,
            collate_fn=collate_fn,
            **self._loader_kwargs
        )

    def test_dataloader(self) -> CombinedLoader:
        return DataLoader(
            self._dataset,
            batch_sampler=self._test,
            collate_fn=collate_fn,
            **self._loader_kwargs
        )
