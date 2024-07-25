"""This module manages multiple datasets and loads them in the required training stages."""

from typing import Dict
import lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

from Dataflux.pytorch import Dataset, collate_fn
from Dataflux.pytorch import RandomBatchSampler as RBS


class DataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading and processing data.

    Args:
        data_kwargs (Dict): Keyword arguments for initializing the dataset.
        loader_kwargs (Dict): Keyword arguments for initializing the data loader.
        batch_size (int, optional): The size of each mini-batch. Defaults to 64.
    """

    def __init__(
        self, data_kwargs: Dict, loader_kwargs: Dict, sampler_kwargs: Dict
    ) -> None:
        super().__init__()

        self._loader_kwargs = loader_kwargs
        self._sampler_kwargs = sampler_kwargs
        self._data_kwargs = data_kwargs

        self._dataset = Dataset(**data_kwargs)
        self._sampler_kwargs["data_length"] = len(self._dataset)

        self._train = None
        self._val = None
        self._test = None

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def _free_memory(self) -> None:
        self._train = None
        self._val = None
        self._test = None

    def setup(self, stage: str) -> None:
        # assert stage in ["fit", "test"], "Unsopported stage, supported are [fit, test]."
        self._free_memory()

        # load the datasets
        if stage == "fit":
            self._train = RBS(**self._sampler_kwargs)
            self._train.set_mode(RBS.TRAIN)

            self._val = RBS(**self._sampler_kwargs)
            self._val.set_mode(RBS.VALIDATION)

        if stage == "test":
            self._val = RBS(**self._sampler_kwargs)
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
