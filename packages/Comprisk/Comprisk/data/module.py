from copy import deepcopy
from typing import List

import lightning as L
from torch.utils.data import DataLoader

from Dataflux.pytorch import Dataset, collate_fn, DatasetTypes
from Dataflux.pytorch import RandomBatchSampler as RBS
from Dataflux import logging_mode
from Dataflux.logger import get_logger


# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class DataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading and processing data. This class
    uses the `Dataflux.pytorch` module to load a dataset and the `torch.utils.data`
    DataLoader to sample data using multiple threads.

    Args:
        train (str): Path to the dataset root folder. The folder must contain
            many `pandas.DataFrames` stored as `Feather` files next to a sub-folder
            `metadata` which contains the dataset's information.
        validate (str, optional): Path to the validation dataset root folder.
        sequence_in (int): The input sequence length of one sample.
        sequence_out (int): The output sequence length of one sample.
        features_in (List[str]): The amount of input features.
        features_out (List[str]): The amount of output features.
        stride (int): How many stime steps a window should slide.
        behavior (str): Either 'autoencoder' or 'predictive'.
        skip (int): Skip every n-th sample in the dataset.
        batch_size (int): Batch size of the model's input.
        shuffle (bool): Whether to shuffle the dataset or not.
        num_workers (int): Number of workers to use in data loading.
        persistent_workers (bool): Whether to persist the dataset or not.
        pin_memory (bool): Whether to pin memory or not in data loading.
    """

    def __init__(
        self,
        train: str = "",
        validate: str = "",
        sequence_in: int = 1,
        sequence_out: int = 1,
        features_in: List[str] = None,
        features_out: List[str] = None,
        skip: int = 1,
        stride: int = 0,
        behavior: str = DatasetTypes.AUTOENCODER.value,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 1,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        # add public attributes to link them with a model. Check out the CompriskCLI.add_arguments_to_parser()
        # to learn more about this
        self.features_in = sorted(features_in)
        self.features_out = sorted(features_out)
        self.behavior = behavior

        self._loader_kwargs = {
            "num_workers": num_workers,
            "persistent_workers": persistent_workers,
            "pin_memory": pin_memory,
        }

        self._sampler_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
        }

        stride = stride if stride != 0 else sequence_in

        self._dataset_args = {
            "sequence_in": sequence_in,
            "sequence_out": sequence_out,
            "feature_names_in": self.features_in,
            "feature_names_out": self.features_out,
            "stride": stride,
            "skip": skip,
            "behavior": behavior
        }

        self._train = None
        self._val = None
        self._collate_fn = None

        self.set_path(train, validate)

    def set_path(self, train: str, validate: str) -> None:
        """This method sets a new path to the training and validation datasets which
        triggers both to load again. This function will be called by the constructor automatically,
        if the paths are not 'None'.

        Args:
            train (str): The path of the training dataset.
            validate (str): The path of the validation dataset.
        """
        if not train and not validate:
            return

        self._collate_fn = collate_fn
        # online collate normalizes the dataset on the fly
        # self._collate_fn = online_collate_train

        if train:
            self._train = Dataset(
                path=train,
                **self._dataset_args
            )
            logger.debug("Dataset (train) consists of %s samples", len(self._train))

        if validate:
            self._val = Dataset(
                path=validate,
                **self._dataset_args
            )
            logger.debug("Dataset (validate) consists of %s samples", len(self._val))

        logger.debug("Dataset datapoint skip is set to %s", self._dataset_args["skip"])
        logger.debug("Dataset window stride is set to %s", self._dataset_args["stride"])
        logger.debug("Dataset input sequence: %s", self._dataset_args["sequence_in"])
        logger.debug("Dataset output sequence: %s",self._dataset_args ["sequence_out"])

    @property
    def trainset(self) -> Dataset:
        """Returns the loaded training dataset of this data module.

        Returns:
            Dataset: the loaded dataset.
        """
        return self._train

    @property
    def valset(self) -> Dataset:
        """Returns the loaded validation dataset of this data module.

        Returns:
            Dataset: the loaded dataset.
        """
        return self._val

    def _reset_samplers(self) -> None:
        self._train_sampler = None
        self._val_sampler = None

    def setup(self, stage: str) -> None:
        self._reset_samplers()

        # customize sampler kwargs for each dataset (train/validate)
        # hence, copy the kwargs first to prevent changes to the original
        kwargs = deepcopy(self._sampler_kwargs)

        # create new random samplers
        if stage == "fit":
            kwargs["data_length"] = len(self._train)
            self._train_sampler = RBS(**kwargs)

            kwargs["data_length"] = len(self._val)
            self._val_sampler = RBS(**kwargs)

        if stage == "test":
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_sampler=self._train_sampler,
            collate_fn=self._collate_fn,
            **self._loader_kwargs
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_sampler=self._val_sampler,
            collate_fn=self._collate_fn,
            **self._loader_kwargs
        )

    def test_dataloader(self) -> DataLoader:
        return None
