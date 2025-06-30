import os

import yaml
from lightning import Trainer, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from Comprisk.custom.components import DataModule
from Comprisk.model import NormalizeWrapper
from Dataflux import get_logger, logging_mode
from Dataflux.pytorch import DatasetArgs

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class CASTrainer(Trainer):
    def __init__(self, config: dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._config = config
        self._setup_complete = False

        self._environment = os.path.join("environment", config["environment"])
        self._log_path = os.path.join(self._environment, "models")

    @property
    def config(self) -> dict:
        return self._config

    def _create_log_folder(self) -> None:
        # create the tensorboard logger and add it to the trainer
        model_name = self._config["model"]["class_path"].rsplit(".", maxsplit=1)[-1]
        tag = self._config["tag"]
        tb_logger = TensorBoardLogger(self._log_path, f"{model_name}/{tag}")
        self.logger = tb_logger

        os.makedirs(tb_logger.log_dir)
        logger.debug("Using log folder %s", tb_logger.log_dir)

    def apply_normalization(self, model: NormalizeWrapper, args: DatasetArgs) -> None:
        path = self._config.get("normalizer", "")
        if path:
            # get the features of the dataset
            meta = {"features": {"input": args.names_in, "output": args.names_out}}
            with open(os.path.join(path), "r") as f:
                meta |= yaml.safe_load(f)

            model.normalization_import(export=meta)

    def fit(
        self,
        model: NormalizeWrapper,
        train_dataloader: TRAIN_DATALOADERS | LightningDataModule | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        if not self._setup_complete:
            self._create_log_folder()

        if isinstance(train_dataloader, DataModule):
            dataset_args = train_dataloader.trainset.get_args()
        else:
            dataset_args = train_dataloader.dataset.get_args()

        self.apply_normalization(model, dataset_args)
        super().fit(model, train_dataloader, val_dataloaders, datamodule, ckpt_path)
