import os
import shutil
from copy import deepcopy

import yaml
from lightning import Trainer, Callback, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from Comprisk.custom import DataModule
from Comprisk.helper import BasicCheckpointCallback
from Comprisk.model import NormalizeWrapper
from Dataflux import get_logger, logging_mode
from Dataflux.pytorch import DatasetArgs

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


def get_trainer_callback(trainer, callback_name: str) -> Callback | None:
    """
    Finds the given callback class in the trainer callbacks list. If the
    callback could not be found, then the returned value is None.

    Args:
        trainer (Any): The trainer object with the list of configured callbacks.
        callback_name (str): The `lightning.Callback` class.

    Returns:
        Callback: The callback object, if found.
    """
    for c in trainer.callbacks:
        if c.__class__.__name__ == callback_name:
            return c
    return None


class CASTrainer(Trainer):
    def __init__(self, config: dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._config = config
        self._config_folder = self._config.pop("config")
        if os.sep in self._config_folder:
            self._config_folder = self._config_folder.rsplit("/", maxsplit=1)[0]
        self._setup_complete = False

        pre_path = os.path.join("environment", config["environment"])
        self._train_config_path = os.path.join(pre_path, self._config["data"]["train"])
        self._val_config_path = os.path.join(pre_path, self._config["data"]["validate"])
        self._normalizer_config_path = os.path.join(pre_path, self._config["normalizer"])
        self._tuner_config_path = os.path.join(pre_path, str(self._config.get("tuner_config", "N/A")))

        self._log_path = os.path.join(pre_path, "models")

    @property
    def config(self) -> dict:
        return self._config

    def _copy_configs(self) -> None:
        config = deepcopy(self.config)

        # copy the dataset config of the training data to the model
        src = self._train_config_path
        if not os.path.exists(self._train_config_path):
            src = os.path.join(self._config_folder, "train_dataset.yaml")
        shutil.copyfile(src, os.path.join(self.logger.log_dir, "dataset_train.yaml"))
        config["data"]["train"] = "dataset_train.yaml"

        src = self._val_config_path
        if not os.path.exists(src):
            src = os.path.join(self._config_folder, "dataset_val.yaml")
        shutil.copyfile(src, os.path.join(self.logger.log_dir, "dataset_val.yaml"))
        config["data"]["validate"] = "dataset_val.yaml"

        # copy the normalizer as is
        src = self._normalizer_config_path
        if not os.path.exists(src):
            src = os.path.join(self._config_folder, "normalizer.yaml")
        shutil.copyfile(src, os.path.join(self.logger.log_dir, "normalizer.yaml"))
        config["normalizer"] = "normalizer.yaml"

        # copy the tuner as is
        src = self._tuner_config_path
        if not os.path.exists(src):
            src = os.path.join(self._config_folder, "tuner.yaml")
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(self.logger.log_dir, "tuner.yaml"))
                config["tuner_config"] = "tuner.yaml"

        dst = os.path.join(self.logger.log_dir, "config.yaml")
        with open(dst, "w") as f:
            yaml.safe_dump(config, f)

    def _post_init(self) -> None:
        # create the tensorboard logger and add it to the trainer
        model_name = self._config["model"]["class_path"].rsplit(".", maxsplit=1)[-1]
        tag = self._config["tag"]
        tb_logger = TensorBoardLogger(self._log_path, f"{model_name}/{tag}")
        self.logger = tb_logger

        os.makedirs(tb_logger.log_dir)
        logger.debug("Using log folder %s", tb_logger.log_dir)

        # copy all configurations
        self._copy_configs()

        # set up the model saving as configuring the checkpoint path in the yaml config does not
        # work. The config file does not know the model's output folder name with the increasing
        # suffix number
        checkpoint = get_trainer_callback(self, BasicCheckpointCallback.__name__)
        if checkpoint:
            checkpoint_path = os.path.join(tb_logger.log_dir, "checkpoints")
            os.makedirs(checkpoint_path)
            checkpoint.dirpath = checkpoint_path

        else:
            logger.warning("ModelCheckpoint callback not defined in yaml-config.")

    def apply_normalization(self, model: NormalizeWrapper, args: DatasetArgs) -> None:
        if self._config.get("normalizer", None):
            # get the features of the dataset
            meta = {"features": {"input": args.names_in, "output": args.names_out}}

            # load the normalization
            with open(os.path.join(self._normalizer_config_path), "r") as f:
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
            self._post_init()

        if isinstance(train_dataloader, DataModule):
            dataset_args = train_dataloader.trainset.get_args()
        else:
            dataset_args = train_dataloader.dataset.get_args()

        self.apply_normalization(model, dataset_args)
        super().fit(model, train_dataloader, val_dataloaders, datamodule, ckpt_path)
