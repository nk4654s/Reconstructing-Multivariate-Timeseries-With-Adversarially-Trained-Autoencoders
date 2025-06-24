"""This module contains helper methods related to the models."""

import os

import torch
from Dataflux import logging_mode
from Dataflux.logger import get_logger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from Comprisk.helper.structure import load_model
from Comprisk.model import BaseModel

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)

CHECKPOINT_FORMAT = ".ckpt"
CHECKPOINT = "checkpoints"


class BasicCheckpointCallback(ModelCheckpoint):
    def __init__(self, monitor: str = "validate/MSELoss"):
        name = monitor
        super().__init__(
            dirpath=CHECKPOINT,
            filename="epoch={epoch}-val={" + name + ":.6f}",
            monitor=name,
            verbose=False,
            save_top_k=3,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_train_steps=None,
            train_time_interval=None,
            every_n_epochs=1,
            enable_version_counter=True
        )
    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save(trainer.model, filepath)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        logger.debug("Saved Checkpoint %s", filepath)


class ModelCheckpoint:
    """This class handles loading the model and related objects/configurations from the model's
    output directory. The loading is handled by Comprisk.helper.structure.load_model

    Args:
        path (str): The path where the model is saved. Defaults to `None`.
        model (BaseModel): The model to use. Defaults to `None`.

    > Note: Make sure to either set `path` or `model`. If both were set, then the provided `model` will be used if there
        is no model in `path/checkpoints`
    """

    def __init__(self, path: str = None, model: BaseModel = None) -> None:
        if not path and not model:
            raise ValueError("Expected to either get a path or model. Got None.")

        self._path = path
        self._from_disk = False
        self._model: BaseModel = model
        self.load_from(path, model)

        if not self._model:
            logger.critical("Unable to load model %s", path)
            raise ValueError("Unable to load model")

        self._configurations = {}
        self.load_configurations()

    def load_from(self, path: str, backup_model: BaseModel = None) -> None:
        """This method tries to load the latest checkpoint from the given folder.

        Args:
            path (str): The model output folder containing a 'checkpoints' subdirectory with
                '.ckpt' files in it.
            backup_model (BaseModel): The model to use when there are no checkpoints.
        """
        self._from_disk = False
        folders = [_ for _ in os.listdir(path)]
        if CHECKPOINT not in folders:
            # no checkpoint folder found
            self._model = backup_model
            logger.warning("Checkpoint folder not found in %s", path)
            return None

        checkpoint_path = os.path.join(path, CHECKPOINT)
        for checkpoint in os.listdir(checkpoint_path):
            if CHECKPOINT_FORMAT in checkpoint:
                # checkpoint folder and checkpoints found, load the best one
                self._model = load_model(path, eval_mode=False)
                self._from_disk = True
                logger.info("ModelCheckpoint initialized from %s", path)
                return None

        # checkpoint folder is empty
        logger.warning("ModelCheckpoint initialized from system memory")
        self._model = backup_model

    def load_configurations(self) -> None:
        """This method checks if a given set of configurations are available in the model output
        folder. The configuration paths are accessible using `get_configuration_path`.
        """
        yaml_files = ["config", "dataset_train", "dataset_val"]
        for yaml_file in yaml_files:
            path = os.path.join(self._path, yaml_file + ".yaml")
            if not os.path.exists(path):
                logger.warning("Unable to find %s", yaml_file + ".yaml")
                continue

            self._configurations[yaml_file] = path

    @property
    def from_disk(self) -> bool:
        return self._from_disk

    @property
    def path(self) -> str:
        """The path where the loaded model was loaded from. The loading process was handled by
        Comprisk.helper.structure.load_model"""
        return self._path

    def get_configuration_path(self, config_name: str) -> str | None:
        """This method returns a configuration path given the name.

        Args:
            config_name (str): The configuration of which the path should be returned.

        Returns:
            str: The configurations path.
        """
        return self._configurations.get(config_name, None)

    def get_model(self) -> BaseModel:
        """A simple getter for the loaded model during object initialization.

        Returns:
            NormalizeModule: The loaded model wrapped by a NormalizeWrapper.
        """
        return self._model
