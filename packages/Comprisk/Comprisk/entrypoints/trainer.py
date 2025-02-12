import os
import shutil
import lightning as L
import yaml

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from Comprisk.entrypoints.base import (
    EntrypointBase,
    EntrypointContext,
)

from Dataflux import logging_mode
from Dataflux.logger import get_logger

from Comprisk.helper import ModelLoader


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


class TrainerEntrypoint(EntrypointBase):
    """
    This entrypoints builds on the FluxEntrypoint and uses the datasets to train a model. First, the model's
    output folders were created. The structure is as follows

    ```
    model_name/tag/version_num/
    ├── checkpoints/
    │   ├── model-0.pkl
    │   └── ...
    ├── metadata/
    │   ├── dataset_config.yaml
    │   └── normalizer.pkl
    ├── config.yaml
    └── upload.yaml
    ```

    Lightning stores the used configuration file automatically in the model's output folder. Additionally, the
    entrypoint generates the upload configuration next to it. This file contains information about the latest
    sensor input and pre-processing. When generating the scores in production, this file will be utilized.
    """
    def __init__(self) -> None:
        super().__init__()

        self._logging_path = None
        self._checkpoint_path = None
        self._metadata_path = None

    def _set_logging_path(self, context: EntrypointContext) -> None:
        # set up the logger depending on the environment
        sub_path = context.model.__class__.__name__
        if context.config.tag:
            sub_path = f"{sub_path}/{context.config.tag}"

        logger.debug("Defining TensorBoard logger for %s", sub_path)
        path = os.path.join(context.base_folder, context.environment.value, "models")
        tb_logger = TensorBoardLogger(path, sub_path)
        self._logging_path = tb_logger.log_dir
        context.trainer.logger = tb_logger
        os.makedirs(tb_logger.log_dir)
        logger.debug("Using log folder %s", tb_logger.log_dir)

    def _set_checkpoint_path(self, context: EntrypointContext) -> None:
        # set up the model saving as configuring the checkpoint path in the yaml config does not
        # work. The config file does not know the model's output folder name with the increasing
        # suffix number
        checkpoint = get_trainer_callback(context.trainer, ModelCheckpoint.__name__)
        if checkpoint:
            self._checkpoint_path = os.path.join(self._logging_path, "checkpoints")
            os.makedirs(self._checkpoint_path)
            checkpoint.dirpath = self._checkpoint_path

        else:
            logger.warning("ModelCheckpoint callback not defined in yaml-config.")

    def _set_metadata_path(self, context: EntrypointContext) -> None:
        # save model metadata to log folder, which should not exist yet
        dataset_path = context.datamodule.trainset.directory
        self._metadata_path = os.path.join(self._logging_path, "metadata")
        os.makedirs(self._metadata_path)

        # copy the normalizer of the training data to the model
        norm_file = os.path.join(dataset_path, "metadata", "normalizer.pkl")
        shutil.copyfile(norm_file, os.path.join(self._metadata_path, "normalizer.pkl"))

        # copy the dataset config of the training data to the model
        config_file = os.path.join(dataset_path, "metadata", "config.yaml")
        shutil.copyfile(
            config_file, os.path.join(self._metadata_path, "dataset_config.yaml")
        )

    def _save_upload_config(self, context: EntrypointContext, source: str = None) -> None:
        if source:
            # a source path is given, the previous model should contain an upload.yaml file
            config_path = os.path.join(source, "upload.yaml")

        elif os.path.isfile(context.config.data.train):
            # copy the training configuration to the output folder
            config_path = os.path.join(context.config.data.train)

        else:
            # the train path is a folder and not a dataflux config file
            config_path = os.path.join(context.config.data.train, "metadata", "config.yaml")

        upload_path = os.path.join(self._logging_path, "upload.yaml")
        shutil.copyfile(config_path, upload_path)
        logger.debug("Copied %s to %s", config_path, upload_path)

        # make sure to have the key values set
        with open(upload_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            config["folder"] = "upload"
            config["latest"] = True
            config.pop("intervals", None)

        with open(upload_path, mode="w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        # set up the logging and checkpoint directory
        self._set_logging_path(context)
        self._set_checkpoint_path(context)

        # save model metadata to log folder
        self._set_metadata_path(context)

        # check if a model should be fine-tuned
        loader = ModelLoader(path=context.config_folder, model=context.model)
        context.model = loader.get_model()
        context.model.train()

        # use a different seed then set if the model was loaded from disk
        # this ensures different probabilities (e.g. for the Dropout layer) in the model
        upload_path = None
        if loader.from_disk:
            context.config.seed_everything += 1
            L.seed_everything(context.config.seed_everything)
            upload_path = loader.path

        self._save_upload_config(context, upload_path)

        # start the training
        logger.debug("Starting training (seed=%s)", context.config.seed_everything)
        context.trainer.fit(context.model, context.datamodule)

        logger.info(
            "Finished training model %s on machine %s.",
            context.model.__class__.__name__,
            context.datamodule.trainset.UUID,
        )

        context.config_folder = self._logging_path
        context.config.data.shuffle = False
        return context
