import os
import shutil

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from Comprisk.entrypoints.base import (
    EntrypointBase,
    EntrypointContext,
    Environment,
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

        # move the upload config if provided
        # when the model was loaded, there should be the upload yaml next to the training config
        upload_config_path = os.path.join(context.config_folder, "upload.yaml")
        if context.config.upload_config and not loader.from_disk:
            upload_config_path = os.path.join(
                context.base_folder,
                context.environment.value,
                context.config.upload_config,
            )

        # if copying the upload.yaml fails, then abort training and cleanup
        try:
            shutil.copyfile(
                os.path.join(upload_config_path),
                os.path.join(self._logging_path, "upload.yaml"),
            )
        except FileNotFoundError as e:
            logger.error(e)
            logger.critical("Could not find upload.yaml in %s", context.config_folder)
            shutil.rmtree(self._logging_path)
            exit(0)

        # start the training
        logger.debug("Starting training")
        context.trainer.fit(context.model, context.datamodule)

        # if a model was fine tuned, make sure to have the same hparams.yaml in the
        # output folder.
        # FIXME: this should not be a problem, this is a workaround
        if context.config.finetune:
            config_path = os.path.dirname(context.config.finetune)
            if os.path.exists(os.path.join(self._logging_path, "hparams.yaml")):
                os.remove(os.path.join(self._logging_path, "hparams.yaml"))
            shutil.copyfile(
                os.path.join(config_path, "hparams.yaml"),
                os.path.join(self._logging_path, "hparams.yaml"),
            )

        logger.info(
            "Finished training model %s on machine %s.",
            context.model.__class__.__name__,
            context.datamodule.trainset.UUID,
        )

        context.config_folder = self._logging_path
        return context
