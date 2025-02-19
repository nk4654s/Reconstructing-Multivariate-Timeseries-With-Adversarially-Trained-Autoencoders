import os
import shutil
import lightning as L
import yaml

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers import TensorBoardLogger

from Comprisk.entrypoints.base import (
    EntrypointBase,
    EntrypointContext,
)

from Dataflux import logging_mode
from Dataflux.logger import get_logger

from Comprisk.helper import ModelCheckpoint


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
        path = os.path.join(context.base_folder, context.environment.value, "models")

        # set up the logger depending on the environment and create a sub folder if a tag was given
        sub_path = context.model.__class__.__name__
        if context.config.tag:
            sub_path = f"{sub_path}/{context.config.tag}"

        # create the tensorboard logger and add it to the trainer
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

    def _import_normalization(self, context) -> None:
        # get the features of the dataset
        args = context.datamodule.trainset.get_args()
        meta = {"features": {"input": args.names_in, "output": args.names_out}}

        # load the normalization
        norm_file = os.path.join(
            context.base_folder, context.environment.value, context.config.normalizer
        )
        with open(norm_file, "r") as f:
            meta |= yaml.safe_load(f)

        context.model.normalization_import(export=meta)

    def _import_dataset_config(self, context: EntrypointContext) -> None:
        # copy the dataset config of the training data to the model
        source = context.datamodule.trainset.config_file
        dest = os.path.join(self._logging_path, "dataset.yaml")
        shutil.copyfile(source, dest)

        # copy the dataset's config file to the model output
        dest = os.path.join(self._logging_path, "upload.yaml")
        shutil.copyfile(source, dest)

        # make sure to have the key values set
        with open(dest, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            config["folder"] = "upload"
            config["latest"] = True
            config.pop("intervals", None)

        with open(dest, mode="w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        # set up the logging and checkpoint directory
        self._set_logging_path(context)
        self._set_checkpoint_path(context)

        # save model metadata to log folder and import the normalization
        self._import_dataset_config(context)
        self._import_normalization(context)

        # check if a model should be fine-tuned
        checkpoint = ModelCheckpoint(context.config_folder, context.model)
        model = checkpoint.get_model()

        # use a different seed then set if the model was loaded from disk
        # this ensures different probabilities (e.g. for the Dropout layer) in the model
        if checkpoint.from_disk:
            context.config.seed_everything += 1
            L.seed_everything(context.config.seed_everything)

        # start the training
        logger.debug("Starting training (seed=%s)", context.config.seed_everything)
        context.trainer.fit(model, context.datamodule)

        # save the normalization after training
        context.model.normalization_save(self._logging_path)

        logger.info(
            "Finished training model %s on machine %s.",
            model.__class__.__name__,
            context.datamodule.trainset.UUID,
        )

        context.model = model
        context.config_folder = self._logging_path
        context.config.data.shuffle = False
        return context
