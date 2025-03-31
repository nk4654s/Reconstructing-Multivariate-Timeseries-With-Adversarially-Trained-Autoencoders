import os
import yaml
from Dataflux import logging_mode
from Dataflux.logger import get_logger
from lightning.pytorch.callbacks.callback import Callback

from Comprisk.entrypoints.base import (
    EntrypointBase,
    EntrypointContext,
)
from Comprisk.helper import BasicCheckpointCallback

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
    ├── model.onnx
    └── config.yaml
    ```

    Lightning stores the used configuration file automatically in the model's output folder. Additionally, the
    entrypoint generates the upload configuration next to it. This file contains information about the latest
    sensor input and pre-processing. When generating the scores in production, this file will be utilized.
    """

    def __init__(self) -> None:
        super().__init__()

        self._checkpoint_path = None
        self._metadata_path = None

    def _set_checkpoint_path(self, context: EntrypointContext) -> None:
        # set up the model saving as configuring the checkpoint path in the yaml config does not
        # work. The config file does not know the model's output folder name with the increasing
        # suffix number
        checkpoint = get_trainer_callback(context.trainer, BasicCheckpointCallback.__name__)
        if checkpoint:
            self._checkpoint_path = os.path.join(context.config_folder, "checkpoints")
            os.makedirs(self._checkpoint_path)
            checkpoint.dirpath = self._checkpoint_path

        else:
            logger.warning("ModelCheckpoint callback not defined in yaml-config.")

    def _import_normalization(self, context) -> None:
        # get the features of the dataset
        args = context.datamodule.trainset.get_args()
        meta = {"features": {"input": args.names_in, "output": args.names_out}}

        # load the normalization
        norm_file = os.path.join(context.config_folder, "normalizer.yaml")
        with open(norm_file, "r") as f:
            meta |= yaml.safe_load(f)

        context.model.normalization_import(export=meta)

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        # checkpoint directory and import the normalization
        self._set_checkpoint_path(context)
        self._import_normalization(context)

        # start the training
        logger.debug("Starting training (seed=%s)", context.config.seed_everything)
        context.trainer.fit(context.model, context.datamodule)

        logger.info(
            "Finished training model %s on machine %s.",
            context.model.__class__.__name__,
            context.datamodule.trainset.UUID,
        )

        context.config.data.shuffle = False
        return context
