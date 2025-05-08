import os
import shutil
from copy import deepcopy

import lightning as L
import yaml
from Dataflux import logging_mode
from Dataflux.logger import get_logger

from lightning.pytorch.loggers import TensorBoardLogger

from Comprisk.entrypoints.base import (
    EntrypointBase,
    EntrypointContext,
)
from Comprisk.helper import ModelCheckpoint

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


def set_logging_path(context: EntrypointContext) -> str:
    path = os.path.join(context.base_folder, context.environment.value, "models")

    # set up the logger depending on the environment and create a sub folder if a tag was given
    sub_path = context.model.__class__.__name__
    if context.config.tag:
        sub_path = f"{sub_path}/{context.config.tag}"

    # create the tensorboard logger and add it to the trainer
    tb_logger = TensorBoardLogger(path, sub_path)
    context.trainer.logger = tb_logger

    os.makedirs(tb_logger.log_dir)
    logger.debug("Using log folder %s", tb_logger.log_dir)
    return tb_logger.log_dir


class PrepareEntrypoint(EntrypointBase):
    """
    This entrypoints copies the flux configuration to the output directory.

    ```
    model_name/tag/version_num/
    ├── config.yaml
    ├── dataset_train.yaml
    └── dataset_val.yaml
    ```

    Lightning stores the used model configuration file automatically in the model's output folder. Hence, this will
    be handled by the TrainEntrypoint.
    """

    def __init__(self, copy_normalizer: bool = True) -> None:
        super().__init__()

        self._copy_normalizer = copy_normalizer

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        # set up the logging and checkpoint directory
        output_folder = set_logging_path(context)

        # check if a model should be fine-tuned
        checkpoint = ModelCheckpoint(context.config_folder, context.model)
        context.model = checkpoint.get_model()

        # use a different seed then set if the model was loaded from disk
        # this ensures different probabilities (e.g. for the Dropout layer) in the model
        if checkpoint.from_disk:
            context.config.seed_everything += 1
            L.seed_everything(context.config.seed_everything)
            dst = os.path.join(output_folder, "derived_from")
            shutil.copytree(context.config_folder, dst)

        # copy the dataset config of the training data to the model
        src = context.datamodule.trainset.config_file
        dst_train = os.path.join(output_folder, "dataset_train.yaml")
        shutil.copyfile(src, dst_train)

        src = context.datamodule.valset.config_file
        dst_val = os.path.join(output_folder, "dataset_val.yaml")
        shutil.copyfile(src, dst_val)

        # copy the normalizer as is
        if self._copy_normalizer:
            src = os.path.join(context.base_folder, context.environment.value, context.config["normalizer"])
            if not os.path.exists(src):
                # in case the model continues training
                logger.debug("Copying normalizer from previous traing as path %s does not exist.", src)
                src = os.path.join(context.config_folder, "normalizer.yaml")
            dst_norm = os.path.join(output_folder, "normalizer.yaml")
            shutil.copyfile(src, dst_norm)

        # copy the model config
        dst = os.path.join(output_folder, "config.yaml")
        config = deepcopy(vars(context.config))
        del config["config"]
        config["data"]["train"] = "dataset_train.yaml"
        config["data"]["validate"] = "dataset_val.yaml"

        if self._copy_normalizer:
            config["normalizer"] = "normalizer.yaml"

        with open(dst, "w") as f:
            yaml.safe_dump(config, f)

        context.config_folder = output_folder
        return context
