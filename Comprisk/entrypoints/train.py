"""This module creates a model summary and plots it."""

import os
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from Dataflux.helper import get_logger

from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskCLI

logger = get_logger(__name__, path="./logs/logfile")

def get_trainer_callback(trainer, callback_cls: Callback) -> Callback:
    """Finds the fiven callback class in the trainer callbacks list. If the
    callback could not be found, then the returned value is None.
    
    Args:
        trainer (Any): The trainer object with the list of configured callbacks.
        callback_cls (Callback): The `lightning.Callback` class.
    
    Returns:
        Callback: The callback object, if found.
    """
    for c in trainer.callbacks:
        if isinstance(c, callback_cls):
            return c
    return None


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    logger.debug("Starting CompriskCLI")
    cli = CompriskCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    # set up the logger
    sub_path = cli.model.__class__.__name__
    if cli.config.tag:
        sub_path = f"{cli.model.__class__.__name__}/{cli.config.tag}"

    logger.debug("Defining TensorBoard logger for %s", sub_path)
    tb_logger = TensorBoardLogger("models", name=sub_path)
    cli.trainer.logger = tb_logger
    os.makedirs(tb_logger.log_dir)
    logger.debug("Using log folder %s", tb_logger.log_dir)

    # set up the model saving
    checkpoint = get_trainer_callback(cli.trainer, ModelCheckpoint)
    if checkpoint:
        os.makedirs(tb_logger.log_dir + "/checkpoints")
        checkpoint.dirpath = tb_logger.log_dir + "/checkpoints"
    else:
        logger.warning("ModelCheckpoint callback not defined in yaml-config.")

    # save model summary to log folder
    summary_path = os.path.join(tb_logger.log_dir, "summary.txt")
    with open(summary_path, mode="w", encoding="UTF-8") as sum_file:
        meta = cli.datamodule.dataset.get_meta()
        sum_file.write(cli.model.summary(meta))

    # run methods before fit
    logger.debug("CompriskCLI prepare training.")
    cli.before_fit()

    # train
    logger.debug("Starting training")
    model = cli.model
    if model.device == "cuda":
        logger.debug("Compile model for faster training speeds")
        model = torch.compile(cli.model)
    cli.trainer.fit(model, cli.datamodule)

    # run methods after fit
    logger.debug("Training finished, clean up now.")
    cli.after_fit()


if __name__ == "__main__":
    cli_main()
