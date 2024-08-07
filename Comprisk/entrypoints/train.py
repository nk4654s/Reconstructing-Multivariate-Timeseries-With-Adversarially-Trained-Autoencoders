"""This module creates a model summary and plots it."""

import os
import shutil
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from Dataflux.helper import get_logger

from Comprisk.helper import get_config_path
from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskCLI

logger = get_logger(__name__, path="./logs/logfile")


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
    logger.debug("Using log folder %s", tb_logger.log_dir)

    # copy config to tb_logger.logdir
    os.makedirs(tb_logger.log_dir)
    shutil.copyfile(get_config_path(cli), os.path.join(tb_logger.log_dir, "config.yml"))

    # save model summary to log folder
    with open(
        os.path.join(tb_logger.log_dir, "summary.txt"), mode="w", encoding="UTF-8"
    ) as sum_file:
        sum_file.write(cli.model.summary())

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
