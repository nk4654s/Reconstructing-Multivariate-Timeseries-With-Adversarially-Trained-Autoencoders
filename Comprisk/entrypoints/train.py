"""This module creates a model summary and plots it."""

import os
import json
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from Dataflux.pytorch import Dataset

from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskCLI

from Comprisk import logger

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


def save_training_meta(base_path: str, dataset: Dataset) -> None:
    """This function creates the model's metadata folder and generates the
    training dataset correlation matrix, as well as copies the dataset's
    normalizer.
    
    Args:
        base_path (str): The path where the metadata folder will be created.
        dataset (Dataset): The dataset used for training.
    """
    # create the metadata base dir
    os.makedirs(base_path)

    # save the correlation of the training data
    correlation_path = os.path.join(base_path, "correlation.csv")
    df = dataset.to_dataframe()
    df.corr().to_csv(correlation_path, index=True)

    # copy the normalizer of the training data to the model
    meta = os.path.join(dataset.directory, "metadata", "info.json")
    with open(meta, mode="r", encoding="UTF-8") as meta_file:
        meta_dict = json.load(meta_file)["NORMALIZER"]

    meta = os.path.join(base_path, "normalizer.json")
    with open(meta, mode="w", encoding="UTF-8") as norm_file:
        json.dump(meta_dict, norm_file)


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    # run methods before fit
    cli.before_fit()

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
        os.makedirs(tb_logger.log_dir + "/checkpoints/")
        checkpoint.dirpath = tb_logger.log_dir + "/checkpoints/"
    else:
        logger.warning("ModelCheckpoint callback not defined in yaml-config.")

    # save model summary to log folder
    data = cli.datamodule.dataset
    metafolder = tb_logger.log_dir + "/metadata/"
    save_training_meta(metafolder, data)

    # train
    logger.debug("Starting training")
    model = cli.model
    if model.device == "cuda":
        logger.debug("Compile model for faster training speeds")
        model = torch.compile(cli.model)
    cli.trainer.fit(model, cli.datamodule)

    # run methods after fit
    cli.after_fit()


if __name__ == "__main__":
    cli_main()
