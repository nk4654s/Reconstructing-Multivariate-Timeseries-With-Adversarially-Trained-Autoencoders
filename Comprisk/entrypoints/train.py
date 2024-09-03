"""This module creates a model summary and plots it."""

import os
import pickle
import shutil
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from Dataflux.pytorch import Dataset

from Comprisk.anomaly import ScoreTransformer
from Comprisk.callbacks import MeasurementCallback
from Comprisk.helper import ModelLoader
from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskCLI

from Comprisk import logger

def get_trainer_callback(trainer, callback_cls: Callback) -> Callback:
    """
    Finds the fiven callback class in the trainer callbacks list. If the
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
    """
    This function creates the model's metadata folder and generates the
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
    normfile = os.path.join(dataset.directory, "metadata", "normalizer.pkl")
    shutil.copyfile(normfile, os.path.join(base_path, "normalizer.pkl"))

    # copy the dataset config of the training data to the model
    config_file = os.path.join(dataset.directory, "metadata", "config.yaml")
    shutil.copyfile(config_file, os.path.join(base_path, "dataset_config.yaml"))


def run_trainer(cli: CompriskCLI) -> str:
    """
    This function trains a model given the cli which contains all parameters, e.g.
    for the model itself or the dataset.

    Args:
        cli (CompriskCLI): The CLI containing all parameters for training.

    Returns:
        str: The path where the model was saved.
    """
    # set up the logger
    sub_path = cli.model.__class__.__name__
    if cli.config.tag:
        sub_path = f"{cli.model.__class__.__name__}/{cli.config.tag}"

    logger.debug("Defining TensorBoard logger for %s", sub_path)
    tb_logger = TensorBoardLogger("models", name=sub_path)
    cli.trainer.logger = tb_logger
    os.makedirs(tb_logger.log_dir)
    logger.debug("Using log folder %s", tb_logger.log_dir)

    # set up the model saving as configuring the checkpoint path in the yaml config does not
    # work. The config file does not know the model's output folder name with the increasing
    # suffix number
    checkpoint = get_trainer_callback(cli.trainer, ModelCheckpoint)
    if checkpoint:
        os.makedirs(tb_logger.log_dir + "/checkpoints/")
        checkpoint.dirpath = tb_logger.log_dir + "/checkpoints/"

    else:
        logger.warning("ModelCheckpoint callback not defined in yaml-config.")

    # save model metadata to log folder
    data = cli.datamodule.dataset
    metafolder = os.path.join(tb_logger.log_dir, "metadata")
    cli.trainer.callbacks.append(MeasurementCallback())
    save_training_meta(metafolder, data)
    
    # check iif a model should be finetuned
    if cli.config.finetune:
        config_path = os.path.dirname(cli.config.finetune)
        loader = ModelLoader(config_path)
        loaded_model = loader.get_model()
        model = loaded_model
        model.train()
    else:
        model = cli.model
        if model.device == "cuda":
            logger.debug("Compile model for faster training speeds")
            model = torch.compile(cli.model)

    # train
    logger.debug("Starting training")
    cli.trainer.fit(model, cli.datamodule)

    # run methods after fit
    cli.after_fit()

    # if a model was finetuned, make sure to have the same hparams.yaml in the
    # output folder.
    # FIXME: this should not be a problem, this is a workaround
    if cli.config.finetune:
        if os.path.exists(os.path.join(tb_logger.log_dir, "hparams.yaml")):
            os.remove(os.path.join(tb_logger.log_dir, "hparams.yaml"))
        shutil.copyfile(os.path.join(config_path, "hparams.yaml"),
                        os.path.join(tb_logger.log_dir, "hparams.yaml"))

    # load the (best) trained model back
    loader = ModelLoader(tb_logger.log_dir)
    predicted, target = loader.predict_loop(data)

    # fit and save the score transformer
    score = ScoreTransformer()
    score.fit(predicted, target)

    score_path = os.path.join(metafolder, "score_transformer.pkl")
    with open(score_path, "wb") as f:
        pickle.dump(score, f)
    
    return tb_logger.log_dir


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    # run methods before fit, then train
    cli.before_fit()
    run_trainer(cli)


if __name__ == "__main__":
    cli_main()
