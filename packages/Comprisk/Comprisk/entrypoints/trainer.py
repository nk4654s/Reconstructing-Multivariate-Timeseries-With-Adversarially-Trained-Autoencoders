"""This module creates a model summary and plots it."""

import os
import pickle
import shutil
from typing import Dict

import yaml
from lightning.pytorch.callbacks.callback import Callback
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from Dataflux.pytorch import Dataset

from Comprisk.anomaly import ScoreTransformer
from Comprisk.helper import ModelLoader, set_model_features
from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskCLI

from Comprisk import logger

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
    tb_logger = TensorBoardLogger(f"environment/{cli.config.environment}/models", name=sub_path)
    cli.trainer.logger = tb_logger
    os.makedirs(tb_logger.log_dir)
    logger.debug("Using log folder %s", tb_logger.log_dir)

    # set up the model saving as configuring the checkpoint path in the yaml config does not
    # work. The config file does not know the model's output folder name with the increasing
    # suffix number
    checkpoint = get_trainer_callback(cli.trainer, ModelCheckpoint.__name__)
    if checkpoint:
        os.makedirs(tb_logger.log_dir + "/checkpoints/")
        checkpoint.dirpath = tb_logger.log_dir + "/checkpoints/"

    else:
        logger.warning("ModelCheckpoint callback not defined in yaml-config.")

    # save model metadata to log folder
    data = cli.datamodule.trainset
    metafolder = os.path.join(tb_logger.log_dir, "metadata")
    save_training_meta(metafolder, data)
    
    # check if a model should be finetuned
    if cli.config.finetune:
        config_path = os.path.dirname(cli.config.finetune)
        loader = ModelLoader(config_path)
        model = loader.get_model()
        model.train()
        set_model_features(model, data)
    else:
        model: LightningModule = cli.model

    # train
    logger.debug("Starting training")
    cli.trainer.fit(model, cli.datamodule)

    # run methods after fit
    cli.after_fit()

    # if a model was finetuned, make sure to have the same hparams.yaml in the
    # output folder.
    # FIXME: this should not be a problem, this is a workaround
    if cli.config.finetune:
        config_path = os.path.dirname(cli.config.finetune)
        if os.path.exists(os.path.join(tb_logger.log_dir, "hparams.yaml")):
            os.remove(os.path.join(tb_logger.log_dir, "hparams.yaml"))
        shutil.copyfile(os.path.join(config_path, "hparams.yaml"),
                        os.path.join(tb_logger.log_dir, "hparams.yaml"))
    
    valset: Dataset = cli.datamodule.valset
    train_score_transformer(tb_logger.log_dir, metafolder, valset)
    return tb_logger.log_dir


def train_score_transformer(model_dir: str, meta_dir: str, dataset: Dataset,
                            transformer_args: Dict = {}) -> None:
    """This method trains the ScoreTransformer given the model and dataset. Usually, this is immediately done on the
    validation set after a training.

    Args:
        model_dir (str): The model's directory.
        meta_dir (str): The model's metadata directory.
        dataset (Dataset): The dataset to train the ScoreTransformer on.
    """
    dataset_args = dataset.get_args()
    dataset.set_stride(dataset_args.sequence_out)

    # load the (best) trained model back
    loader = ModelLoader(model_dir)
    loader.model.set_output_behavior(dataset_args.behavior, dataset_args.sequence_out)
    predicted, target = loader.predict_loop(dataset)

    # fit and save the score transformer
    score = ScoreTransformer(**transformer_args)
    score.fit(predicted, target)

    score_path = os.path.join(meta_dir, "score_transformer.pkl")
    with open(score_path, mode="wb") as f:
        pickle.dump(score, f)


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    cli = CompriskCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    # run methods before fit, then train
    cli.before_fit()
    model_folder = run_trainer(cli)

    # delete dataset paths from the saved config file
    with open(os.path.join(model_folder, "config.yaml"), mode="r+", encoding="utf-8") as f:
        model_args = yaml.safe_load(f)
        model_args["data"]["train"] = None
        model_args["data"]["validate"] = None
        yaml.safe_dump(model_args, f)


if __name__ == "__main__":
    cli_main()
