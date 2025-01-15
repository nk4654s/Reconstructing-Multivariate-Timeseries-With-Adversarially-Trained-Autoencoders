"""This module loads a model using the **Comprisk.cli.CompriskCLI** and
apply the loaded model on the configured dataset. The ScoreTransformer
will be learned based on the resulting loss. Only use configure this entry
point with a trained model and the matching validation dataset."""
import os


from Comprisk.entrypoints.trainer import train_score_transformer

from Comprisk.helper import get_config_folder
from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskLoadCLI

from Comprisk import logger


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    logger.debug("Starting CompriskCLI")
    cli = CompriskLoadCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    base_path = get_config_folder(cli)
    logger.info("Logging model: %s", base_path)

    # load the dataset and configure it to work with the trained model
    dataset = cli.datamodule.valset
    if cli.config.dataset:
        cli.datamodule.set_path(None, cli.config.dataset)
        dataset = cli.datamodule.valset

    if dataset is None:
        logger.error("No dataset available for training the ScoreTransformer. Make sure to define the validation set path.")
        exit(0)

    # load the model with the helper class, then apply the model on the dataset
    meta_path = os.path.join(base_path, "metadata")
    train_score_transformer(base_path, meta_path, dataset, vars(cli.config.score))

    logger.info(
        "Finished ScoreTransformer training for machine %s.",
        dataset.UUID,
    )


if __name__ == "__main__":
    cli_main()
