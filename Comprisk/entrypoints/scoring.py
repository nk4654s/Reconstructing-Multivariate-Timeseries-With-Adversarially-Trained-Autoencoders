"""This module loads a model using the **Comprisk.cli.CompriskCLI** and
applys the loaded model on the configured dataset. The ScoreTransformer
will be learned based on the resulting loss. Only use configure this entry
point with a trained model and the matching validation dataset."""
import os

from Dataflux.pytorch import Dataset
from Dataflux.helper import get_logger

from Comprisk.entrypoints.trainer import train_score_transformer

from Comprisk.helper import get_config_folder
from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskLoadCLI

logger = get_logger(__name__, path="./logs/logfile")


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    logger.debug("Starting CompriskCLI")
    cli = CompriskLoadCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    logger.debug("Logging model: %s", cli.model.__class__.__name__)
    base_path = get_config_folder(cli)

    # load the dataset and configure it to work with the trained model
    dataset = cli.datamodule.valset
    if cli.config.dataset:
        meta = dataset.get_meta()
        dataset = Dataset(cli.config.dataset,
                          meta.num_input_sequence,
                          meta.num_output_sequence,
                          meta.skip,
                          meta.stride)

    # load the model with the helper class, then apply the model on the dataset
    meta_path = os.path.join(base_path, "metadata")
    train_score_transformer(base_path, meta_path, dataset)


if __name__ == "__main__":
    cli_main()
