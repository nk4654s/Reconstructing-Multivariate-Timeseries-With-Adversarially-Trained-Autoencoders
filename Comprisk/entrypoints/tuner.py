"""This module creates a model and tunes its hyperparameters."""
from Dataflux.helper import get_logger

from Comprisk.tune import Tuner
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

    tuner = Tuner(cli)
    tuner.config(cli.config.tuner_config)
    tuner.optimize(cli.config.tuner_trials, cli.config.tag)


if __name__ == "__main__":
    cli_main()
