"""This module creates a model and tunes its hyperparameters."""
import os
import yaml
from Dataflux.helper import get_logger

from Comprisk.tune import Tuner, ConfigManager
from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskCLI

logger = get_logger(__name__, path="./logs/logfile")


def run_tuner(cli: CompriskCLI) -> str:
    """
    This function runs the hyperparameter tuning using the Tuner class. It takes a
    CompriskCLI object as input and returns the best model configuration.

    Args:
        cli (CompriskCLI): A CompriskCLI object that contains the configuration
            and data modules for hyperparameter tuning.
    
    Returns:
        str: The best model configuration as a string in YAML.
    """
    # load the parameters to tune in a ConfigManager
    config = ConfigManager(cli)
    config.load(cli.config.tuner_config)

    # run the hyperparameter optimization with the ConfigManager
    tuner = Tuner(config)
    params = tuner.optimize(cli.config.tuner_trials, cli.config.tag)
    logger.info("Best parameters: %s", params)

    # get the base config for training and modify the parameters with the best ones
    modified = config.modify_config(params)

    # save the modified config to a new file with increasing suffix
    modified_path = config.train_path.replace(".yml", "_tuned_0.yml")
    i = 0
    while os.path.exists(modified_path):
        modified_path = config.train_path.replace(".yml", f"_tuned_{i}.yml")
        i += 1

    with open(modified_path, "w", encoding="utf-8") as f:
        yaml.dump(modified, f)
    
    return modified_path


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    logger.debug("Starting CompriskCLI")
    cli = CompriskCLI(BaseModel, DataModule, subclass_mode_model=True, run=False,
                      save_config_callback=None)
    cli.before_fit()
    run_tuner(cli)


if __name__ == "__main__":
    cli_main()
