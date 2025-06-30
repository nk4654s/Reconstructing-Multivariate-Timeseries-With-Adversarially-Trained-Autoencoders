"""This module implements an advanced CLI for training or loading
a model. Moreover, the module expands the CLI by **lightning.ai**."""
import datetime
from typing import Any

from Comprisk.custom.components import DataModule, CASTrainer
from Comprisk.custom.callbacks import AdvancedSaveConfigCallback
from Comprisk.helper.structure import deepvars
from Comprisk.model import NormalizeWrapper
from Dataflux import logging_mode
from Dataflux.logger import get_logger
from lightning import Trainer
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
)

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class CompriskCLI(LightningCLI):
    """The CompriskCLI adds methods to the LightningCLI for training a model.
    """
    def __init__(self, **kwargs) -> None:
        kwargs["save_config_callback"] = AdvancedSaveConfigCallback
        super().__init__(
            model_class=NormalizeWrapper,
            datamodule_class=DataModule,
            trainer_class=Trainer,
            **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--tag", type=str, default=None)
        parser.add_argument("--environment", type=str, default="dev")
        parser.add_argument("--normalizer", type=str, default=None)
        parser.add_argument("--prefix", type=str, default=None)
        parser.add_argument("--tuner-config", nargs="?")
        parser.add_argument("--tuner-trials", type=int, default=1)

        # link the model's input/output features with the dataset's configuration
        parser.link_arguments("data.features_in", "model.init_args.features_in", apply_on="parse")
        parser.link_arguments("data.features_out", "model.init_args.features_out", apply_on="parse")
        parser.link_arguments("data.behavior", "model.init_args.behavior", apply_on="parse")
        parser.link_arguments("data.sequence_out", "model.init_args.sequence_out", apply_on="parse")

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        # use the custom trainer class with some modifications which are implicitly given in the config
        config = deepvars(self.config)
        self.trainer_class = CASTrainer
        return super().instantiate_trainer(config = config, **kwargs)

    def after_fit(self):
        """Method executes after fit automatically."""
        logger.debug("Training finished, clean up now.")
        end_str = "-" * 40 + datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "-" * 40
        footer_str = "-" * 100
        logger.debug(end_str)
        logger.debug(footer_str)
