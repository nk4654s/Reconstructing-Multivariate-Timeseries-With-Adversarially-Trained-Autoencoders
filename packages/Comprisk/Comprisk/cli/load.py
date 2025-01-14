"""This module implements an advanced CLI for training or loading
a model. Moreover, the module expands the CLI by **lightning.ai**."""
import datetime
import os

from lightning.pytorch.cli import (
    LightningArgumentParser,
)

from .cli import CompriskCLI
from Comprisk import logger


class CompriskLoadCLI(CompriskCLI):
    """The CompriskCLI adds methods to the LightningCLI for loading a model."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument("--prefix", type=str, default=None)
        parser.add_argument("--dataset", type=str, default=None)
        parser.add_argument("--anomaly.name", type=str, default=None)
        parser.add_argument("--anomaly.factor", type=float, default=None)

        start_str = "-" * 40 + datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "-" * 40
        header_str = "-" * 100
        logger.debug(header_str)
        logger.debug(start_str)
        logger.debug("Initializing CompriskCLI for inference")

        debug = os.environ.get("CAS_DEBUG_MODE", "0") == "1"
        logger.info("Publishing scores to test device? %s", "yes" if debug else "no")

    def before_fit(self):
        pass

    def after_fit(self):
        pass
