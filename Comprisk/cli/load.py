"""This module implements an advanced CLI for training or loading
a model. Moreover, the module expands the CLI by **lightning.ai**."""

from lightning.pytorch.cli import (
    LightningArgumentParser,
)

from .cli import CompriskCLI


class CompriskLoadCLI(CompriskCLI):
    """The CompriskCLI adds methods to the LightningCLI for loading a model."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument("--prefix", type=str, default=None)
        parser.add_argument("--dataset", type=str, default=None)
        parser.add_argument("--anomaly.name", type=str, default=None)
        parser.add_argument("--anomaly.factor", type=float, default=None)
        parser.add_argument("--step", "-s", type = int, default = None,
                            help = "The step size to slide the model's input/output window over the dataset.")

    def before_fit(self):
        pass

    def after_fit(self):
        pass
