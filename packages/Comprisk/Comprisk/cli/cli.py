"""This module implements an advanced CLI for training or loading
a model. Moreover, the module expands the CLI by **lightning.ai**."""
import inspect
import datetime
from lightning import Trainer
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
)

from Dataflux import logging_mode
from Dataflux.logger import get_logger

from Comprisk.anomaly import ScoreTransformer

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


def generate_class_arguments(cls, prefix: str, parser: LightningArgumentParser) -> None:
    """This method adds a classes constructor arguments to the CLI. For a proper presentation, make sure that
    each argument is well documented.

    Args:
        cls: The class to inspect.
        prefix (str): The argument prefix which is followed by ".arg_name".
        parser (LightningArgumentParser): The CLI parser.
    """
    args = inspect.signature(cls.__init__).parameters
    docs = cls.__doc__.split("Args:")[1].strip().split("\n")
    exclude = ["self", "args", "kwargs"]
    index = 0
    for key, value in args.items():
        if key in exclude:
            continue

        arg_name = f"--{prefix}.{key}"
        desc = docs[index].split(":", maxsplit=1)[1].strip() if docs[index].startswith(key) else ""
        parser.add_argument(arg_name, type=value.annotation, default=value.default, help=desc)
        index += 1


class CompriskCLI(LightningCLI):
    """The CompriskCLI adds methods to the LightningCLI for training a model.
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--tag", type=str, default=None)
        parser.add_argument("--environment", type=str, default="dev")
        parser.add_argument("--prefix", type=str, default="analysis")
        parser.add_argument("--tuner-config", nargs="?")
        parser.add_argument("--tuner-trials", type=int, default=1)

        # link the model's input/output features with the dataset's configuration
        parser.link_arguments("data.features_in", "model.init_args.features_in", apply_on="instantiate")
        parser.link_arguments("data.features_out", "model.init_args.features_out", apply_on="instantiate")
        parser.link_arguments("data.behavior", "model.init_args.behavior", apply_on="instantiate")

        # generate arguments for the custom classes
        generate_class_arguments(ScoreTransformer, "score_transformer", parser)

    def before_fit(self):
        """Method executes before fit automatically."""
        # if the package should tune the model's hyperparameters, then reset the number
        # of training iterations
        if self.config.tuner_config:
            trainer_args = vars(self.config.trainer)
            trainer_args["callbacks"] = self.trainer.callbacks # type: ignore
            trainer_args["max_epochs"] = 1
            self.trainer = Trainer(**trainer_args)

    def after_fit(self):
        """Method executes after fit automatically."""
        logger.debug("Training finished, clean up now.")
        end_str = "-" * 40 + datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "-" * 40
        footer_str = "-" * 100
        logger.debug(end_str)
        logger.debug(footer_str)
