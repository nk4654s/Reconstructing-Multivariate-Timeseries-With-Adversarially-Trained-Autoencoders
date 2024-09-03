"""This module implements an advanced CLI for training or loading
a model. Moreover, the module expands the CLI by **lightning.ai**."""

import datetime
from lightning import Trainer
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
)
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from Comprisk import logger


class CompriskCLI(LightningCLI):
    """The CompriskCLI adds methods to the LightningCLI for training a model.
    Before training, the user can set options to:
        - find the best `lr_rate` and `batch_size`
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--tag", type=str, default=None)
        parser.add_argument("--finetune", "-f", type=str, default=None)
        parser.add_argument("--find-lr", default=False, action="store_true")
        parser.add_argument("--find-batch-size", default=False, action="store_true")
        parser.add_argument("--save-every", default=1)
        parser.add_argument("--tuner-config", nargs="?")
        parser.add_argument("--tuner-trials", type=int, default=1)

    def before_fit(self):
        """Method executes before fit automatically."""
        start_str = "-" * 40 + datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "-" * 40
        header_str = "-" * 100
        logger.debug(header_str)
        logger.debug(start_str)
        logger.debug("Starting CompriskCLI for training")
        
        # if the package should tune the model's hyperparameters, then reset the number
        # of training iterations
        if self.config.tuner_config:
            trainer_args = vars(self.config.trainer)
            trainer_args["callbacks"] = self.trainer.callbacks
            trainer_args["max_epochs"] = 1
            self.trainer = Trainer(**trainer_args)

        # tune the learning rate and batcg size if required
        # this operation may take a while before the training can start
        tuner = Tuner(self.trainer)
        # find best batch size
        if self.config.find_batch_size:
            logger.debug("Tuning batch size")
            tuner.scale_batch_size(self.model, mode="power", init_val=128)

        # find the best lr rate
        if self.config.find_lr:
            lr_finder = tuner.lr_find(
                self.model,
                self.datamodule,
                num_training=100,
                mode="linear",
                min_lr=1e-8,
                max_lr=0.3,
            )
            lr = lr_finder.suggestion()
            logger.debug("Tuned learning rate: lr=%s", lr)
            self.model.lr = lr


    def after_fit(self):
        """Method executes after fit automatically."""
        logger.debug("Training finished, clean up now.")
        end_str = "-" * 40 + datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "-" * 40
        footer_str = "-" * 100
        logger.debug(end_str)
        logger.debug(footer_str)
