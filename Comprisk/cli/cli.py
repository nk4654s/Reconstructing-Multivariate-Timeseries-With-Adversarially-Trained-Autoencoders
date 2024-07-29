"""This module contains an advanced CLI for training a model."""

from argparse import Namespace
from typing import Any, Callable, Dict, List
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint

from Dataflux.helper import get_logger

logger = get_logger(__name__)


class CompriskCLI(LightningCLI):
    """The CompriskCLI adds methods to the LightningCLI for training a model.

        Before training, the user can set options to:
        - find the best `lr_rate` and `batch_size`

    Args:
        LightningCLI (LightningCLI): The base class.
    """
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--find-lr", default=False, action="store_true")
        parser.add_argument("--find-batch-size", default=False, action="store_true")
        parser.add_argument("--save-every", default=1)

    def prepare(self) -> None:
        """Prepares the model by passing metadata from the dataset to the model using
        `model.set_meta()`. Then, the `model.init()` is called to finalize the model
        initialization.
        """
        self.model.set_meta(self.datamodule.dataset.get_meta())
        self.model.init()

    def before_fit(self):
        """Method executes before fit automatically."""
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

        # add model checkpoint callback
        model_check = ModelCheckpoint(
            self.trainer.logger.log_dir, every_n_epochs=self.config.save_every
        )
        for i, c in enumerate(self.trainer.callbacks):
            if isinstance(c, ModelCheckpoint):
                self.trainer.callbacks[i] = model_check

    def after_fit(self):
        """Method executes after fit automatically."""
        pass
