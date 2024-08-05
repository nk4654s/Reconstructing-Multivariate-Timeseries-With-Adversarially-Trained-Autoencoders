"""This module implements an advanced CLI for training or loading
a model. Moreover, the module expands the CLI by **lightning.ai**."""

import importlib
from typing import Any
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
)
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer import Trainer

from Dataflux.helper import get_logger

from Comprisk.data import DataModule

logger = get_logger(__name__, path="./logs/logfile")


class CompriskCLI(LightningCLI):
    """The CompriskCLI adds methods to the LightningCLI for training a model.
    Before training, the user can set options to:
        - find the best `lr_rate` and `batch_size`
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--find-lr", default=False, action="store_true")
        parser.add_argument("--find-batch-size", default=False, action="store_true")
        parser.add_argument("--save-every", default=1)
        parser.add_argument("--tuner-config", nargs="?")
        parser.add_argument("--tuner-trials", type=int, default=25)

    def _class_for_name(self, module_name, class_name):
        # load the module, will raise ImportError if module cannot be loaded
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        c = getattr(m, class_name)
        return c

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        # instantiate the data module
        data_kwargs = self.config.data.as_dict()
        self.datamodule = DataModule(**data_kwargs)
        self.config.pop("data")

        # instantiate the model, combining the config and dataset parameters
        meta = self.datamodule.dataset.get_meta()
        model_kwargs = self.config.model.as_dict()
        class_string = model_kwargs["class_path"].rsplit(".", maxsplit=1)

        model_kwargs = model_kwargs["init_args"]
        model_kwargs["num_samples"] = meta.num_input_sequence
        model_kwargs["num_weather"] = meta.num_features_weather
        model_kwargs["num_sensors"] = meta.num_features_sensors

        # create model class, or load from pickled file
        model_class = self._class_for_name(*class_string)
        self.model = model_class(**model_kwargs)
        self.config.pop("model")

        # instantiate the trainer
        self.trainer = self.instantiate_trainer()

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        """Instantiates the trainer.

        Args:
            kwargs: Any custom trainer arguments.

        """
        # instantiate callbacks
        extra_callbacks = []
        for namespace in self.config.trainer.callbacks:
            class_path = self._class_for_name(*namespace.class_path.rsplit(".", maxsplit=1))
            callback = class_path(**namespace.init_args.as_dict())
            extra_callbacks.append(callback)

        # get the args as dict without callback args
        trainer_config = self.config.trainer.as_dict()
        del trainer_config["callbacks"]

        return Trainer(callbacks=extra_callbacks, ** trainer_config)

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
