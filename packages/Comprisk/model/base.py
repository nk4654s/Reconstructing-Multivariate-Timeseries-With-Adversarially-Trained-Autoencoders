"""This module contains the base model implementing widely-used functions. Layers
of a neural network are not implemented."""
from typing import List

import numpy as np
import torch
from Dataflux import logging_mode
from Dataflux.logger import get_logger
from Dataflux.pytorch import BATCH_TYPE, DatasetTypes, dataset_type_from_string
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer

from Comprisk.helper.structure import (
    get_activation_from_string,
    get_loss_by_name,
    get_optimizer_by_name,
    r2_score,
)
from Comprisk.model import NormalizeWrapper

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class BaseModel(NormalizeWrapper):
    """BaseModel class of any neural network. The base model implements a learning
    rate decay using an exponential learning rate scheduler. Also, when predicting
    using a trained model, the base class uses Monte-Carlo sampling to obtain a
    more stable prediction.

    Args:
        loss (str, optional): Loss function to use. Defaults to "MSELoss".
        activation (str, optional): Activation function to use. Defaults to "sigmoid"
        optimizer (str, optional): Optimizer to use. Defaults to "Adam".
        lr (float, optional): Initial learning rate. Defaults to 1e-3.
        lr_decay (float, optional): Learning rate decaying factor. Defaults to 0.99.
    """

    def __init__(
        self,
        loss: str = "MSELoss",
        activation: str = "sigmoid",
        optimizer: str = "Adam",
        lr: float = 1e-3,
        lr_decay: float = 0.99,
        sequence_out: int = 0,
        features_in: List[str] = None,
        features_out: List[str] = None,
        behavior: str = DatasetTypes.AUTOENCODER.value,
        output_length: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(features_in, features_out, *args, **kwargs)

        # define training hyperparameters
        self._loss_name = loss
        self._loss = get_loss_by_name(loss)
        self._activation = get_activation_from_string(activation)
        self._optim: Optimizer | str = optimizer
        self._lr_scheduler: ExponentialLR | None = None
        self._lr_decay = lr_decay
        self.lr = lr

        self._sequence_out = sequence_out
        self._features_in = sorted(features_in)
        self._features_out = sorted(features_out)

        self._input_dim = len(features_in)
        self._output_dim = len(features_out)
        logger.info("The model gets %s features and outputs %s values", self._input_dim, self._output_dim)

        self._behavior = dataset_type_from_string(behavior)
        self._output_length = output_length

    def set_output_behavior(self, behavior: DatasetTypes | str, length: int) -> None:
        self._behavior = dataset_type_from_string(behavior) if isinstance(behavior, str) else behavior
        self._output_length = length

    @staticmethod
    def _get_feature_loss(y_hat, y) -> np.ndarray:
        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        # mean over batch and sequence, but keep sensors
        return np.sqrt(np.power(y_hat - y, 2)).mean(axis=(0, 1))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """The forward method defines how the input is passed through the
        model's layers. This method needs to be overwritten by a child class,
        as the base model just implements a placeholder.
        """
        raise NotImplementedError

    def on_fit_start(self) -> None:
        self.normalizer.to(self.device)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        assert self._lr_scheduler, "Learning rate scheduler not initialized."
        self.log("epoch/lr", self._lr_scheduler.get_last_lr()[-1])
        logger.debug("Starting epoch %s with lr=%s", self.current_epoch, self._lr_scheduler.get_last_lr()[-1])

    def training_step(self, batch: BATCH_TYPE, **kwargs) -> torch.Tensor:
        x, y = batch
        y_hat, y = self.predict_step(x, y)
        loss = self._loss(y_hat, y)

        self.log(f"train/{self._loss_name}", loss)

        # the feature loss is required by the LossMeasurement callback to calculate
        # the model's performance during training.
        loss_ = self._get_feature_loss(y_hat, y).tolist()
        for i, name in enumerate(self._features_out):
            self.log(f"train/{name}", loss_[i])

        return loss

    def test_step(self, batch: BATCH_TYPE, **kwargs) -> torch.Tensor:
        x, y = batch
        y_hat, y = self.predict_step(x, y)
        loss = self._loss(y_hat, y)

        self.log(f"test/{self._loss_name}", loss)
        return loss

    def validation_step(self, batch: BATCH_TYPE, **kwargs) -> torch.Tensor:
        x, y = batch
        y_hat, y = self.predict_step(x, y)
        loss = self._loss(y_hat, y)

        self.log(f"validate/{self._loss_name}", loss)

        batch_size, sequence, features = y.shape
        y = y.reshape(batch_size * sequence, features)
        y_hat = y_hat.reshape(batch_size * sequence, features)
        self.log("validate/R2Score", r2_score(y_hat, y))
        return loss

    def configure_optimizers(self) -> Optimizer:
        # init optimizer if it's a string
        if isinstance(self._optim, str):
            logger.debug("Initialize optimizer: %s", self._optim)
            optim_cls = get_optimizer_by_name(self._optim)
            self._optim = optim_cls(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
            self._lr_scheduler = ExponentialLR(self._optim, gamma=self._lr_decay)

        return [self._optim], [{"scheduler": self._lr_scheduler, "interval": "epoch"}] # type: ignore

    def predict_step(self, x: torch.Tensor, y: torch.Tensor) -> BATCH_TYPE:
        """
        This method is used to make predictions on a batch of data during inference.

        Args:
            x (torch.Tensor): The model input (will be normalized by the BaseModel).
            y (torch.Tensor): The expected output (will be normalized by the BaseModel).

        Returns:
            torch.Tensor: The predicted output for the given batch.
        """
        x, y = self.normalizer.apply((x, y))
        return self(x), y
