"""This module contains the base model."""

from typing import List, Dict

import lightning as L
import torch
from torch.optim.lr_scheduler import ExponentialLR

from Dataflux.pytorch import Metadata
from Dataflux.helper import get_logger
from Comprisk.helper import (
    get_loss_by_name,
    get_optimizer_by_name,
    get_activation_from_string,
)

logger = get_logger(__name__)


class BaseModel(L.LightningModule):
    """BaseModel class of any neural network predicting time series.

    Args:
        loss (str, optional): Loss function to use. Defaults to "MSELoss".
        optimizer (str, optional): Optimizer to use. Defaults to "Adam".
        lr (float, optional): Initial learning rate. Defaults to 1e-3.
    """

    def __init__(
        self,
        loss: str = "MSELoss",
        activation: str = "relu",
        optimizer: str = "Adam",
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self._loss_name = loss
        self._loss = get_loss_by_name(loss)
        self._activation = get_activation_from_string(activation)
        self._optim = optimizer
        self._lr_scheduler = ExponentialLR
        self._lr_decay = 0.99
        self.lr = lr

        self._num_features = None
        self._dropout = torch.nn.Dropout()
        self._mc_iteration = 10

    @property
    def target(self) -> str:
        """The target name of inputs to predict."""
        return "targets"

    def init(self) -> None:
        """
        Initializes the model.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def set_meta(self, meta: Metadata) -> None:
        """
        Sets the metadata for this model.

        Args:
            meta (Metadata): The metadata to be used.
        """
        self._meta = meta

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.log("epoch/lr", self._lr_scheduler.get_last_lr()[-1])

    def training_step(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        y_hat = self(batch)
        loss = self._loss(y_hat, batch[self.target])
        self.log(f"train/{self._loss_name}", loss)
        return loss

    def test_step(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        y_hat = self(batch)
        loss = self._loss(y_hat, batch[self.target])
        self.log(f"test/{self._loss_name}", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        y_hat = self(batch)
        loss = self._loss(y_hat, batch[self.target])
        self.log(f"validate/{self._loss_name}", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # init optimizer if it's a string
        if isinstance(self._optim, str):
            logger.debug("Initialize optimizer: %s", self._optim)
            self._optim = get_optimizer_by_name(self._optim)(
                self.parameters(), lr=self.lr
            )
            self._lr_scheduler = self._lr_scheduler(self._optim, gamma=0.99)
        return [self._optim], [{"scheduler": self._lr_scheduler, "interval": "epoch"}]

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        This method is used to make predictions on a batch of data during inference. It
        employs a Monte Carlo Dropout (MCD) approach, which involves randomly disabling
        neurons in the network during prediction. The output of the network is then
        averaged over multiple runs with different to output a more stable prediction.

        Args:
            batch (torch.Tensor): A tensor containing the input data.

        Returns:
            torch.Tensor: The predicted output for the given batch, averaged over multiple MCD runs.
        """
        # enable Monte Carlo Dropout
        self._dropout.train()

        # take average of MC-iterations
        pred = [
            self._dropout(self(batch)).unsqueeze(0) for _ in range(self._mc_iteration)
        ]
        pred = torch.vstack(pred).mean(dim=0)
        return pred

    def predict(self, x: torch.tensor, as_array: bool = False) -> List:
        """This method only predicts future steps based on the given curve described by
        the datapoints `x`. Each step will be predicted using `self.predict_step(x)` which
        employs Monte Carlo Dropout.

        Args:
            x (torch.tensor): The datapoints.
            as_array (bool, optional): Returns the prediction as np.array. Defaults to False.

        Returns:
            List: The prediction.
        """
        with torch.no_grad():
            out = self.predict_step(x)
            if as_array:
                out = out.cpu().numpy()

        return out
