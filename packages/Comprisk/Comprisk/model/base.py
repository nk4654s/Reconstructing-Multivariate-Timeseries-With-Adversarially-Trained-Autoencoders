"""This module contains the base model implementing widely-used functions. Layers
of a neural network are not implemented."""

from typing import List, Tuple, Union
import numpy as np
import lightning as L
import torch

from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer
from Dataflux.pytorch import DatasetTypes, dataset_type_from_string
from Comprisk.helper import (
    r2_score,
    get_loss_by_name,
    get_optimizer_by_name,
    get_activation_from_string,
)

from Comprisk import logger


class BaseModel(L.LightningModule):
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
        monte_carlo (int, optional): Number of Monte-Carlo iterations, higher will be
            slower but more stable. Defaults to 0 which is disabled.
    """

    def __init__(
        self,
        loss: str = "MSELoss",
        activation: str = "sigmoid",
        optimizer: str = "Adam",
        lr: float = 1e-3,
        lr_decay: float = 0.99,
        type_: str = DatasetTypes.AUTOENCODER.value,
        output_length: int = 1,
        feature_names: List[str] = []
    ) -> None:
        super().__init__()

        # define training hyperparameters
        self._loss_name = loss
        self._loss = get_loss_by_name(loss)
        self._activation = get_activation_from_string(activation)
        self._optim: Optimizer | str = optimizer
        self._lr_scheduler: ExponentialLR | None = None
        self._lr_decay = lr_decay
        self.lr = lr

        self._behavior = dataset_type_from_string(type_)
        self._output_length = output_length
        
        self._feature_names = feature_names
    
    def set_output_behavior(self, behavior: DatasetTypes | str, length: int) -> None:
        self._behavior = dataset_type_from_string(behavior) if isinstance(behavior, str) else behavior
        self._output_length = length
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        self._feature_names = feature_names

    def _get_feature_loss(self, y_hat, y) -> np.ndarray:
        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        return np.sqrt(np.power(y_hat - y, 2)).mean(1)[0]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """The forward method defines how the input is passed through the
        model's layers. This method needs to be overwritten by a child class,
        as the base model just implements a placeholder.
        """
        raise NotImplementedError

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        assert self._lr_scheduler, "Learning rate scheduler not initialized."
        self.log("epoch/lr", self._lr_scheduler.get_last_lr()[-1])
        logger.debug("Starting epoch %s with lr=%s", self.current_epoch, self._lr_scheduler.get_last_lr()[-1])
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0) -> None:
        logger.debug("Validation finished (loss=%s)", outputs.cpu().numpy().mean())

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], **kwargs) -> torch.Tensor:
        x, y = batch
        y = y if self._behavior == DatasetTypes.PREDICTIVE else x
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log(f"train/{self._loss_name}", loss)

        # the feature loss is required by the LossMeasurement callback to calculate
        # the model's performance during training.
        for name_, loss_ in zip(self._feature_names, self._get_feature_loss(y_hat, y)):
            self.log(f"train/{name_}", loss_)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], **kwargs) -> torch.Tensor:
        x, y = batch
        y = y if self._behavior == DatasetTypes.PREDICTIVE else x
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log(f"test/{self._loss_name}", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], **kwargs) -> torch.Tensor:
        x, y = batch
        y = y if self._behavior == DatasetTypes.PREDICTIVE else x
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log(f"validate/{self._loss_name}", loss)

        batch_size, sequence, features = y.shape
        y = y.reshape(batch_size * sequence, features)
        y_hat = y_hat.reshape(batch_size * sequence, features)
        self.log(f"validate/R2Score", r2_score(y_hat, y))
        return loss

    def configure_optimizers(self) -> Optimizer:
        # init optimizer if it's a string
        if isinstance(self._optim, str):
            logger.debug("Initialize optimizer: %s", self._optim)
            optim_cls = get_optimizer_by_name(self._optim)
            self._optim = optim_cls(self.parameters(), lr=self.lr)
            self._lr_scheduler = ExponentialLR(self._optim, gamma=self._lr_decay)

        return [self._optim], [{"scheduler": self._lr_scheduler, "interval": "epoch"}] # type: ignore

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        This method is used to make predictions on a batch of data during inference.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tensor containing the input data.

        Returns:
            torch.Tensor: The predicted output for the given batch.
        """
        return self(batch[0])

    def predict(self,  x: torch.Tensor, as_array: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """This method only predicts future steps based on the given curve described by
        the datapoints `x`.

        Args:
            x (torch.Tensor): The datapoints.
            as_array (bool, optional): Returns the prediction as np.array. Defaults to False.

        Returns:
            np.ndarray, torch.Tensor: The prediction.
        """
        out = np.array([])
        with torch.no_grad():
            out = self.predict_step((x, x))
            if as_array:
                out = out.cpu().numpy()

        if self._behavior == DatasetTypes.AUTOENCODER:
            out = out[:, out.shape[1] - self._output_length:, :]

        return out


