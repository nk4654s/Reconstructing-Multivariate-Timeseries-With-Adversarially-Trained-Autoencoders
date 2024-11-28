"""This module contains the base model implementing widely-used functions. Layers
of a nerual network are not implemented."""

from typing import Dict, List, Union
import numpy as np
import lightning as L
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer
from Dataflux.pytorch.metadata import Metadata
from Comprisk.helper import (
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
        lr_decay (float, optional): Learning rate dacying factor. Defaults to 0.99.
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
        monte_carlo: int = 0,
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

        # define stable predict-step parameters
        self._dropout = torch.nn.Dropout()
        self._mc_iteration = monte_carlo
        if self._mc_iteration > 0:
            logger.debug("Predict step uses Monte-Carlo (num samples: %s)", self._mc_iteration)
        
        self._feature_names = feature_names
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        self._feature_names = feature_names

    def set_monte_carlo(self, num_iterations: int) -> None:
        """Overwrites the constructor monte-carlo iteration number. Setting this value to 0
        disables monte carlo sampling.

        Args:
            num_iterations (int): The amount of iterations to sample from the model. The mean
                of the sampled outputs defines the model's output given an input.
        """
        logger.debug("Overwriting MC iteration in predict from %s to %s", self._mc_iteration, num_iterations)
        self._mc_iteration = num_iterations

    def summary(self, meta: Metadata) -> str:
        """Returns a summary of the model as string. Requires to be
        implmenented by a child class to work. This function is just a
        placeholder, as the base model does not initialize any layers.

        Args:
            meta (Metadata): Metadata object containing information about the
                dataset. May be required to build a dummy input for retrieving
                the model's summary.
        
        Returns:
            str: The summary of the model as string.
        """
        raise NotImplementedError

    @property
    def target(self) -> List[str]:
        """The target name of inputs to predict."""
        return ["targets"]

    def _get_target_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns the target batch from the given batch."""
        if len(self.target) == 1:
            return batch[self.target[0]]

        x = batch[self.target[0]]
        for t in self.target[1:]:
            x = torch.cat((x, batch[t]), dim=-1)
        return x

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

    def training_step(self, batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        y_hat = self(batch)
        y = self._get_target_batch(batch)
        loss = self._loss(y_hat, y)
        self.log(f"train/{self._loss_name}", loss)

        # the feature loss is required by the LossMeasurement callback to calculate
        # the model's performance during training.
        for name_, loss_ in zip(self._feature_names, self._get_feature_loss(y_hat, y)):
            self.log(f"train/{name_}", loss_)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        y_hat = self(batch)
        loss = self._loss(y_hat, self._get_target_batch(batch))
        self.log(f"test/{self._loss_name}", loss)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        y_hat = self(batch)
        loss = self._loss(y_hat, self._get_target_batch(batch))
        self.log(f"validate/{self._loss_name}", loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        # init optimizer if it's a string
        if isinstance(self._optim, str):
            logger.debug("Initialize optimizer: %s", self._optim)
            optim_cls = get_optimizer_by_name(self._optim)
            self._optim = optim_cls(self.parameters(), lr=self.lr)
            self._lr_scheduler = ExponentialLR(self._optim, gamma=self._lr_decay)

        return [self._optim], [{"scheduler": self._lr_scheduler, "interval": "epoch"}] # type: ignore

    def predict_step(self, batch:  Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This method is used to make predictions on a batch of data during inference. It
        employs a Monte Carlo Dropout (MCD) approach, which involves randomly disabling
        neurons in the network during prediction. The output of the network is then
        averaged over multiple runs with different to output a more stable prediction.

        Args:
            batch ( Dict[str, torch.Tensor]): A tensor containing the input data.

        Returns:
            torch.Tensor: The predicted output for the given batch, averaged over multiple MCD runs.
        """
        if self._mc_iteration > 0:
            # enable Monte Carlo Dropout
            self._dropout.train()

            # take average of MC-iterations
            pred = [
                self._dropout(self(batch)).unsqueeze(0) for _ in range(self._mc_iteration)
            ]
            pred = torch.vstack(pred).mean(dim=0)
            return pred
        return self(batch)

    def predict(self,  x: Dict[str, torch.Tensor], as_array: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """This method only predicts future steps based on the given curve described by
        the datapoints `x`. Each step will be predicted using `self.predict_step(x)` which
        employs Monte Carlo Dropout.

        Args:
            x (Dict[str, torch.Tensor]): The datapoints given as dict of tensors with the keys "sensors" and "weather".
            as_array (bool, optional): Returns the prediction as np.array. Defaults to False.

        Returns:
            np.ndarray, torch.Tensor: The prediction.
        """
        out = np.array([])
        with torch.no_grad():
            out = self.predict_step(x)
            if as_array:
                out = out.cpu().numpy()

        return out
