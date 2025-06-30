import random

import numpy as np
import torch
from torch.optim import Optimizer

from Comprisk.helper.structure import r2_score, get_loss_by_name, get_optimizer_by_name
from Comprisk.model import BaseModel
from Comprisk.custom import TrainingContext

from Dataflux.pytorch import BATCH_TYPE


class InvalidCertaintyModelLoss(Exception):
    pass


class CertaintyBase(BaseModel):
    def __init__(self, regression_prob: float = 0.8, regression_decay: float = 0.8, regression_loss: str = "L1Loss",
                 warmup_epochs: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

        self._regression_loss = get_loss_by_name(regression_loss)
        self._certainty_loss = torch.nn.GaussianNLLLoss(reduction="sum")

        self._regression_decay = regression_decay
        self._regression_prob = regression_prob
        self._train_step = 0
        self._warmup_epochs = warmup_epochs

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self._warmup_epochs -= 1
        if self._warmup_epochs < 0:
            self._train_step += 1

    def training_step(self, batch: BATCH_TYPE, **kwargs) -> torch.Tensor:
        l1_opt, gauss_op = self.optimizers()

        x, y = batch
        y_hat, y = self.predict_step(x, y, variance=True)
        mean, var = y_hat

        prob = self._regression_decay ** self._train_step * self._regression_prob
        if np.random.rand() < prob:
            # train with L1Loss
            loss = self._regression_loss(mean, y)
            self.log("train/L1Loss", loss)

            l1_opt.zero_grad()
            self.manual_backward(loss)
            l1_opt.step()
        else:
            # train with GaussianNLLLoss
            loss = self._certainty_loss(mean, y, var)
            self.log("train/GaussianNLLLoss", loss)

            gauss_op.zero_grad()
            self.manual_backward(loss)
            gauss_op.step()

        self.log("train/L1AssistProbability", prob)

        # the feature loss is required by the LossMeasurement callback to calculate
        # the model's performance during training.
        feature_names = TrainingContext.get_instance().map_sensors(self._features_out)
        loss_ = self._get_feature_loss(mean, y).tolist()
        for i, name in enumerate(feature_names):
            self.log(f"mean/{name}", loss_[i])
            self.log(f"var/{name}", var[:, 0, i].mean())

        return loss

    def validation_step(self, batch: BATCH_TYPE, **kwargs) -> torch.Tensor:
        x, y = batch
        y_hat, y = self.predict_step(x, y, variance=True)

        mean, var = y_hat
        loss = self._loss(mean, y, var)

        self.log(f"validate/{self._loss_name}", loss)
        self.log(f"validate/{self._regression_loss.__class__.__name__}", self._regression_loss(mean, y))

        batch_size, sequence, features = y.shape
        y = y.reshape(batch_size * sequence, features)
        mean = mean.reshape(batch_size * sequence, features)
        self.log("validate/R2Score", r2_score(mean, y))
        return loss

    def predict_step(self, x: torch.Tensor, y: torch.Tensor, variance: bool = False) -> BATCH_TYPE:
        """
        This method is used to make predictions on a batch of data during inference.

        Args:
            x (torch.Tensor): The model input (will be normalized by the BaseModel).
            y (torch.Tensor): The expected output (will be normalized by the BaseModel).
            variance (bool): Whether to return the variance or not

        Returns:
            torch.Tensor: The predicted output for the given batch.
        """
        x, y = self.normalizer.apply((x, y))
        if variance:
            return self(x), y
        return self(x)[0], y

    def configure_optimizers(self) -> Optimizer:
        if isinstance(self._optim, dict):
            optimizer_name = self._optim.pop("name")
            optim_cls = get_optimizer_by_name(optimizer_name)

            self._optim = [
                optim_cls(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr),
                optim_cls(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
                ]

        return self._optim