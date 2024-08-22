"""Htis module contains the measurement callback used by the trainer in the train entrypoint."""
import os
from typing import Any
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl
import pandas as pd

from Comprisk import logger


class MeasurementCallback(Callback):
    """This callback measures and stores the loss during training, which is useful
    to know a model's performance afterward. With that knowledge, defining thresholds
    to detect anomalies will get easier.
    """

    def __init__(self) -> None:
        self._data = {
            "epoch": [],
            "mean": [],
            "std": [],
        }
        self._i = 0
        self._steps = 0
        self._ready = False

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        if not self._ready:
            return

        logger.debug("MeasurementCallback: setup %s", stage)

    def on_sanity_check_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._ready = True

    def _collect(self, mean: float, std: float) -> None:
        if not self._ready:
            return

        if len(self._data["mean"]) - 1 < self._i:
            self._data["mean"].append([])
            self._data["std"].append([])

        self._data["mean"][self._i].append(mean)
        self._data["std"][self._i].append(std)
        self._steps += 1

    def _summarize(self) -> None:
        if not self._ready or self._steps == 0:
            return

        self._data["mean"][self._i] = sum(self._data["mean"][self._i]) / self._steps
        self._data["std"][self._i] = sum(self._data["std"][self._i]) / self._steps
        self._data["epoch"].append(self._i)

        self._i += 1
        self._steps = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self._ready:
            return

        loss = outputs["loss"].detach().cpu().numpy()
        self._collect(loss.mean(), loss.std())

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self._ready:
            return

        self._summarize()

        path = os.path.join(trainer.logger.log_dir, "metadata", "loss.csv")
        df = pd.DataFrame(self._data)
        df.to_csv(path)
        logger.debug("Saved MeasurementCallbackresult to %s", path)
