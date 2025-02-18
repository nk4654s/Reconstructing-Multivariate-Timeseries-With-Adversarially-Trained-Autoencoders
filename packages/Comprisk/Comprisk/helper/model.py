"""This module contains helper methods related to the models."""

from typing import  Tuple
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from Comprisk.anomaly import ScoreWrapper
from Dataflux import DEBUG_MODE
from Dataflux.pytorch import collate_fn, DatasetTypes
from Dataflux.pytorch import Dataset

from Dataflux import logging_mode
from Dataflux.logger import get_logger


# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class ModelIterator:
    """This class handles iterating over a dataset while applying a model.

    Args:
        model (ScoreWrapper): The model to use.
        data (Dataset): The dataset to use
    """

    def __init__(self, model: ScoreWrapper, data: Dataset) -> None:
        self._model = model
        self._device = model.device

        self._data = data
        self._args = data.get_args()
        self._stride = self._args.stride
        self._skip = self._args.skip

        self._ts_offset = 0
        self._window = self._args.sequence_in
        if self._args.behavior == DatasetTypes.PREDICTIVE:
            self._window = self._args.sequence_out
            self._ts_offset = self._args.sequence_in * self._skip

        self._total_length = len(data) * self._stride
        self._features = self._args.features_in
        self._targets = self._args.features_out

    def _init_container(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sensor_shape = (self._total_length, self._features)
        target_shape = (self._total_length, self._targets)
        value_shape = (self._total_length,)

        inputs = np.zeros(shape=sensor_shape, dtype=np.float32)
        targets = np.zeros(shape=target_shape, dtype=np.float32)
        counter = np.zeros(shape=value_shape, dtype=np.float32)
        ts_index = np.zeros(shape=value_shape, dtype=np.float64)

        return inputs, targets, counter, ts_index

    def run(self, batch_size: int = 64, num_workers: int =4):
        inputs, targets, counter, ts_index = self._init_container()

        loader = DataLoader(
            self._data,
            batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=False,
        )

        p_bar = tqdm(loader, desc="Prediction with Model", disable=not DEBUG_MODE)
        for i, (x, y) in enumerate(p_bar):
            x: torch.Tensor = x.to(self._device)

            # run the prediction, the input equals the target for auto encoders
            outputs = self._model.predict_step(x, y).cpu().detach().numpy()

            batch_size_, _, _ = outputs.shape
            for j in range(0, batch_size_):
                idx = 0 if i == 0 else i * batch_size * self._stride
                idx = idx + j * self._stride

                window_, _ = outputs[j].shape
                window_ = min(window_, len(targets[idx : idx + window_]))
                inputs[idx : idx + window_] += x[j][window_ - self._window:].cpu().detach().numpy()
                targets[idx : idx + window_] += outputs[j][window_ - self._window:]
                counter[idx : idx + window_] += np.ones(shape=(window_,))

                ts = self._data.get_timestamp(i * batch_size + j) + self._ts_offset
                ts_index[idx : idx + window_] = [ts + k * self._skip for k in range(window_)]

        limit = counter.argmin()
        limit = limit if counter[limit] == 0 else -1
        ts_index = pd.Index(ts_index[:limit], name="ts")

        counter = counter[:limit]
        targets = targets[:limit]
        inputs = inputs[:limit]

        # create the output DataFrames
        names = self._args.names_in
        inputs = pd.DataFrame(
            {n: inputs[:, i] / counter for i, n in enumerate(names)}, index=ts_index
        )
        names = self._args.names_out
        targets = pd.DataFrame(
            {n: targets[:, i] / counter for i, n in enumerate(names)}, index=ts_index
        )

        return targets, inputs
