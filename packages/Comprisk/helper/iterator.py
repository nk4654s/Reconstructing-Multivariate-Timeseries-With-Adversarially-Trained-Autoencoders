"""This module contains helper methods related to the models."""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import onnxruntime
import pandas as pd

from Dataflux import logging_mode
from Dataflux.logger import get_logger
from Dataflux.pytorch import Dataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Comprisk.helper import ONNXArgs

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


@dataclass
class IteratorResult:
    """This dataclass holds a model's iteration result including the scores, model inputs, and model outputs. 
    Furthermore, this class is able to convert the result to pandas DataFrames.
    """
    ts: np.ndarray
    scores: np.ndarray
    expected: np.ndarray
    prediction: np.ndarray
    names: List[str]

    def to_dataframe(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """This method returns the iteration result as pandas DataFrames in the following order:

            - scores
            - expected
            - predicted
        
            Returns:
                tuple: The three mentioned DataFrames in a tuple.
        """
        result = []
        for data in [self.scores, self.expected, self.prediction]:
            data_ = {n: data[:, i] for i, n in enumerate(self.names)}
            data_["ts"] = self.ts
            data_ = pd.DataFrame(data_)
            data_ = data_.sort_values(by=["ts"])
            data_ = data_.set_index(keys=["ts"])
            result.append(data_)

        return tuple(result)


class ModelIterator:
    """This class handles iterating over a dataset while applying a model.

    Args:
        onnx_path (str): The model to load (ONNX format).
        data (Dataset | str): The dataset to use (if given aas path, the dataset will be configured to
            match the provided ONNX model).
    """

    def __init__(self, onnx_path: str, data: Dataset | str) -> None:
        if isinstance(data, str):
            args = ONNXArgs.load_from_model(onnx_path)
            data = args.build_dataset(data)

        self._data = data
        self._args = data.get_args()
        self._len = len(data)

        x_shape, y_shape = self._data[0]
        self.x_shape, self.y_shape = x_shape.shape, y_shape.shape
        logger.info("ModelIterator input shape %s, output shape %s", self.x_shape, self.y_shape)

        # create the ONNX session and find the input/output names
        self._session = onnxruntime.InferenceSession(onnx_path)
        self._i_onnx = [i.name for i in self._session.get_inputs()]
        self._o_onnx = [i.name for i in self._session.get_outputs()]
        self._s_idx = self._o_onnx.index("score")
        self._y_idx = self._o_onnx.index("y_hat")
        logger.info(f"ModelIterator initialized. {self._session._model_meta.description}")

    def _init_container(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        f_out = self._args.features_out
        ts = np.zeros(shape=(self._len,), dtype=np.int64)
        scores = np.zeros(shape=(self._len, f_out), dtype=np.float32)
        sensor_input = np.zeros(shape=(self._len, f_out), dtype=np.float32)
        sensor_output = np.zeros(shape=(self._len, f_out), dtype=np.float32)

        return ts, scores, sensor_input, sensor_output

    def run(self, batch_size: int = 64, num_workers: int = 4) -> IteratorResult:
        """This function executes a ModelCheckpoint on a given dataset. Then the model's output and input will
        be returned.

        Args:
            batch_size (int): The number of samples in a single pass. Defaults to 64.
            num_workers (int): Number of threads iterating the dataset. Defaults to 4.

        Returns:
            IteratorResult: The model's result on the given dataset.

        """
        ts, scores, sensor_input, sensor_output = self._init_container()

        limit = 0
        dataloader = DataLoader(self._data, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers)
        for i, (x, y) in enumerate(tqdm(dataloader)):
            # run onnx inference
            inputs = [x.numpy().astype(np.float32), y.numpy().astype(np.float32)]
            inputs = {n: inputs[i] for i, n in enumerate(self._i_onnx)}
            onnx_output = list(self._session.run(None, inputs))

            # extract the model output/input into the containers
            batch_size = onnx_output[self._s_idx].shape[0]
            batch_size = len(scores[limit:limit + batch_size])
            scores[limit:limit + batch_size] = onnx_output[self._s_idx][:batch_size, -1, :]
            sensor_input[limit:limit + batch_size] = y[:batch_size, -1, :].numpy()
            sensor_output[limit:limit + batch_size] = onnx_output[self._y_idx][:batch_size, -1, :]

            ts_sub = np.array([self._data.get_timestamp(limit) + t for t in range(batch_size)], dtype=np.int64)
            ts[limit:limit + batch_size] = ts_sub
            limit += batch_size

        # ensure correct length
        ts = ts[:limit]
        scores = scores[:limit]
        sensor_input = sensor_input[:limit]
        sensor_output = sensor_output[:limit]

        result = IteratorResult(
            ts=ts,
            scores=scores,
            expected=sensor_input,
            prediction=sensor_output,
            names=self._args.names_out
        )

        return result
