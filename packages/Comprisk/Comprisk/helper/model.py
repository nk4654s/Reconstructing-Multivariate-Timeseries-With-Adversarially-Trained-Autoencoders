"""This module contains helper methods related to the models."""

import os
import pickle
from typing import Dict, Tuple, List
from lightning import LightningModule
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataflux import DEBUG_MODE
from Dataflux.pytorch import collate_fn, DatasetTypes
from Dataflux.pytorch import Dataset

from Comprisk.anomaly import ScoreTransformer
from Comprisk.helper.structure import load_model
from Comprisk.model import BaseModel

from Dataflux import logging_mode
from Dataflux.logger import get_logger


# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


def store_model_output(
    names: List[str], container: Dict[str, np.ndarray], values: np.ndarray
) -> None:
    """Store the provided values to the container dictionary with the keys equal to the
    names list.

    Args:
        names (List[str]): The keys of the dictionary to store values under.
        container (Dict[str, list]): The dictionary to store the values under.
        values (np.array): The values to store under the keys of names.
    """
    for j, key in enumerate(names):
        container[key] += list(values[:, j])


class ModelLoader:
    """This class handles loading the model and related objects/configurations from the model's
    output directory. The loading is handled by Comprisk.helper.structure.load_model

    Args:
        path (str): The path where the model is saved. Defaults to `None`.
        model (BaseModel): The model to use. Defaults to `None`.

    > Note: Make sure to either set `path` or `model`. If both were set, then the provided `model` will be used.
    """

    def __init__(self, path: str = None, model: BaseModel = None) -> None:
        if not path and not model:
            raise ValueError("Expected to either get a path or model. Got None.")

        self._path = path
        self.model: BaseModel = model or load_model(path, eval_mode=False)

        if not self.model:
            raise ValueError("Unable to load model")
        logger.info("ModelLoader fully initialized.")

    @property
    def path(self) -> str:
        """The path where the loaded model was loaded from. The loading process was handled by
        Comprisk.helper.structure.load_model"""
        return self._path

    def _init_container(
        self, length: int, features: int, window: int = 1, stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sensor_shape = (length * stride + window, features)
        value_shape = (length * stride + window,)

        targets = np.zeros(shape=sensor_shape, dtype=np.float32)
        predict = np.zeros(shape=sensor_shape, dtype=np.float32)
        counter = np.zeros(shape=value_shape, dtype=np.float32)
        ts_index = np.zeros(shape=value_shape, dtype=np.float64)

        return targets, predict, counter, ts_index

    def predict_loop(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This method applies a model on a given dataset. For an AutoEncoder, the reconstructed
        output will be returned. For future prediction models, the prediction of the future is
        output. In addition, the original input data will be temporarily stored and returned.

        Args:
            dataset (Dataset): The dataset to use.
            batch_size (int): The amount of sample to process at a time. Defaults to `64`.
            num_workers (int): The number of threads feeding data from the dataset to the model. Defaults to `4`.

        Returns
            Tuple[pd.DataFrame, pd.DataFrame]: Pandas dataframes containing the (1) input and (2) prediction.
        """
        self.model.eval()
        device = self.model.device
        args = dataset.get_args()
        self.model.set_output_behavior(args.behavior, args.sequence_out)
        skip = args.skip
        stride = args.stride
        features = args.features_out

        # depending on the dataset type, different window sizes are possible
        ts_offset = 0
        window = args.sequence_in
        if args.behavior == DatasetTypes.PREDICTIVE:
            window = args.sequence_out
            ts_offset = args.sequence_in * skip

        logger.debug("Predict on dataset %s (Original)", len(dataset))
        logger.debug("Predict loop step size %s and batch size %s", skip, batch_size)
        logger.debug("Window size %s with stride %s", window, stride)

        # initialize the containers which hold model input/output values
        containers = self._init_container(len(dataset), features, window, stride)
        targets, predict, counter, ts_index = containers

        collate = collate_fn
        loader = DataLoader(
            dataset,
            batch_size,
            collate_fn=collate,
            num_workers=num_workers,
            drop_last=False,
        )

        for i, inputs in enumerate(
            tqdm(loader, desc="Prediction with Model", disable=not DEBUG_MODE)
        ):
            inputs, expected_outputs = inputs
            expected_outputs = expected_outputs.cpu().numpy()
            inputs = inputs.to(device)

            # run the prediction, the input equals the target for auto encoders
            outputs = self.model.predict(inputs, as_array=True)

            batch_size_, _, _ = outputs.shape
            for j in range(0, batch_size_):
                idx = 0 if i == 0 else i * batch_size * stride
                idx = idx + j * stride

                window_, _ = outputs[j].shape
                window_ = min(window_, len(targets[idx : idx + window_]))
                targets[idx : idx + window_] += expected_outputs[j]
                predict[idx : idx + window_] += outputs[j]
                counter[idx : idx + window_] += np.ones(shape=(window_,))

                ts = dataset.get_timestamp(i * batch_size + j) + ts_offset
                ts_index[idx : idx + window_] = [ts + k * skip for k in range(window_)]

        limit = counter.argmin()
        limit = limit if counter.min() != counter.mean() else -1
        ts_index = pd.Index(ts_index[:limit], name="ts")

        counter = counter[:limit]
        targets = targets[:limit]
        predict = predict[:limit]

        # create the output DataFrames
        names = args.names_out
        targets = pd.DataFrame(
            {n: targets[:, i] / counter for i, n in enumerate(names)}, index=ts_index
        )
        predict = pd.DataFrame(
            {n: predict[:, i] / counter for i, n in enumerate(names)}, index=ts_index
        )

        return targets, predict

    def get_score_transformer(self) -> ScoreTransformer:
        """A simple getter for the score transformer. This method loads the transformer from
        disk and returns it.
        """
        score_path = os.path.join(self._path, "metadata", "score_transformer.pkl")
        if os.path.exists(score_path):
            with open(score_path, "rb") as f:
                return pickle.load(f)
        raise LookupError(
            "Could not find 'score_transformer.pkl' in model metadata path."
        )

    def get_model(self) -> LightningModule:
        """A simple getter for the loaded model during object initialization.

        Returns:
            Any: The loaded model.
        """
        return self.model
