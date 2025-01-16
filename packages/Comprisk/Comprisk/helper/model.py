"""This module contains helper methods related to the models."""

import os
import json
import pickle
from typing import Dict, Tuple, Type, List, Union
from lightning import LightningModule
import numpy as np
import pandas as pd
import yaml

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataflux import DEBUG_MODE
from Dataflux.pytorch import collate_fn, DatasetTypes
from Dataflux.pytorch import Dataset

from Comprisk.anomaly import ScoreTransformer
from Comprisk.helper.file_structure import class_for_name, get_checkpoints
from Comprisk.model.base import BaseModel

from Comprisk import logger


def init_modules(path: str) -> Tuple[Type, Dict[str, nn.Module]]:
    """This method loads the hyperparameters of an already trained model
    and initializes the model's sub_modules if existing.

    Args:
        path (str): The folder path where the model is saved.

    Returns:
        Tuple: The class of the model to load and a dictionary of
            initialized submodules modules.
    """
    hparam_path = os.path.join(path, "hparams.yaml")
    with open(hparam_path, mode="r", encoding="UTF-8") as f:
        hparams = yaml.safe_load(f)
        sub_modules = {}

        for drop in [key for key in hparams.keys() if key.startswith("_")]:
            if drop == "class_path":
                continue
            del hparams[drop]

        for key, value in hparams.items():
            if drop == "class_path":
                continue

            if isinstance(value, dict):
                libpath = value["class_path"]
                module = class_for_name(*libpath.rsplit(".", maxsplit=1))
                sub_modules[key] = module(**value["init_args"])
            else:
                sub_modules[key] = value

        model_class = class_for_name(*hparams["class_path"].rsplit(".", maxsplit=1))
        return model_class, sub_modules


def load_model(path: str, eval_mode: bool = True) -> Union[BaseModel, None]:
    """This method loads the best checkpoint of a model. Requires checkpoints of model to have
    the filename structure `...val=<validation_loss>.ckpt`.

    Args:
        path (str): The path where the model checkpoints were saved.
        eval_mode (bool): Weather to set the loaded model to eval mode or not. Defaults to True.
    """
    checkpoints = get_checkpoints(path)
    checkpoints = sorted(checkpoints, key=lambda x: float(x[:-5].split("val=")[-1]))
    if len(checkpoints) == 0:
        logger.warning("Could not find a checkpoint to load a model from. (%s)", path)
        return None

    # 1. get dataset configuration: start, end, step size (size of input window)
    model_class, sub_modules = init_modules(path)
    model = model_class.load_from_checkpoint(
        os.path.join(path, "checkpoints", checkpoints[0]), **sub_modules
    )

    logger.info("Loaded model %s from %s", checkpoints[0], path)
    if eval_mode:
        model.eval()
    return model


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
    output directory.

        Args:
            path (str): The path where the model is saved.
    """

    def __init__(self, path: str = None, model: BaseModel = None) -> None:
        if not path and not model:
            raise ValueError("Expected to either get a path or model. Got None.")

        self._path = path
        model = model or load_model(path, eval_mode=False)
        assert model, f"Unable to load model, no checkpoints found for {path}."
        self.model: BaseModel = model

    @property
    def path(self) -> str:
        """The path where the model is saved."""
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
            batch_size (int): The amount of sample to process at a time

        Returns
            Tuple[pd.DataFrame, pd.DataFrame]: Pandas dataframes containing the prediction and input.
        """
        self.model.eval()
        device = self.model.device
        args = dataset.get_args()
        self.model.set_output_behavior(args.behavior, args.sequence_out)
        skip = args.skip
        stride = args.stride
        window = (
            args.sequence_out
            if args.behavior != DatasetTypes.AUTOENCODER
            else args.sequence_in
        )
        features = args.features_out

        logger.debug("Predict on dataset %s (Original)", len(dataset))
        logger.debug("Predict loop step size %s and batch size %s", skip, batch_size)
        logger.debug("Window size %s with stride %s", window, stride)

        # initialize the containers which hold model input/output values
        containers = self._init_container(len(dataset), features, window, stride)
        targets, predict, counter, ts_index = containers

        inputs_indicies = [
            i for i, x in enumerate(args.names_in) if x in args.names_out
        ]

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
            inputs, _ = inputs
            inputs = inputs.to(device)

            # run the prediction, the input equals the target for auto encoders
            model_pred = self.model.predict(inputs, as_array=True)
            inputs = inputs.cpu().numpy()

            batch_size_, _, _ = model_pred.shape
            for j in range(0, batch_size_):
                idx = 0 if i == 0 else i * batch_size * stride
                idx = idx + j * stride

                window_, _ = inputs[j].shape
                window_ = min(window_, len(targets[idx : idx + window_]))
                targets[idx : idx + window_] += inputs[j][:, inputs_indicies]
                predict[idx : idx + window_] += model_pred[j]
                counter[idx : idx + window_] += np.ones(shape=(window_,))

                ts = dataset.get_timestamp(i * batch_size + j)
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

    def get_normalization(self) -> pd.DataFrame | None:
        norm_path = os.path.join(self._path, "metadata", "normalizer.json")
        if not os.path.exists(norm_path):
            return None

        norm_data = {}
        with open(norm_path, "r", encoding="UTF-8") as f:
            norm_file = json.load(f)
            for i, key in enumerate(norm_file["NORM_KEYS"]):
                norm_data[key] = (
                    norm_file["NORM_VAL_A"][i],
                    norm_file["NORM_VAL_B"][i],
                )

        # TODO: should create a custom Normalize-class to handle this?
        norm_data = pd.DataFrame(norm_data)
        return norm_data
