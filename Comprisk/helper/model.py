"""This module contains helper methods related to the models."""
import operator
import os
import json
import pickle
from typing import Any, Dict, Tuple, Type, List
from lightning import LightningModule
import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
from tqdm import tqdm

from Dataflux.helper import get_logger
from Dataflux.pytorch import Dataset

from Comprisk.anomaly import ScoreTransformer
from Comprisk.helper.file_structure import class_for_name, get_checkpoints

logger = get_logger(__name__, path="./logs/logfile")


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
        for key, value in hparams["init_args"].items():
            if isinstance(value, dict):
                libpath = value["class_path"]
                module = class_for_name(*libpath.rsplit(".", maxsplit=1))
                sub_modules[key] = module(**value["init_args"])
            else:
                sub_modules[key] = value

        model_class = class_for_name(*hparams["class_path"].rsplit(".", maxsplit=1))
        return model_class, sub_modules


def load_model(path: str, eval_mode: bool = True) -> Any:
    """This method loads the best checkpoint of a model. Requires checkpoints of model to have
    the filename structure `...val=<validation_loss>.ckpt`.

    Args:
        cli (CompriskCLI): The cli object created when starting an entrypoint.
        eval_mode (bool): Wether to set the loaded model to eval mode or not. Defaults to True.
    """
    checkpoints = get_checkpoints(path)
    checkpoints = sorted(checkpoints, key = lambda x : float(x[:-5].split("val=")[-1]))
    if len(checkpoints) == 0:
        return None

    # 1. get dataset configuration: start, end, step size (size of input window)
    model_class, sub_modules = init_modules(path)
    model = model_class.load_from_checkpoint(
        os.path.join(path, "checkpoints", checkpoints[0]),
        **sub_modules
    )

    logger.debug("Loaded model %s from %s", checkpoints[0], path)
    if eval_mode:
        model.eval()
    return model


def store_model_output(names: List[str], container: Dict[str, np.ndarray], values: np.ndarray) -> None:
    """Store the provided values to the container dictionary with the keys equal to the 
    names list.

    Args:
        names (List[str]): The keys of the dictionary to store values under.
        container (Dict[str, list]): The dictionary to store the values under.
        values (np.array): The values to store under the keys of names.
    """
    for j, key in enumerate(names):
        container[key] += list(values[:, j])


def get_tensor_from_dataset(dataset: Dataset, i: int, device: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """This method gets a sample from a dataset given the index and transforms it
    to a tensor including moving the tensor to the provided device.

    Args:
        dataset (Dataset): The dataset from which samples should be gathered.
        i (int): The index of the sample to get.
        device (str): The device to move the tensors to (e.g. 'cpu', 'cuda', or 'mps').
    
    Returns:
        Tuple[torch.tensor, torch.tensor]: Sensor and weather data as tensor.
    """
    sensors, weather, _ = dataset[i]

    if device.type == "mps":
        sensors = sensors.astype(np.float32)
        weather = weather.astype(np.float32)

    sensors = torch.tensor(sensors, device=device)
    sensors = torch.unsqueeze(sensors, dim=0)
    weather = torch.tensor(weather, device=device)
    weather = torch.unsqueeze(weather, dim=0)
    return sensors, weather


class ModelLoader:
    """This class handles loading the model and related objects/configurations from the model's
    output directory.

        Args:
            path (str): The path where the model is saved.
    """
    def __init__(self, path: str) -> None:
        self._path = path
        self._model = load_model(path, eval_mode=False)

    @property
    def path(self) -> str:
        """The path where the model is saved.
        """
        return self._path

    def get_correlations(self) -> pd.DataFrame:
        """Loads the correlations of input features from the model metadata folder. These
        correlations are computed from the training data.

        Returns:
            pd.DataFrame: The correlations of input features as pandas DataFrame.
        """
        correlation_path = os.path.join(self._path, "metadata", "correlation.csv")
        if not os.path.exists(correlation_path):
            return pd.DataFrame()

        return pd.read_csv(correlation_path, index_col=0)

    def predict_loop(self, dataset: Dataset, mode: str = "autoencoder",
                     modifier: Tuple[str, float] = ("", 1.0), step: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This method applies a model on a given dataset. For an AutoEncoder, the reconstructed
        output will be returned. For future prediction models, the prediction of the future is
        output. In addition, the original input data will be temporarily stored and returned.

        Args:
            dataset (Dataset): The dataset to use.
            mode (str): The mode to use for prediction. Defaults to 'autoencoder'.
            modifier (Tuple[str, float]): The sensor name and factor to apply. This is handy to inject
                some kind of anomaly.
            step (int | None): The step size of the window to slide over the dataset.
        
        Returns
            Tuple[pd.DataFrame, pd.DataFrame]: Pandas dataframes containing the prediction and input.
        """
        self._model.eval()
        meta = dataset.get_meta()
        if not step:
            step = meta.num_input_sequence if mode == "autoencoder" else meta.num_output_sequence

        logger.debug("Predict on dataset, length=%s", len(dataset))
        logger.debug("Dataset input sequence: %s", meta.num_input_sequence)
        logger.debug("Dataset output sequence: %s", meta.num_output_sequence)
        logger.debug("Predict loop step size: %s", step)

        # 0. find the index of the sensor to modify with the given factor
        modifier_name, factor = modifier
        modifier_index = -1
        if modifier_name:
            modifier_index = operator.indexOf(meta.column_names_sensor, modifier_name)

        # 1. iterate over dataset, with shuffle deactivated
        counts = np.zeros(shape=(len(dataset), len(meta.column_names_sensor)))
        predicted = np.zeros(shape=(len(dataset), len(meta.column_names_sensor)))
        inputs = np.zeros(shape=(len(dataset), len(meta.column_names_sensor)))

        ts_index = []

        for i in tqdm(range(0, len(dataset), step), desc="Prediction with Model"):
            # 2. move the data to the correct device and pass the data to the model
            sensors, weather = get_tensor_from_dataset(dataset, i, self._model.device)
            if modifier_index > -1:
                sensors[:, :, modifier_index] *= factor

            target = self._model.predict({"sensors": sensors, "weather": weather}, True)[0]

            # 3. store the model inputs/outputs to a temporary dict
            window = len(target) if i + len(target) < len(dataset) else len(dataset) - i
            counts[i:i + window] += np.ones_like(target)[:window]
            predicted[i:i + window] = target[:window]
            inputs[i:i + window] = sensors.cpu().numpy()[0][:window]

            ts_index += [dataset.get_timestamp(i) for _ in range(min(step, window))]
        
        predicted /= counts
        inputs /= counts

        # 4. finalize dataframe, give columns names matching input dataframe column names
        index = pd.Index(ts_index, name="ts")
        logger.debug("Creating 'prediction' Dataframe of gathered data")
        predicted = pd.DataFrame({key: predicted[:, j] for j, key in enumerate(meta.column_names_sensor)}, index=index)

        logger.debug("Creating 'inputs' Dataframe of gathered data")
        inputs = pd.DataFrame({key: inputs[:, j] for j, key in enumerate(meta.column_names_sensor)}, index=index)
        return predicted, inputs

    def get_score_transformer(self) -> ScoreTransformer:
        """A simple getter for the score transformer. This method loads the transformer from
        disk and returns it.
        """
        score_path = os.path.join(self._path, "metadata", "score_transformer.pkl")
        if os.path.exists(score_path):
            with open(score_path, "rb") as f:
                return pickle.load(f)
        raise LookupError("Could not find 'score_transformer.pkl' in model metadata path.")

    def get_model(self) -> LightningModule:
        """A simple getter for the loaded model during object initialization.
        
        Returns:
            Any: The loaded model.
        """
        return self._model

    def get_normalization(self) -> pd.DataFrame | None:
        norm_path = os.path.join(self._path, "metadata", "normalizer.json")
        if not os.path.exists(norm_path):
            return None

        norm_data = {}
        with open(norm_path, "r", encoding="UTF-8") as f:
            norm_file = json.load(f)
            for i, key in enumerate(norm_file["NORM_KEYS"]):
                norm_data[key] = (norm_file["NORM_VAL_A"][i],
                                  norm_file["NORM_VAL_B"][i])

        # TODO: should create a custom Normalize-class to handle this?
        norm_data = pd.DataFrame(norm_data)
        return norm_data
