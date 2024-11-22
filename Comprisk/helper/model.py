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
from tqdm import tqdm, trange

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


class DayPrediction:
    """This class manages the predicted values, e.g. handle overlapping windows and
    build the final DataFrame.
    
    Args:
        ts (List): The timestamps of which the sensor values will be predicted.
        step (int): The step size between two timestamps.
        n_features (int): The amount of features per step.
    """
    def __init__(self, ts: List, step: int, n_features: int) -> None:
        self._ts = ts
        self._step = step

        self._start = 0
        self._prediction = np.zeros((len(ts), n_features))
        self._target = np.zeros((len(ts), n_features))
        self._counts = np.zeros((len(ts), n_features))
    
    def add_values(self, target: np.ndarray, prediction: np.ndarray) -> None:
        """This method adds values to this class. Also, this method keeps track about the
         amount of additions in the overlapping sections.

        Args:
            target (np.ndarray): The model's input.
            prediction (np.ndarray): The model's output.
        """
        end = min(self._start + len(target), len(self._ts))
        size = end - self._start

        self._target[self._start:end] += target[:size]
        self._prediction[self._start:end] += prediction[:size]
        self._counts[self._start:end] += np.ones_like(prediction[:size])

        self._start += self._step
    
    def calculate(self, names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This method calculates the average of the overlapping values using the previously
        created array, which counts the amount of additions. Then, this method creates the
        DataFrames of the arrays.

        Args:
            names (List): The column names of the DataFrame.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Target and Prediction.
        """
        # fix zero division
        self._target[self._counts == 0] = 0
        self._prediction[self._counts == 0] = 0
        self._counts[self._counts == 0] = 1

        target = self._target / self._counts
        data = {names[i]: target[:, i] for i in range(len(names))}
        target = pd.DataFrame(data, index=pd.Index(self._ts, name="ts"))

        prediction = self._prediction / self._counts
        data = {names[i]: prediction[:, i] for i in range(len(names))}
        prediction = pd.DataFrame(data, index=pd.Index(self._ts, name="ts"))

        return target, prediction


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

    def _prepare_container(self, dataset: Dataset, safe_length: int, step: int) -> Dict[str, DayPrediction]:
        meta = dataset.get_meta()

        # prepare timestep recording
        last_day = dataset.get_day(0)[0]
        ts_start = dataset.get_timestamp(0)
        ts_index = []
        read_len = 0

        days = {}

        for i in trange(0, safe_length, desc="Prepare Prediction"):
            day, _ = dataset.get_day(i)
            read_len += 1
            if day != last_day or i == safe_length - 1:
                offset = ts_start + (meta.num_input_sequence * meta.skip) // 2
                day_indicies = [_ for _ in range(offset, int(offset + read_len * meta.skip), meta.skip)]

                day_info = DayPrediction(day_indicies, step, meta.num_features_sensors)
                days[last_day] = day_info

                ts_index += day_indicies
                ts_start = dataset.get_timestamp(i)
                last_day = day
                read_len = 0
        
        return days

    def predict_loop(self, dataset: Dataset, mode: str = "autoencoder",
                     modifier: Tuple[str, float] = ("", 1.0), batch_size: int = 64,
                     step: int | None = None, init_step: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This method applies a model on a given dataset. For an AutoEncoder, the reconstructed
        output will be returned. For future prediction models, the prediction of the future is
        output. In addition, the original input data will be temporarily stored and returned.

        Args:
            dataset (Dataset): The dataset to use.
            mode (str): The mode to use for prediction. Defaults to 'autoencoder'.
            modifier (Tuple[str, float]): The sensor name and factor to apply. This is handy to inject
                some kind of anomaly.
            step (int | None): The step size of the window to slide over the dataset.
            init_step (int): The initial step when running the predict loop. Default to 0.
            batch_size (int): The amount of sample to process at a time
        
        Returns
            Tuple[pd.DataFrame, pd.DataFrame]: Pandas dataframes containing the prediction and input.
        """
        self._model.eval()
        meta = dataset.get_meta()
        if not step:
            step = meta.num_input_sequence if mode == "autoencoder" else meta.num_output_sequence
        
        # calculate the pool and iterable sizes
        pool = step * batch_size
        safe_length = len(dataset) // pool * pool
        leftover = (len(dataset) - safe_length) // step

        logger.debug("Predict on dataset %s | %s (Iterated | Original)", safe_length, len(dataset))
        logger.debug("Predict loop step size %s and batch size %s", step, batch_size)

        # 0. find the index of the sensor to modify with the given factor
        modifier_name, factor = modifier
        modifier_index = -1
        if modifier_name:
            modifier_index = operator.indexOf(meta.column_names_sensor, modifier_name)
        
        # prepare timestep recording
        day_predictions = self._prepare_container(dataset, safe_length, step)

        # 1. iterate over dataset, with shuffle deactivated
        for i in tqdm(range(0, safe_length, step * batch_size), desc="Prediction with Model"):
            sensors = torch.zeros(batch_size, meta.num_input_sequence, meta.num_features_sensors, device=self._model.device)
            weather = torch.zeros(batch_size, meta.num_input_sequence, meta.num_features_weather, device=self._model.device)
            if modifier_name:
                sensors[:, :, modifier_index] *= factor

            # read from the dataset into a batch
            broke = False
            for j in range(0, batch_size):
                if len(dataset) <= i + j * step:
                    broke = True

                s, w = get_tensor_from_dataset(dataset, i + j, self._model.device)
                sensors[j] = s[0]
                weather[j] = w[0]

            if broke:
                break

            # get average prediction
            target = self._model.predict({"sensors": sensors, "weather": weather}, True)
            _, w, _ = target.shape
            
            # add predictions to the prepared
            for j, elem in enumerate(target):
                idx = i + j * step
                if idx + len(elem) > safe_length:
                    break

                day, _ = dataset.get_day(idx)
                day_predictions[day].add_values(sensors.cpu().numpy()[0], target[j])
        
        # 4. finalize dataframe, give columns names matching input dataframe column names
        inputs = pd.DataFrame()
        predicted = pd.DataFrame()

        names = meta.column_names_sensor
        for key, day in day_predictions.items():
            i, p = day.calculate(names)
            inputs = pd.concat([inputs, i])
            predicted = pd.concat([predicted, p])

        inputs.sort_values(by="ts")
        predicted.sort_values(by="ts")

        # todo: recursive call to iterate over leftover then merge dataframes
        return inputs, predicted

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
