"""This module contains helper methods related to the models."""
import operator
import os
import json
import pickle
from typing import Dict, Tuple, Type, List, Union
from uuid import UUID
from lightning import LightningModule
import numpy as np
import pandas as pd
import yaml

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataflux.pytorch import online_collate_inference, collate_fn
from Dataflux.helper import get_logger, get_sensor_uuid_to_name
from Dataflux.pytorch import Dataset

from Comprisk.anomaly import ScoreTransformer
from Comprisk.helper.file_structure import class_for_name, get_checkpoints
from Comprisk.model.base import BaseModel

from Comprisk import logger


def set_model_features(model, dataset: Dataset):
    # set correct feature names to model, the feature names of the dataset should be
    # sorted according to the sample's feature indices
    feature_dict = get_sensor_uuid_to_name(UUID(dataset.UUID))
    feature_names = dataset.get_args().names
    if feature_dict:
        feature_names = [feature_dict.get(n, n) for n in feature_names]

    # remove the 'ts' value from the model's feature set as it should not use it
    if "ts" in feature_names:
        feature_names.remove("ts")
        logger.debug("Removed 'ts' from model's feature set.")

    model.set_feature_names(feature_names)


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
            if drop == "_class_path":
                continue
            del hparams[drop]

        for key, value in hparams.items():
            if drop == "_class_path":
                continue

            if isinstance(value, dict):
                libpath = value["class_path"]
                module = class_for_name(*libpath.rsplit(".", maxsplit=1))
                sub_modules[key] = module(**value["init_args"])
            else:
                sub_modules[key] = value

        model_class = class_for_name(*hparams["_class_path"].rsplit(".", maxsplit=1))
        return model_class, sub_modules


def load_model(path: str, eval_mode: bool = True) -> Union[BaseModel, None]:
    """This method loads the best checkpoint of a model. Requires checkpoints of model to have
    the filename structure `...val=<validation_loss>.ckpt`.

    Args:
        cli (CompriskCLI): The cli object created when starting an entrypoint.
        eval_mode (bool): Weather to set the loaded model to eval mode or not. Defaults to True.
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



class ModelLoader:
    """This class handles loading the model and related objects/configurations from the model's
    output directory.

        Args:
            path (str): The path where the model is saved.
    """
    def __init__(self, path: str) -> None:
        self._path = path
        model = load_model(path, eval_mode=False)
        assert model, f"Unable to load model, no checkpoints found for {path}."
        self.model: BaseModel = model

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

    def predict_loop(self, dataset: Dataset, modifier: Tuple[str, float] = ("", 1.0), batch_size: int = 64,
                     step: int | None = None, num_workers: int = 4, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This method applies a model on a given dataset. For an AutoEncoder, the reconstructed
        output will be returned. For future prediction models, the prediction of the future is
        output. In addition, the original input data will be temporarily stored and returned.

        Args:
            dataset (Dataset): The dataset to use.
            modifier (Tuple[str, float]): The sensor name and factor to apply. This is handy to inject
                some kind of anomaly.
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
        window = args.sequence_out
        features = args.features_out
        # output_length = batch_size * window - batch_size * stride + window // 2
        # assert window == stride, f"Something went wrong, the dataset's stride {stride} should equal the window size {window}."

        logger.debug("Predict on dataset %s (Original)", len(dataset))
        logger.debug("Predict loop step size %s and batch size %s", step, batch_size)
        logger.debug("Window size %s with stride %s", window, stride)

        # 0. find the index of the sensor to modify with the given factor
        modifier_name, factor = modifier
        modifier_index = -1
        if modifier_name:
            modifier_index = operator.indexOf(args.names, modifier_name)

        # 1. iterate over dataset, with shuffle deactivated
        targets = np.zeros(shape=(len(dataset) * stride + window, features), dtype=np.float32)
        predict = np.zeros(shape=(len(dataset) * stride + window, features), dtype=np.float32)
        counter = np.zeros(shape=(len(dataset) * stride + window, ), dtype=np.float32)
        ts_indx = np.zeros(shape=(len(dataset) * stride + window, ), dtype=np.float32)

        collate = online_collate_inference if args.key == "data" else collate_fn
        loader = DataLoader(dataset, batch_size, collate_fn=collate, num_workers=num_workers, drop_last=False)
        for i, inputs in enumerate(tqdm(loader, desc="Prediction with Model", disable=not verbose)):
            inputs, scaler = inputs
            inputs = inputs.to(device)

            # modify the sensor if required, e.g. if an anomaly should be injected
            if modifier_name:
                inputs[:, :, modifier_index] = factor

            # run the prediction, the input equals the target for auto encoders
            model_pred = self.model.predict(inputs, as_array=True)
            inputs = inputs.cpu().numpy()

            # the scaler only exists, if the model uses a dataloader with 'online_collate'
            # this happens if the dataset's key is 'data'
            if args.key == "data":
                model_pred = scaler.inverse_transform(model_pred) # type: ignore
                inputs = scaler.inverse_transform(inputs)

            batch_size_, _, _ = model_pred.shape
            for j in range(0, batch_size_):
                idx = 0 if i == 0 else i * batch_size * stride
                idx = idx + j * stride

                window_, _ = inputs[j].shape
                targets[idx:idx + window_] += inputs[j]
                predict[idx:idx + window_] += model_pred[j]
                counter[idx:idx + window_] += np.ones(shape=(window_, ))

                ts = dataset.get_timestamp(i * batch_size + j)
                ts_indx[idx:idx + window_] = [ts + k * skip for k in range(window_)]

        limit = counter.argmin()
        ts_index = pd.Index(ts_indx[:limit], name="ts")

        counter = counter[:limit]
        targets = targets[:limit]
        predict = predict[:limit]

        # create the output DataFrames
        names = sorted(args.names)
        targets = pd.DataFrame({n: targets[:, i] / counter for i, n in enumerate(names)}, index=ts_index)
        predict = pd.DataFrame({n: predict[:, i] / counter for i, n in enumerate(names)}, index=ts_index)

        # todo: recursive call to iterate over leftover then merge dataframes
        return targets, predict

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
        return self.model

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
