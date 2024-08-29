from argparse import ArgumentError
import os
from threading import Lock
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from Dataflux.helper import Meta, Metadata
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency
from Dataflux import logger

mutex = Lock()


class ContextParam:
    """Represents context parameters for normalization operations.
    
    This class provides constants for values and headers used
    in normalization operations.
    """
    SCALER = "TrainNormalize_recorder"
    HEADER = "TrainNormalize_header"
    METHOD = "TrainNormalize_method"


class NormalizeFit(FileWorkerCallable):
    """Represents a fit operation for normalization.
    
    This class provides methods and constants for performing
    normalization fits. To apply a normalization, use the `NormalizeTransform`
    class.
    
    Attributes:
        METHODS (list): A list of supported normalization methods.
        [`minmax`, `standard`]
    """
    METHODS = ["minmax", "standard", "robust"]

    def __init__(self, method: str = "minmax"):
        self._method = method
        self._header = None
        assert self._method in NormalizeFit.METHODS, f"{method} is not supported."

    def get_dependencies(self) -> List[str]:
        return {NormalizeTransform.__name__: CallableDependency.IMMEDIATELY_AFTER}

    def _get_scaler(self, context: FileWorkerContext):
        with mutex:
            # get minimum and maximum values from previous files
            recorder = context.get(ContextParam.SCALER)
            if recorder is None:
                if self._method == "minmax":
                    recorder = MinMaxScaler(feature_range=(-1, 1))
                if self._method == "standard":
                    recorder = StandardScaler()
                if self._method == "robust":
                    recorder = RobustScaler()

            fit = recorder.fit
            if self._method != "robust":
                fit = recorder.partial_fit
            return recorder, fit

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        print("trainnorm:", file_path)
        # Load the file into a pandas DataFrame, sort by column names to ensure that
        # dataframe is consistent
        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1)
        self._header = df.columns.values

        # get minimum and maximum values from previous files
        recorder, fit = self._get_scaler(context)
        fit(df)

        with mutex:
            context.put(ContextParam.SCALER, recorder)

    def finish(self, context: FileWorkerContext, _: Metadata) -> None:
        # pass the fitted values to the next callable
        with mutex:
            context.put(ContextParam.METHOD, self._method)
            context.put(ContextParam.HEADER, self._header)


class NormalizeTransform(FileWorkerCallable):
    """
    Represents a transformation operation for normalization. This class is responsible
    for transforming data into the normalized form.
    
    It assumes that the `NormalizeFit` class has been run beforehand, and the
    `FileWorkerContext` contains values needed to normalize a dataframe.

    Args:
        from_other (str): Another dataset to use the normalizer from.
    """
    def __init__(self, from_other: str = None) -> None:
        self._scaler = None
        if from_other:
            m = Metadata()
            m.load(from_other)

            norm_meta = m[Meta.NORMALIZER]
            path = norm_meta[Meta.NORM_PATH.name]

            with open(path, "rb") as f:
                self._scaler = pickle.load(f)
                self._scaler.set_output(transform="pandas")

    def get_dependencies(self) -> List[str]:
        return {NormalizeFit.__name__: CallableDependency.IMMEDIATELY_BEFORE}

    def _get_scaler(self, context: FileWorkerContext):
        if self._scaler:
            return self._scaler

        scaler = context.get(ContextParam.SCALER)
        scaler.set_output(transform="pandas")
        return scaler

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        print("applynorm:", file_path)
        # read the file as pandas dataframe and sort by column names
        # to be consistent with the fitted values
        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1)

        # get the used normalization method and the fitted values
        scaler = self._get_scaler(context)

        # apply normalization
        normed = scaler.transform(df)
        normed.astype(np.float32).to_hdf(file_path, key="norm")

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        with mutex:
            # get the fitted values and method
            scaler = self._get_scaler(context)
            method = context.get(ContextParam.METHOD)
            header = context.get(ContextParam.HEADER)

            # clear the context parameters as they are not needed anymore
            context.clear(ContextParam.METHOD)
            context.clear(ContextParam.HEADER)
            context.clear(ContextParam.SCALER)

            # save the normalizer
            path = os.path.join(metadata[Meta.PATH], "metadata", "normalizer.pkl")
            with open(path, "wb") as f:
                pickle.dump(scaler, f)

            # save the used normalization method and fitted values to the
            # dataset's metadata
            data = {
                Meta.NORM_METHOD.name: method,
                Meta.NORM_KEYS.name: [str(_) for _ in header],
                Meta.NORM_PATH.name: path,
            }
            metadata.put(data, Meta.NORMALIZER)
