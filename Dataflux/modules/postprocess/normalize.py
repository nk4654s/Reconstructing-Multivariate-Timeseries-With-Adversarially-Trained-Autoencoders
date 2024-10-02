import os
from threading import Lock
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pickle

from Dataflux.helper import Meta, Metadata, FluxScaler, ScalerMethod
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency

mutex = Lock()


class ContextParam:
    """Represents context parameters for normalization operations.
    
    This class provides constants for values and headers used
    in normalization operations.
    """
    SCALER = "normalize-context"


class NormalizeFit(FileWorkerCallable):
    """Represents a fit operation for normalization.
    
    This class provides methods and constants for performing
    normalization fits. To apply a normalization, use the `NormalizeTransform`
    class.
    
    Attributes:
        METHODS (list): A list of supported normalization methods. For supported methods, see
            `Dataflux.helper.ScalerMethod`.
    
    Args:
        method (str): The normalization method to use. Defaults to `minmax`. For supported
            methods, see `Dataflux.helper.ScalerMethod`.
        exclude_keys (list): A list of keys to exclude from normalization.
    """
    METHODS = {m.value: m for m in ScalerMethod}

    def __init__(self, method: str = "minmax", exclude_keys: List[str] = ["ts"]):
        assert method in self.METHODS.keys(), f"Scaling method {method} is not supported."
        self._method = self.METHODS[method]
        self._exclude = exclude_keys

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {NormalizeTransform.__name__: CallableDependency.IMMEDIATELY_AFTER}

    def _get_scaler(self, context: FileWorkerContext) -> FluxScaler:
        # get minimum and maximum values from previous files
        scaler = context.get(ContextParam.SCALER)
        if scaler is None:
            scaler = FluxScaler(self._method, self._exclude)

        return scaler

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        # Load the file into a pandas DataFrame, sort by column names to ensure that
        # dataframe is consistent
        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1) # type: ignore

        with mutex:
            # get minimum and maximum values from previous files
            scaler = self._get_scaler(context)
            scaler.fit(df)

            context.put(ContextParam.SCALER, scaler)

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        pass


class NormalizeTransform(FileWorkerCallable):
    """
    Represents a transformation operation for normalization. This class is responsible
    for transforming data into the normalized form.
    
    It assumes that the `NormalizeFit` class has been run beforehand, and the
    `FileWorkerContext` contains values needed to normalize a dataframe.

    Args:
        from_other (str): Another dataset to use the normalizer from.
    """
    def __init__(self, from_other: str = "") -> None:
        self._scaler = None
        if from_other:
            m = Metadata()
            m.load(from_other)

            norm_meta = m[Meta.NORMALIZER]
            path = norm_meta[Meta.NORM_PATH.name]

            with open(path, "rb") as f:
                self._scaler = pickle.load(f)

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {NormalizeFit.__name__: CallableDependency.IMMEDIATELY_BEFORE}

    def _get_scaler(self, context: FileWorkerContext):
        if self._scaler:
            return self._scaler

        scaler = context.get(ContextParam.SCALER)
        return scaler

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        # read the file as pandas dataframe and sort by column names
        # to be consistent with the fitted values
        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1) # type: ignore

        # get the used normalization method and the fitted values
        scaler = self._get_scaler(context)
        normed = scaler.transform(df, dtype=np.float32)

        normed.to_hdf(file_path, key="norm")

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        # get the fitted values and method
        scaler: FluxScaler = self._get_scaler(context)

        # clear the context parameters as they are not needed anymore
        context.clear(ContextParam.SCALER)

        # save the normalizer
        path = os.path.join(metadata[Meta.PATH], "metadata", "normalizer.pkl")
        with open(path, "wb") as f:
            pickle.dump(scaler, f)

        # save the used normalization method and fitted values to the
        # dataset's metadata
        data = {
            Meta.NORM_METHOD.name: scaler.method.value,
            Meta.NORM_PATH.name: path,
        }
        metadata.put(data, Meta.NORMALIZER)
