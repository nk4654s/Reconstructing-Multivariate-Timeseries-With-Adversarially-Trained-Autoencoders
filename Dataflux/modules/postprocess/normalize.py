from threading import Lock
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from Dataflux.helper import Meta, Metadata, get_logger
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency

logger = get_logger(__name__)
mutex = Lock()


class ContextParam:
    """Represents context parameters for normalization operations.
    
    This class provides constants for values and headers used
    in normalization operations.
    """
    VALUES = "TrainNormalize_values"
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
    METHODS = ["minmax", "standard"]

    def __init__(self, method: str = "minmax"):
        self._method = method
        self._header = None
        self._a, self._b = None, None
        self._counter = []
        assert self._method in NormalizeFit.METHODS, f"{method} is not supported."

    def _fit_minmax(self, a: np.array, b: np.array) -> None:
        """Keeps track of the minimum and maxium values in the arrays a and b.

        Args:
            a (np.array): Array with minima.
            b (np.array): Array with maxima.
        """
        if self._a is None:
            self._a = a
        else:
            self._a[a < self._a] = a[a < self._a]

        if self._b is None:
            self._b = b
        else:
            self._b[b > self._b] = b[b > self._b]

    def _fit_standard(self, a: np.array, b: np.array) -> None:
        if self._a is None:
            self._a = [a]
            self._b = [b]
            return

        self._a.append(a)
        self._b.append(b)

    def get_dependencies(self) -> List[str]:
        return {NormalizeTransform.__name__: CallableDependency.IMMEDIATELY_AFTER}

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        print("trainnorm:", file_path)
        # Load the file into a pandas DataFrame
        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1)
        self._header = df.columns.values

        # Calculate minimum and maximum per column
        if self._method == NormalizeFit.METHODS[0]:
            df_min = df.min(axis=0).to_numpy()
            df_max = df.max(axis=0).to_numpy()
            self._fit_minmax(df_min, df_max)

        # Calculate mean and std per columns
        elif self._method == NormalizeFit.METHODS[1]:
            df_mean = df.to_numpy().mean(axis=0)
            df_std = df.to_numpy().std(axis=0)
            self._counter.append(len(df))
            self._fit_standard(df_mean, df_std)

    def finish(self, context: FileWorkerContext, _: Metadata) -> None:
        # pass the trained variables to the next callable
        with mutex:
            context.put(ContextParam.METHOD, self._method)
            context.put(ContextParam.HEADER, self._header)
            if self._method == NormalizeFit.METHODS[0]:
                logger.debug("Normalize with MinMax and min=%s; max=%s", self._a, self._b)
                context.append_list(ContextParam.VALUES, (self._a, self._b))

            # Calculate mean and std per file
            elif self._method == NormalizeFit.METHODS[1]:
                logger.debug("Normalize with STD, counted %s values per file", self._counter)
                a = np.array(self._a)
                b = np.array(self._b)

                factors = np.array(self._counter) / sum(np.array(self._counter))
                factors = np.repeat([factors], a.shape[1], axis=0)
                factors = np.swapaxes(factors, 0, 1)

                a = (a * factors).sum(axis=0)
                b = (b * factors).sum(axis=0)

                logger.debug("Normalize with STD and mean=%s", a)
                logger.debug("Normalize with STD and std=%s", b)
                context.append_list(ContextParam.VALUES, (a, b))


class NormalizeTransform(FileWorkerCallable):
    """
    Represents a transformation operation for normalization. This class is responsible
    for transforming data into the normalized form.
    
    It assumes that the `NormalizeFit` class has been run beforehand, and the
    `FileWorkerContext` contains values needed to normalize a dataframe.
    """
    def __init__(self, kwargs) -> None:
        pass

    def get_dependencies(self) -> List[str]:
        return {NormalizeFit.__name__: CallableDependency.IMMEDIATELY_BEFORE}

    def _get_single_norm_params(
        self, context: FileWorkerContext
    ) -> Tuple[np.array, np.array]:
        method = context.get(ContextParam.METHOD)
        values = np.array(context.get(ContextParam.VALUES))

        # minmax
        if method == NormalizeFit.METHODS[0]:
            a = values[:, 0].min(axis=0)
            b = values[:, 1].max(axis=0)
            return a, b

        # standard
        elif method == NormalizeFit.METHODS[1]:
            a = values[:, 0].mean(axis=0)
            b = values[:, 1].mean(axis=0)
            return a, b

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        print("applynorm:", file_path)
        a, b = self._get_single_norm_params(context)
        method = context.get(ContextParam.METHOD)
        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1)

        if method == NormalizeFit.METHODS[0]:
            normed = (df - a).div(b - a).fillna(0)
        elif method == NormalizeFit.METHODS[1]:
            normed = (df - a).div(b).fillna(0)

        normed.astype(np.float32).to_hdf(file_path, key="norm")

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        a, b = self._get_single_norm_params(context)
        method = context.get(ContextParam.METHOD)
        header = context.get(ContextParam.HEADER)
        context.clear(ContextParam.METHOD)
        context.clear(ContextParam.HEADER)
        context.clear(ContextParam.VALUES)

        data = {
            Meta.NORM_METHOD.name: method,
            Meta.NORM_KEYS.name: [str(_) for _ in header],
            Meta.NORM_VAL_A.name: [float(_) for _ in a],
            Meta.NORM_VAL_B.name: [float(_) for _ in b],
        }
        metadata.put(data, Meta.NORMALIZER)
