from threading import Lock
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from Dataflux.helper import Meta, Metadata
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency
from Dataflux import logger

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
    METHODS = ["minmax"]

    def __init__(self, method: str = "minmax"):
        self._method = method
        self._header = None
        assert self._method in NormalizeFit.METHODS, f"{method} is not supported."

    def get_dependencies(self) -> List[str]:
        return {NormalizeTransform.__name__: CallableDependency.IMMEDIATELY_AFTER}

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        print("trainnorm:", file_path)
        # Load the file into a pandas DataFrame, sort by column names to ensure that
        # dataframe is consistent
        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1)
        self._header = df.columns.values

        with mutex:
            # get minimum and maximum values from previous files
            values = context.get(ContextParam.VALUES)
            if values is None:
                # if this is the first file, simply use the current min/max values
                df_min = df.min().to_numpy()
                df_max = df.max().to_numpy()

                # pass fitted values to next files
                context.put(ContextParam.VALUES, (df_min, df_max))

            else:
                mins, maxs = values
                df_min = df.min().to_numpy()
                df_max = df.max().to_numpy()

                # update minimum and maximum values for columns which
                # are lower/greater then recorded min/max values
                mins[mins > df_min] = df_min[mins > df_min]
                maxs[maxs < df_max] = df_max[maxs < df_max]

                # pass fitted values to next files
                context.put(ContextParam.VALUES, (mins, maxs))

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

        # minmax method stores min/max values
        if method == NormalizeFit.METHODS[0]:
            mins, maxs = values
            return mins, maxs

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        print("applynorm:", file_path)
        # read the file as pandas dataframe and sort by column names
        # to be consistent with the fitted values
        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1)

        # get the used normalization method and the fitted values
        method = context.get(ContextParam.METHOD)
        mins, maxs = self._get_single_norm_params(context)

        # apply normalization according to method
        if method == NormalizeFit.METHODS[0]:
            normed = ((df - mins).div(maxs - mins) - 0.5).div(0.5).fillna(0)

        normed.astype(np.float32).to_hdf(file_path, key="norm")

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        with mutex:
            # get the fitted values and method
            a, b = self._get_single_norm_params(context)
            method = context.get(ContextParam.METHOD)
            header = context.get(ContextParam.HEADER)

            # clear the context parameters as they are not needed anymore
            context.clear(ContextParam.METHOD)
            context.clear(ContextParam.HEADER)
            context.clear(ContextParam.VALUES)

            # save the used normalization method and fitted values to the
            # dataset's metadata
            data = {
                Meta.NORM_METHOD.name: method,
                Meta.NORM_KEYS.name: [str(_) for _ in header],
                Meta.NORM_VAL_A.name: [float(_) for _ in a],
                Meta.NORM_VAL_B.name: [float(_) for _ in b],
            }
            metadata.put(data, Meta.NORMALIZER)
