import os
from threading import Lock
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from Dataflux.helper import Metadata, Meta
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency
from Dataflux import logger

from .normalize import NormalizeTransform

mutex = Lock()


class ContextParam:
    """Represents context parameters for outlier operations.
    
    This class provides constants for values and headers used
    in outlier operations.
    """
    VALUES = "Outlier_values"
    IGNORE = "Outlier_ignore_columns"


class OutlierFit(FileWorkerCallable):
    """Represents a fit operation for outlier detection.
    
    This class provides methods and constants for performing
    outlier detection. To remove the detected outliers, use the `OutlierTransform`
    class.

    Args:
        ignore (str): List of columns to ignore, proved as string of column names
            separated by a ','.
    """
    def __init__(self, ignore: str = ""):
        self._ignore = set([_.strip() for _ in ignore.replace("\"", "").split(",")])
        self._means = []
        self._stds = []

        logger.debug("Ignoring following columns for outlier detection: %s", self._ignore)

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {OutlierTransform.__name__: CallableDependency.IMMEDIATELY_AFTER,
                NormalizeTransform.__name__: CallableDependency.NOT_BEFORE}

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        # Load the file into a pandas DataFrame
        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1) # type: ignore

        include = list(set(df.columns.values) - self._ignore)

        self._means.append(df[include].mean())
        self._stds.append(df[include].std())

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        # pass the trained variables to the next callable
        with mutex:
            mean = sum(self._means) / len(self._means)
            std = sum(self._stds) / len(self._stds)

            context.put(ContextParam.VALUES, (mean, std))
            context.put(ContextParam.IGNORE, self._ignore)

            logger.debug("Outlier mean \n%s", mean)
            logger.debug("Outlier std \n%s", std)


class OutlierTransform(FileWorkerCallable):
    """
    Removes detected outliers found by the `OutlierFit` class.
    
    It assumes that the `OutlierFit` class has been run beforehand, and the
    `FileWorkerContext` contains values needed to normalize a dataframe.

    Args:
        sigma (float): The sigma is used in the following calculation:
            mean +- sigma * std
        enable (bool): Enable or disable this module, Defaults to True.
    """
    def __init__(self, sigma: float = 3.0, enable: bool = True) -> None:
        self._sigma = sigma
        self._enable = enable
        self._removed_files = []
        self._lengths_files = {}
        logger.debug("Outlier detection uses sigma %s", sigma)

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {OutlierFit.__name__: CallableDependency.IMMEDIATELY_BEFORE}

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        if self._enable is not True or self._enable != "True":
            return None

        df = pd.read_hdf(file_path, key="data")
        df = df.reindex(sorted(df.columns), axis=1) # type: ignore

        include = list(set(df.columns.values) - context.get(ContextParam.IGNORE))

        # remove files, which have outliers
        mean, std = context.get(ContextParam.VALUES)
        sigma = self._sigma * std
        df[(df[include] < (mean - sigma)) | (df[include] > (mean + sigma))] = np.nan
        df = df.dropna(how="any")

        day = file_path.split("_", maxsplit=1)[-1][:-3]
        if df.empty:
            self._removed_files.append(day)
            os.remove(file_path)
        else:
            self._lengths_files[day] = len(df)
            df.to_hdf(file_path, key="data")

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        if self._enable is not True or self._enable != "True":
            return None

        context.clear(ContextParam.VALUES)

        new_files = []
        for f in metadata[Meta.FILES]:
            day = f["day"]
            if day in self._removed_files:
                continue

            if day in self._lengths_files.keys():
                f["recordings"] = self._lengths_files[day]
                new_files.append(f)

        metadata.drop(Meta.FILES)
        metadata.put(new_files, Meta.FILES)
