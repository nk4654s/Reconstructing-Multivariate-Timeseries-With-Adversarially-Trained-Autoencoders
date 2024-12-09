"""This module replaces the sensor UUID with a human readable text."""
from enum import Enum, auto
import os
from typing import Dict
import pandas as pd

from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency, CALLABLE_RETURN_TYPE
from Dataflux.helper import Metadata, Meta, get_sensor_uuid_to_name
from Dataflux.modules.normalize import NormalizeFit

from Dataflux import logger


class ColumnContext(Enum):
    HEADER = auto()


class Rename(FileWorkerCallable):
    """
    A pre-processing step for renaming columns in a dataset. This class provides
    methods and attributes for renaming columns based on specific rules or
    configurations.

    Args:
        enable (bool): Enable renaming, defaults to True.
    """
    def __init__(self, enable: bool = True) -> None:
        self._sensor_names = None
        self._names = set()

        self._enable = enable

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {
            NormalizeFit.__name__: CallableDependency.NOT_BEFORE
        }

    def run(self, file_path: str, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        df = pd.read_hdf(file_path, key="data")
        if df.empty or isinstance(df, pd.Series):
            return None, [file_path]

        if self._sensor_names is None:
            logger.debug("Module enabled: %s", "yes" if self._enable else "no")

            # initial call of this module, download uuid -> name dictionary
            metadata = Metadata()
            metadata.load(os.path.dirname(file_path))

            uuid = metadata[Meta.MACHINE_UUID]
            self._sensor_names = get_sensor_uuid_to_name(uuid)
            assert self._sensor_names, "No sensor name dictionary available."

        if self._enable:
            df.rename(columns=self._sensor_names, inplace=True)
        
        df.to_hdf(file_path, key="data", format="table")
        if len(self._names) == 0:
            self._names = set(df.columns.values)
        else:
            self._names = self._names.intersection(set(df.columns.values))
        return None, None

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        context.put(ColumnContext.HEADER, self._names)
        logger.debug("Found header intersection %s", self._names)
