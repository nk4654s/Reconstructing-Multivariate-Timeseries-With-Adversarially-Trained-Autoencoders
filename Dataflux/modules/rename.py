"""This module replaces the sensor UUID with a human readable text."""
from enum import Enum, auto
import os
from typing import Any, Dict
import pandas as pd

from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency
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
    """
    def __init__(self) -> None:
        self._sensor_names = None
        self._names = set()

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {
            NormalizeFit.__name__: CallableDependency.NOT_BEFORE
        }

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        df = pd.read_hdf(file_path, key="data")
        if df.empty or isinstance(df, pd.Series):
            return

        if self._sensor_names is None:
            # initial call of this module, download uuid -> name dictionary
            metadata = Metadata()
            metadata.load(os.path.dirname(file_path))

            uuid = metadata[Meta.MACHINE_UUID]
            self._sensor_names = get_sensor_uuid_to_name(uuid)
            assert self._sensor_names, "No sensor name dictionary available."

        names: Dict[str, str] = {id: self._sensor_names.get(id, id) for id in list(df.columns.values)}
        df.rename(columns=names, inplace=True)
        logger.debug("Renamed columns.")
        
        df.to_hdf(file_path, key="data", format="table")
        if len(self._names) == 0:
            self._names = set(df.columns.values)
        else:
            self._names = self._names.intersection(set(df.columns.values))

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        context.put(ColumnContext.HEADER, self._names)
        logger.debug("Found header intersection %s", self._names)