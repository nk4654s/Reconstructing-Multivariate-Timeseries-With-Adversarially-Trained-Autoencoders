"""This module replaces the sensor UUID with a human readable text."""
from typing import List
from pandas import DataFrame
from Dataflux.helper import Metadata, Meta, get_sensor_uuid_to_name
from Dataflux.modules.module import BasePreProcessor
from Dataflux import logger


class RenameColumns(BasePreProcessor):
    """
    A pre-processing step for renaming columns in a dataset. This class provides
    methods and attributes for renaming columns based on specific rules or
    configurations.
    """
    def __init__(self) -> None:
        self._sensor_names = None

    def on_download_finished(
        self, df: DataFrame, metadata: Metadata
    ) -> DataFrame | List[DataFrame]:
        if self._sensor_names is None:
            # initial call of this module, download uuid -> name dictionary
            uuid = metadata[Meta.MACHINE_UUID]
            self._sensor_names = get_sensor_uuid_to_name(uuid)

        # apply dictionary to columns
        assert self._sensor_names, "No sensor name dictionary available."
        names = {id: self._sensor_names.get(id, id) for id in list(df.columns.values)}
        df.rename(columns=names, inplace=True)
        logger.debug("Renamed columns.")
        return df
