import os
from threading import Lock
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from Dataflux.helper import Meta, Metadata, get_machine_location
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency
from Dataflux.external.weather import WeatherData, WeatherStations, Geo
from Dataflux.modules.postprocess.normalize import NormalizeFit

mutex = Lock()


class WeatherDataset(FileWorkerCallable):
    """
    Represents a weather dataset for file worker operations. This class provides
    methods and attributes for working with weather data in the context of
    file worker operations.
    """

    DF = pd.DataFrame()

    def __init__(
        self, normalize: bool = False, merge: str = "inner", interpolate: bool = False, enable: bool = True
    ) -> None:
        self._normalize = normalize
        self._merge = merge
        self._interpolate = interpolate
        self._enable = enable

        # FIXME: fallback when machine UUID is not found
        # get_machine_location(machine_uuid)
        # define the location of the compressor
        duelmen = Geo(51.84964389310662, 7.323914205159883)

        # find closest weather station to the compressor
        stations = WeatherStations()
        self._station_id = stations.closest_station(duelmen)

        # download the temperature data of the closest station
        self._weather = WeatherData()

        # keep track of files removed
        self._removed_files = []
        self._lengths_files = {}

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {NormalizeFit.__name__: CallableDependency.NOT_BEFORE}

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        day = file_path.split("_", maxsplit=1)[-1][:-3]
        try:
            df = pd.read_hdf(file_path, key="data")
        except (KeyError, ValueError):
            os.remove(file_path)
            self._removed_files.append(day)
            return

        if self._enable is not True or self._enable != "True":
            self._lengths_files[day] = df.shape
            return

        with mutex:
            metadata = Metadata()
            metadata.load(os.path.dirname(file_path))

            files = [
                _["day"].replace("-", "").split("_")[0] for _ in metadata[Meta.FILES]
            ]
            kwargs = {
                "station_id": self._station_id,
                "start": min(files),
                "end": max(files),
            }

            # Store the WeatherDataset for all threads
            if WeatherDataset.DF.empty:
                WeatherDataset.DF = self._weather.get_data_by_id(**kwargs)

        # get the weather data fro the time span of the loaded DataFrame
        df = df.astype(np.float32)

        min_ts = df["ts"].min()
        max_ts = df["ts"].max()
        temp_df = WeatherDataset.DF[WeatherDataset.DF["ts"] >= min_ts].copy()
        temp_df = temp_df[temp_df["ts"] <= max_ts]

        # inner merge sensor data reduces frequency from every
        # second to every ten minutes, as temperature data is only
        # available every ten minutes
        df = df.merge(temp_df, how=self._merge, on=["ts"])
        if self._interpolate:
            df.interpolate(method="linear", inplace=True)
            df.bfill(inplace=True)
            df.ffill(inplace=True)

        df.to_hdf(file_path, key="data", format="table", mode="w")
        if df.empty or temp_df.empty:
            self._removed_files.append(day)
            os.remove(file_path)
        else:
            self._lengths_files[day] = df.shape

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        metadata.drop(Meta.FILES)
        for day, shape in self._lengths_files.items():
            metadata.put_list({"day": day, "recordings": shape[0], "features": shape[1]}, Meta.FILES)
