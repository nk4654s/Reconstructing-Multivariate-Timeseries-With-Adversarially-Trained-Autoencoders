import os
from threading import Lock
from typing import Any, List
import pandas as pd
from S3Prep.helper import Meta, Metadata, get_machine_location
from S3Prep.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency
from S3Prep.external.weather import WeatherData, WeatherStations, Geo
from S3Prep.modules.postprocess.normalize import NormalizeFit

mutex = Lock()


class WeatherDataset(FileWorkerCallable):
    """
    Represents a weather dataset for file worker operations. This class provides
    methods and attributes for working with weather data in the context of
    file worker operations.
    """
    DF = None

    def __init__(
        self, normalize: bool = False, merge: str = "inner", interpolate: str = None
    ) -> None:
        self._normalize = normalize
        self._merge = merge
        self._interpolate = interpolate

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

    def get_dependencies(self) -> List[str]:
        return {NormalizeFit.__name__: CallableDependency.NOT_BEFORE}

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        print("weather:", file_path)
        with mutex:
            metadata = Metadata()
            metadata.load(os.path.dirname(file_path))

            files = [
                _["day"].replace("-", "").split("_")[0] for _ in metadata[Meta.FILES]
            ]
            kwargs = {"station_id": self._station_id, "start": min(files), "end": max(files)}

            # Store the WeatherDataset for all threads
            if WeatherDataset.DF is None:
                WeatherDataset.DF = self._weather.get_data_by_id(**kwargs)

        # get the weather data fro the time span of the loaded DataFrame
        df = pd.read_hdf(file_path, key="data")
        min_ts = df["ts"].min()
        max_ts = df["ts"].max()
        temp_df = WeatherDataset.DF[WeatherDataset.DF.index >= min_ts].copy()
        temp_df = temp_df[temp_df.index <= max_ts]

        # inner merge sensor data reduces frequency from every
        # second to every ten minutes, as temperature data is only
        # available every ten minutes
        df = df.merge(temp_df, how=self._merge, on=["ts"])
        if self._interpolate:
            df.interpolate(method=self._interpolate, inplace=True)

        df.to_hdf(file_path, key="data", format="table", mode="w")
        day = file_path.split("_", maxsplit=1)[-1][:-3]
        if df.empty:
            print("\tRemoving empty file", file_path)
            self._removed_files.append(day)
            os.remove(file_path)
        else:
            self._lengths_files[day] = len(df)

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
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
