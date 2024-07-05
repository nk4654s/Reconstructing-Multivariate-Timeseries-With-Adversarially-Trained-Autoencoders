"""This module handles pandas dataframe input/outut for s3 streams"""

import os
from typing import List, TextIO, Tuple
import pandas as pd
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from S3Prep.data import S3
from S3Prep.data.utils import logger, get_sensor_uuid_to_name
from S3Prep.modules import BasePreProcessor


class DFIO:
    """This class handles pandas dataframe input/outut for s3 streams."""

    def __init__(self, s3: S3, limit_headers: bool = False) -> None:
        """Sets the s3 object to this class for iteration over streams.

        Args:
            s3 (S3): The s3 class.
            limit_headers (bool): Limits headers to those of the first
            incoming dataframe. Defaults to False.
        """
        self._s3 = s3
        self._sensor_names = get_sensor_uuid_to_name(self._s3.machine_uuid)

        self._limit_headers = limit_headers
        self._last_header = None

        self._intermediate_modules = []
        self._final_modules = []
        self._features_per_day = {}

    def _check_path(self, file_path: str, day_index: int = 0) -> str:
        """Check the file path to ensure it is valid

        Args:
            file_path (str): The file path to check
        """
        # set default file path to machine uuid
        day = self._s3.day(day_index).strftime("%Y-%m-%d")
        if not file_path:
            file_path = f"{self._s3.machine_uuid}_{day}.h5"
        else:
            file_path = file_path.replace(".", f"_{day}.")

        # check if the folder for the dataframes exist
        if "/" in file_path:
            folder_path = "/".join(file_path.split("/")[:-1])
            if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
                os.makedirs(folder_path)

        # add increasing number to file path
        num = 1
        temp_path = file_path
        while os.path.exists(temp_path):
            parts = file_path.split(".")
            temp_path = f"{parts[0]}_{num}.{parts[-1]}"
            num += 1

        return temp_path

    def _check_header(self, stream: TextIO) -> Tuple[int, dict, bool]:
        header_str = None
        if self._last_header is not None:
            # check if the dataframe's heaeder matches the stream's header
            header_str = stream.readline()
            incoming = header_str.rstrip("\n").split("\t")[1:]
            if incoming == self._last_header:
                return len(header_str), self._last_header, False

            if self._limit_headers:
                return len(header_str), self._last_header, True
        else:
            # get the header from stream and create a new dataframe
            header_str = stream.readline()
            incoming = header_str.rstrip("\n").split("\t")[1:]

        self._last_header = incoming
        return len(header_str), incoming, False

    def _safe_convert_float(self, value: str) -> float:
        if value == "":
            return np.nan
        return float(value)

    def _create_storage(self, file_path: str = None, day_index: int = 0) -> pd.HDFStore:
        return pd.HDFStore(self._check_path(file_path, day_index), "w")

    def _apply_modules(self, modules: List[BasePreProcessor],
                       df: pd.DataFrame) -> List[pd.DataFrame]:
        processed = [df]
        for module in modules:
            pointer = len(processed)
            for i in range(pointer):
                processed += module(processed[i])

            # remove old dataframes
            processed = processed[pointer:]
        return processed

    def export_features_per_day(self, save_plot: str = None) -> dict[str, int]:
        """Returns a dictionary with the number of features per day.

        Args:
            save_plot (str): path where to save the plot with the number of features per days.
            Defaults to None.
        """
        dates = list(self._features_per_day.keys())
        num_features = list(self._features_per_day.values())

        plt.plot(dates, num_features)
        plt.xlabel("day")
        plt.ylabel("num_sensors")

        if save_plot:
            plt.savefig(self._check_path(save_plot))

        return self._features_per_day

    def save(self, df: pd.DataFrame, file_path: str = None, day_index: int = 0) -> None:
        """Saves the dataframe in an HDF5 file and frees memory.

        Args:
            df (pd.DataFrame): dataframe to be saved
            file_path (str): path of the HDF5 file where to save
            day_index (int): index of the day to be saved
        """
        assert df is not None, "No dataframe to save"
        if not df.empty:
            processed = self._apply_modules(self._final_modules, df)

            for df in processed:
                if df.empty:
                    continue
                store = self._create_storage(file_path, day_index)
                store.put("data", df, format="table", append=False)
                store.close()

    def download(self, file_path: str = None, modules: List[BasePreProcessor] = None) -> None:
        """Downloads the dataframe from a S3 stream and downloads the dataframe as file.

        Args:
            file_path (str): Path of the HDF5 file to be saved. If not provided, the
            machines uuid will be used as file name. Defaults to None.
            modules (List[BasePreProcessor]): List of modules to be used. Defaults to None.
        """
        # organize modules
        modules = modules if modules else []
        self._intermediate_modules = [m for m in modules if m.intermediate_run]
        self._final_modules = [m for m in modules if not m.intermediate_run]

        # iterate over all streams and append their content to the dataframe
        num_streams = len(self._s3)
        pbar = trange(
            num_streams,
            desc=f"Fetching TSV daily data for machine {self._s3.machine_uuid}",
        )
        for i in pbar:
            # get the stream, skip if the stream could not be opened
            stream = self._s3[i]
            if stream is None:
                continue

            # check if the dataframe is empty and set it up accordingly
            # also check if the header is consistent with the stream's one
            # if not, save the progress so far and continue on a fresh dataframe instance
            logger.debug("Streaming day %s", self._s3.day(i))
            position, headers, do_merge = self._check_header(stream)
            stream.seek(position)

            # read all values from the incoming sensor data
            values = []
            for line in stream.readlines():
                # extract and format the row
                row = [
                    self._safe_convert_float(s)
                    for s in line.strip("\n").split("\t")[1:]
                ]
                values.append(row)
            values = np.array(values)

            # append the data to the dataframe
            data = {key: val for (key, val) in zip(headers, values.T)}
            raw_df = pd.DataFrame(data)
            if len(raw_df.columns.values) == 0:
                self._features_per_day[self._s3.day(i)] = len(raw_df.columns.values)

            # only select valid columns if inner merge
            if do_merge:
                raw_df = raw_df[headers]

            # rename uuid to sensor name
            names = {id: self._sensor_names[id] for id in list(raw_df.columns.values)}
            raw_df.rename(columns = names, inplace = True)
            logger.debug("Headers: %s", str(list(raw_df.columns.values)))

            # apply pre-processing
            processed = self._apply_modules(self._intermediate_modules, raw_df)
            for df in processed:
                self.save(df, file_path, i)
