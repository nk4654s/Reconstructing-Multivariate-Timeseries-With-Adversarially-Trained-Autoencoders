"""This module handles pandas dataframe input/outut for s3 streams"""

import os
from typing import List, TextIO, Tuple
import pandas as pd
import numpy as np
from tqdm import trange

from S3Prep.data import S3
from S3Prep.data.utils import logger, get_sensor_uuid_to_name
from S3Prep.modules import BasePreProcessor, EntryPoint


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

    def save(self, df: pd.DataFrame, file_path: str = None, day_index: int = 0) -> None:
        """Saves the dataframe in an HDF5 file and frees memory.

        Args:
            df (pd.DataFrame): dataframe to be saved
            file_path (str): path of the HDF5 file where to save
            day_index (int): index of the day to be saved
        """
        assert df is not None, "No dataframe to save"
        if not df.empty:
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
            position, headers, drop_headers = self._check_header(stream)
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
            df = pd.DataFrame(data)

            # run the modules which require non-processed data
            # as result, df might be a list
            meta = {"date": self._s3.day(i).strftime("%Y-%m-%d"), "machine": self._s3.machine_uuid}
            for module in modules:
                df = module(df, EntryPoint.ON_DOWNLOAD_FINISHED, metadata=meta)

            dfs = [df] if isinstance(df, pd.DataFrame) else df
            for df in dfs:
                # only select valid columns if inner merge
                if drop_headers:
                    df = df[headers]

                # rename uuid to sensor name
                names = {id: self._sensor_names.get(id, id) for id in list(df.columns.values)}
                df.rename(columns = names, inplace = True)
                logger.debug("Headers: %s", str(list(df.columns.values)))

                # apply pre-processing, might result in a list of dataframes
                for module in modules:
                    df = module(df, EntryPoint.ON_PROCESSING_STARTED, metadata=meta)
                dfs = [df] if isinstance(df, pd.DataFrame) else df

                for df in dfs:
                    self.save(df, file_path, i)

        # execute modules which require to be executed after processing
        # hence, there is no dataframe to process
        for module in modules:
            module(None, EntryPoint.ON_PROCESSING_FINISHED)
