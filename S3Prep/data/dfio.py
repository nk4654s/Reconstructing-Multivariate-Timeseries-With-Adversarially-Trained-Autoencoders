"""This module handles pandas dataframe input/outut for s3 streams"""

import os
from typing import List, TextIO, Tuple
import pandas as pd
import numpy as np
from tqdm import trange

from S3Prep.data import S3
from S3Prep.data.utils import logger
from S3Prep.modules import BasePreProcessor


class DFIO:
    """This class handles pandas dataframe input/outut for s3 streams."""

    def __init__(self, s3: S3) -> None:
        """Sets the s3 object to this class for iteration over streams.

        Args:
            s3 (S3): The s3 class.
        """
        self._s3 = s3
        self._df = None
        self._last_header = None
        self._store = None

    def _check_path(self, file_path: str, day_index: int = 0) -> str:
        """Check the file path to ensure it is valid

        Args:
            file_path (str): The file path to check
        """
        # set default file path to machine uuid
        if not file_path:
            day = self._s3.day(day_index).strftime("%Y-%m-%d")
            file_path = f"{self._s3.machine_uuid}_{day}.h5"

        # add increasing number to file path
        num = 1
        temp_path = file_path
        while os.path.exists(temp_path):
            parts = file_path.split(".")
            temp_path = f"{parts[0]}_{num}.{parts[-1]}"
            num += 1

        return temp_path

    def _check_header(self, stream: TextIO, day_index: int = 0) -> Tuple[int, dict]:
        header_str = None
        if self._last_header is not None:
            # check if the dataframe's heaeder matches the stream's header
            header_str = stream.readline()
            incoming = header_str.rstrip("\n").split("\t")[1:]
            if incoming == self._last_header:
                return len(header_str), self._last_header

            # inconsistent headers, save the progress so far and continue on a
            # fresh dataframe instance
            logger.info("Perform save and reset, due to inconsistent headers")
            self.reset(day_index=day_index)
        else:
            # get the header from stream and create a new dataframe
            header_str = stream.readline()
            incoming = header_str.rstrip("\n").split("\t")[1:]

        self._last_header = incoming
        return len(header_str), incoming

    def _safe_convert_float(self, value: str) -> float:
        if value == "":
            return np.nan
        return float(value)

    def _create_storage(self, file_path: str = None, day_index: int = 0) -> None:
        self._store = pd.HDFStore(self._check_path(file_path, day_index), "w")

    def save(self) -> None:
        """Saves the dataframe in an HDF5 file and frees memory.
        """
        assert self._df is not None, "No dataframe to save"
        self._store.put("data", self._df, format="table")
        self._df = None

    def reset(self, file_path: str = None, day_index: int = 0) -> None:
        """Frees memory and creates a new HDF5 file.

        Args:
            file_path (str): Path of the HDF5 file to be saved. If not provided, the
            machines uuid will be used as file name. Defaults to None.
            day_index (int): Index of the day to be saved. Defaults to 0.
        """
        # close and create a new HDF5 file
        self._store.close()
        self._create_storage(file_path=file_path, day_index=day_index)

        self._last_header = None

    def download(self, file_path: str = None, modules: List[BasePreProcessor] = None) -> None:
        """Downloads the dataframe from a S3 stream and downloads the dataframe as file.

        Args:
            file_path (str): Path of the HDF5 file to be saved. If not provided, the
            machines uuid will be used as file name. Defaults to None.
            modules (List[BasePreProcessor]): List of modules to be used. Defaults to None.
        """
        # create the hdf store to save ram usage
        self._create_storage(file_path=file_path)

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
            position, headers = self._check_header(stream, day_index=i)
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
            self._df = pd.DataFrame(data)

            # apply pre-processing
            modules = modules if modules else []
            for module in modules:
                self._df = module(self._df)

            self.save()

        self._store.close()
