"""This module handles pandas dataframe input/outut for s3 streams"""

import os
from typing import TextIO
import pandas as pd
import numpy as np
from tqdm import trange

from .s3io import S3
from .utils import logger


class DFIO:
    """This class handles pandas dataframe input/outut for s3 streams."""

    def __init__(self, s3: S3) -> None:
        """Sets the s3 object to this class for iteration over streams.

        Args:
            s3 (S3): The s3 class.
        """
        self._s3 = s3
        self._df = None

    def _check_path(self, file_path: str) -> str:
        """Check the file path to ensure it is valid

        Args:
            file_path (str): The file path to check
        """
        # set default file path to machine uuid
        if not file_path:
            file_path = self._s3.machine_uuid + ".h5"

        # add increasing number to file path
        num = 1
        temp_path = file_path
        while os.path.exists(temp_path):
            parts = file_path.split(".")
            temp_path = f"{parts[0]}_{num}.{parts[-1]}"
            num += 1

        return temp_path

    def _check_header(self, stream: TextIO) -> int:
        header_str = None
        if self._df is not None:
            # check if the dataframe's heaeder matches the stream's header
            header_str = stream.readline()
            incoming = header_str.rstrip("\n").split("\t")[1:]
            existing = self._df.columns.values.tolist()
            print(len(incoming), len(existing))
            if incoming == existing:
                return len(header_str)

            # inconsistent headers, save the progress so far and continue on a
            # fresh dataframe instance
            logger.info("Perform save and reset, due to inconsistent headers")
            self.save_reset()
        else:
            # get the header from stream and create a new dataframe
            header_str = stream.readline()
            incoming = header_str.rstrip("\n").split("\t")[1:]

        data = {s: [] for s in incoming}
        self._df = pd.DataFrame(data=data)
        return len(header_str)

    def _safe_convert_float(self, value: str) -> float:
        if value == "":
            return np.nan
        return float(value)

    def save(self, file_path: str = None) -> None:
        """Saves the dataframe in an HDF5 file.

        Args:
            file_path (str): Path of the HDF5 file to be saved. If not provided, the
            machines uuid will be used as file name. Defaults to None.
        """
        assert self._df is not None, "No dataframe to save"
        self._df.to_hdf(self._check_path(file_path), key="df", mode="w")

    def save_reset(self, file_path: str = None) -> None:
        """Saves the dataframe in an HDF5 file. Then the internal dataframe is reseted.

        Args:
            file_path (str): Path of the HDF5 file to be saved. If not provided, the
            machines uuid will be used as file name. Defaults to None.
        """
        self.save(file_path)
        self._df = None

    def download(self, file_path: str = None) -> None:
        """Downloads the dataframe from a S3 stream and downloads the dataframe as file.

        Args:
            file_path (str): Path of the HDF5 file to be saved. If not provided, the
            machines uuid will be used as file name. Defaults to None.
        """
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
            position = self._check_header(stream)
            stream.seek(position)

            for line in stream.readlines():
                # extract and format the row
                row = [
                    self._safe_convert_float(s)
                    for s in line.strip("\n").split("\t")[1:]
                ]

                # then append to the dataframe
                self._df.loc[self._df.index.max() + 1] = row

        self.save(file_path)
