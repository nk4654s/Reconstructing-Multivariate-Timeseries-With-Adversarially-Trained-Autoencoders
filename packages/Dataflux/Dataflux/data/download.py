"""This module contains classes to download the raw TSV-files from the S3 bucket and handles
the parsing."""

import gzip
import os
from datetime import date, datetime
from io import BytesIO
from multiprocessing import Pool
from typing import List, Tuple
from uuid import UUID
from tqdm import tqdm
import numpy as np
import pandas as pd

from Dataflux.data.header import Header
from Dataflux.data.s3io import S3Bucket
from Dataflux.helper import Meta, Metadata, get_file, get_day_list, line_str_to_array
from Dataflux import logger, DEBUG_MODE


def homogeneous_decompress(df: pd.DataFrame, index_key: str = "ts") -> pd.DataFrame:
    # decompress the dataframe using forward-fill interpolation
    df.ffill(inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(how="any", inplace=True)

    if df.empty:
        return df

    # get time start and end
    range_start = df[index_key].iloc[0]
    range_end = df[index_key].iloc[-1] + 1

    # create a new index based on the start and end range
    new_index = pd.Index(np.arange(range_start, range_end, 1), name=index_key)

    # reindex the dataframe, inserted rows will be filled with NaNs
    df = df.set_index(index_key).reindex(new_index).reset_index(index_key)

    # interpolate added rows
    df.interpolate(method="linear", inplace=True)
    return df


class EmptyStreamError(Exception):
    """Raised when the stream is empty."""

    def __init__(self, day: str) -> None:
        super().__init__(f"Stream of '{day}' is empty")


class DataDownloader(S3Bucket):
    """
    This class downloads the raw TSV files from the S3 bucket and converts them to a pandas dataframe.

    Args:
        machine_uuid (str): The unique ID of the machine to download data from.
    """

    def __init__(
        self, machine_uuid: str | UUID, file_postfix: str = "", appendix: str = ""
    ) -> None:
        super().__init__(machine_uuid)

        self._postfix = f"-{file_postfix}" if file_postfix else ""
        self._appendix = f"-{appendix}" if appendix else ""
        self._header = None
        self._pool = Pool(12)

        logger.debug(
            "Download TSV-files from %s with postfix='%s' and '%s'",
            self._uuid,
            file_postfix,
            appendix,
        )

    @property
    def appendix(self) -> str:
        return self._appendix

    def get_stream(self, day: str | date) -> List[str]:
        """
        Opens and returns the stream from the S3 bucket given the day.

        Args:
            day (str | date): The day to open the stream from. Should be either in the format
                `YYYY-MM-DD` or `YYYY/MM/DD`.

        Raises:
            S3.Client.exceptions.NoSuchKey
            S3.Client.exceptions.InvalidObjectState

        Returns:
            TextIO: The stream from the S3 bucket given the day.
        """
        # build the file-path
        if isinstance(day, str):
            day = day if "/" in day else day.replace("-", "/")
        else:
            day = day.strftime("%Y/%m/%d")

        path = (
            f"{self._uuid}/days/{day}/{self._uuid}{self._postfix}.tsv{self._appendix}"
        )

        # open and uncompressed the s3 stream given the bucket and path
        data = self._client.get_object(Bucket=self._bucket, Key=f"{path}")
        stream = gzip.open(BytesIO(data["Body"].read()), mode="rt", encoding="utf-8")

        lines = stream.readlines()
        return lines

    def parse_header(self, line: str) -> Header:
        """This method parses a line into a header.
        Args:
            line (str): The line to parse into a header.

        Returns:
            Header: The parsed header.
        """
        # build the header instance
        incoming = Header()
        incoming.values = line.rstrip("\n").split("\t")
        return incoming

    def parse_stream(self, lines: List[str]) -> pd.DataFrame:
        """
        This method parses the stream into a pandas DataFrame. If the stream is empty,
        it will return `None`. Otherwise, each line is parsed into an array of float32
        values.

        Args:
            lines (List[str]): The list of lines read from AWS using the `get_stream()` method.

        Returns:
            pd.DataFrame: The parsed pandas DataFrame.
        """
        if len(lines) <= 1:
            logger.warning(
                "Got an empty stream (header? %s).", "yes" if len(lines) == 1 else "no"
            )
            return pd.DataFrame()

        header = self.parse_header(lines.pop(0))

        # read all values from the incoming sensor data
        values = self._pool.map(line_str_to_array, lines)
        values = np.array(values, dtype=np.float64)
        data = {key: val for (key, val) in zip(header.values, values.T)}
        df = pd.DataFrame(data)

        # adjust types to save some storage space
        df["ts"] = df["ts"].astype(np.int64)
        not_ts = list(set(df.columns.values) - {"ts"})
        df[not_ts] = df[not_ts].astype(np.float32)

        return df

    def get_dataframe(self, day: str) -> pd.DataFrame:
        """
        This method combines the `self.get_stream()` and `self.parse_stream()` methods to
        parse the stream into a pandas DataFrame. If the stream is empty, it will raise an
        `EmptyStreamError`.

        Args:
            day (str): The day to stream from.

        Raises:
            EmptyStreamError (Exception): If the stream is empty.

        Returns:
            pd.DataFrame: The parsed pandas DataFrame.
        """
        df = None
        try:
            lines = self.get_stream(day)
            df = self.parse_stream(lines)
            if df.empty:
                logger.warning("Downloaded empty DataFrame.")

        except BaseException as e:
            logger.error("Unable to download dataset: %s", day)
            logger.error("Caught error %s", e)

        df = df if df is not None else pd.DataFrame()
        time = (
            datetime.fromtimestamp(df["ts"].values[0])
            if "ts" in df.columns.values
            else 0
        )
        return df


class ContinuousDataDownloader(DataDownloader):
    """
    This class is used to download multiple files from a S3 bucket given the machine uuid and dates.

    Args:
        machine_uuid (str | UUID): The machine uuid to download data from.
        interval (List[Tuple[date, date]] | Tuple[date, date]): The interval to download data from.
    """

    def __init__(
        self,
        machine_uuid: str | UUID,
        interval: List[Tuple[datetime, datetime]] | Tuple[datetime, datetime],
    ) -> None:
        super().__init__(machine_uuid)

        # the interval can either be a single interval provided as tuple of two days
        # or multiple intervals provided as a list of tuples.
        if isinstance(interval, list):
            self._days = []
            for inter in interval:
                if inter[0] > inter[1]:
                    logger.error(
                        "Start date must be before end date (%s > %s).",
                        inter[0],
                        inter[1],
                    )
                    exit(0)
                self._days += get_day_list(inter[0], inter[1])
        else:
            if interval[0] <= interval[1]:
                logger.error(
                    "Start date must be before end date (%s > %s).",
                    interval[0],
                    interval[1],
                )
                exit(0)
            self._days = get_day_list(interval[0], interval[1])

    def day(self, i: int) -> datetime:
        """Returns the day according to the index.

        Args:
            i (int): The index of the day
        """
        return self._days[i]

    def run(self, dir_name: str = "dataset") -> bool:
        """Downloads the dataframe from a S3 stream and downloads the dataframe as file.

        Args:
            dir_name (str): Path of the HDF5 file to be saved. If not provided, the current
            folder will be used. Defaults to "dataset".

        Returns:
            bool: True if the downloaded dataset is not empty, else false.
        """
        filename = str(self.machine_uuid) + ".feather"

        # organize modules and build metadata as information for modules
        meta = Metadata()
        meta.put(dir_name, Meta.PATH)
        meta.put(str(self.machine_uuid), Meta.MACHINE_UUID)
        files = []

        # iterate over all streams and append their content to the dataframe
        length = (
            len(self) if len(self) != 2 or self.day(1).day - self.day(0).day != 0 else 1
        )
        pbar = tqdm(total=length, desc="Downloading Dataset", disable=not DEBUG_MODE)
        for i in range(length):
            day = self.day(i).strftime("%Y-%m-%d")
            file_path = get_file(os.path.join(dir_name, filename), postfix=day)
            pbar.set_description(f"Downloading {day}")

            # get the DataFrame of the  stream, skip if the stream could not be opened
            try:
                df: pd.DataFrame = self[i]

                # save the dataframe if it is not None or empty
                if df is not None and not df.empty:
                    df = homogeneous_decompress(df)
                    if df.empty:
                        pbar.update(1)
                        continue

                    meta.put(self.day(i).strftime("%Y-%m-%d"), Meta.DAY)
                    # remove timestamps if the day started at a given time
                    if i == 0:
                        df = df[df["ts"] >= self.day(i).timestamp()]

                        # remove timestamps if the dataset only consists of one day.
                        if len(self) == 2 and self.day(1).day - self.day(0).day == 0:
                            df = df[df["ts"] <= self.day(i + 1).timestamp()]

                    if i == len(self) - 1:
                        df = df[df["ts"] <= self.day(i).timestamp()]

                    df.to_feather(file_path)
                    files.append(
                        {
                            Meta.DAY.name: day,
                            "recordings": len(df),
                            "features": df.shape[1],
                        }
                    )
                pbar.update(1)
            except EmptyStreamError:
                logger.debug("Stream for day %s is empty, skipping file download.", day)
                pbar.update(1)
                continue

        logger.info(
            "Finished downloading %s files. Continuing with processing files.",
            len(files),
        )
        logger.debug("Saving metadata info to '%s'.", dir_name)

        meta.put(sum([f["recordings"] for f in files]), Meta.TOTAL_RECORDINGS)
        meta.put(files, Meta.FILES)
        meta.save(dir_name)

        return len(files) != 0

    def __len__(self) -> int:
        return len(self._days)

    def __getitem__(self, index: int) -> pd.DataFrame:
        # create the s3 path from the current date
        day = self._days[index]
        return self.get_dataframe(day.strftime("%Y-%m-%d"))
