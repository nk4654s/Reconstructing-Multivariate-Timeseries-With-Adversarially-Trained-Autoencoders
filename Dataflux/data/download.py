"""This module contains classes to download the raw TSV-files from the S3 bucket and handles
the parsing."""
from datetime import date
import gzip
from io import BytesIO
from multiprocessing import Pool
from typing import List, TextIO, Tuple
from uuid import UUID
import numpy as np
import pandas as pd

from Dataflux.data.header import Header
from Dataflux import logger
from Dataflux.helper.utils import get_day_list, line_str_to_array
from Dataflux.data.s3io import S3Bucket


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
    def __init__(self, machine_uuid: str | UUID, file_postfix: str = None) -> None:
        super().__init__(machine_uuid)

        self._postfix = f"-{file_postfix}" if file_postfix else""
        self._header = None
        self._pool = Pool(12)

    def get_stream(self, day: str | date) -> TextIO:
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

        path = f"{self._uuid}/days/{day}/{self._uuid}{self._postfix}.tsv"
        logger.debug("Retrieving '%s:%s' from S3", self._bucket, path)

        # open and unconmpress the s3 stream given the bucket and path
        data = self._client.get_object(Bucket=self._bucket, Key=f"{path}")[
            "Body"
        ].read()
        stream = gzip.open(BytesIO(data), mode="rt")
        stream.seek(0)

        return stream

    def parse_header(self, line: str) -> Header:
        """This method parses a line into a header and checks for validity. If
        there is a change in number of header.values, this method will define a
        relative complement of the previous header and the new header. This set
        of values will be dropped from the DataFrame.

        Args:
            line (str): The line to parse into a header.

        Returns:
            Header: The parsed header.
        """
        # build the header instance
        incoming = Header()
        incoming.values = line.rstrip("\n").split("\t")

        # if no header is registered so far, just use the latest income
        if self._header is None:
            logger.debug("Set initial header with %s values", incoming.length)
            self._header = incoming
            return self._header

        # if the incoming header is longer than the previous one, drop the
        # values which are unknown or missing in the previous one and
        # return it
        if incoming.values == self._header.values:
            logger.debug("No change in header, returning previous one.")
            return self._header

        # calculate the headers to remove
        assert (
            self._header.length < incoming.length
        ), "Header length mismatch. Incoming header length is smaller than known header length."

        incoming.relative_complement(self._header, fail_safe=True)
        logger.debug("Header length mismatch. Removing %s values.", len(incoming.drop))
        return incoming

    def parse_stream(self, stream: TextIO = "dataset") -> pd.DataFrame:
        """
        This method parses the stream into a pandas DataFrame. If the stream is empty,
        it will return `None`. Otherwise, each line is parsed into an array of float32
        values.

        Args:
            stream (TextIO): The stream to parse into a pandas DataFrame

        Returns:
            pd.DataFrame: The parsed pandas DataFrame.
        """
        # check if the dataframe is empty and set it up accordingly
        # also check if the header is consistent with the stream's one
        # if not, save the progress so far and continue on a fresh dataframe instance
        lines = stream.readlines()
        if len(lines) <= 1:
            return None

        header = self.parse_header(lines.pop(0))

        # read all values from the incoming sensor data
        values = self._pool.map(line_str_to_array, lines)
        values = np.array(values, dtype=np.float64)
        data = {key: val for (key, val) in zip(header.values, values.T)}
        return pd.DataFrame(data)

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
        stream = self.get_stream(day)
        df = self.parse_stream(stream)

        if df is None:
            raise EmptyStreamError(day)

        # adjust types to save some storage space
        df["ts"] = df["ts"].astype(np.int64)
        not_ts = list(set(df.columns.values) - {"ts"})
        df[not_ts] = df[not_ts].astype(np.float32)

        logger.info("Finished streaming day %s", day)
        logger.info("Streamed %s lines in total", len(df))

        return df


class ContinuousDataDownloader(DataDownloader):
    """
    This class is used to download multiple files from a S3 bucket given the machine uuid and dates.

    Args:
        machine_uuid (str | UUID): The machine uuid to download data from.
        interval (List[Tuple[date, date]] | Tuple[date, date]): The interval to download data from.
    """
    def __init__(self, machine_uuid: str | UUID,
                 interval: List[Tuple[date, date]] | Tuple[date, date]) -> None:
        super().__init__(machine_uuid)

        # the interval can either be a single interval provided as tuple of two days
        # or multiple intervals provided as a list of tuples.
        if isinstance(interval, list):
            self._days = []
            for inter in interval:
                assert inter[0] < inter[1], "Start date must be before end date"
                self._days += get_day_list((inter[0], inter[1]))
        else:
            assert interval[0] < interval[1], "Start date must be before end date"
            self._days = get_day_list((interval[0], interval[1]))

    def day(self, i: int) -> date:
        """Returns the day according to the index.

        Args:
            i (int): The index of the day
        """
        return self._days[i]

    def __len__(self) -> int:
        return len(self._days)

    def __getitem__(self, index: int) -> pd.DataFrame:
        # create the s3 path from the current date
        day = self._days[index]
        return self.get_dataframe(day)
