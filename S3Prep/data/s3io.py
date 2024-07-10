"""This module handles the s3 stream."""

import os
import gzip
import logging
from datetime import date
from typing import Tuple, Union, TextIO
from io import BytesIO
from uuid import UUID

import boto3

from S3Prep.helper import get_day_list, get_logger

# create logger object
logger = get_logger(__name__)


class S3:
    """This class opens S3 streams to download machine sensor data.
     """
    def __init__(
        self,
        machine_uuid: str | UUID,
        interval: Tuple[date, date],
    ) -> None:
        """Initialize the class and defines the s3 bucket.

        Args:
            machine_uuid (str): The machine uuid, from which sensor data can be
            downloaded.
            interval (tuple): A tuple of two dates. The first one is the start
            date and the second one is the end date.
        """
        self._bucket = os.getenv("S3_BUCKET", "xplore-measurements")
        self._client = boto3.client("s3")

        self._machine_uuid = machine_uuid
        self._start, self._end = interval
        self._days = get_day_list((self._start, self._end))

        assert self._start < self._end, "Start date must be before end date"

    @property
    def machine_uuid(self) -> str:
        """The machines uuid.
        """
        return self._machine_uuid

    def set_machine(self, machine_uuid: str | UUID) -> None:
        """Sets the machines uuid.

        Args:
            machine_uuid (str | UUID): The machines uuid
        """
        self._machine_uuid = machine_uuid

    def day(self, i: int) -> date:
        """Returns the day according to the index.

        Args:
            i (int): The index of the day
        """
        return self._days[i]

    def _open_stream(self, s3_path: str) -> Union[TextIO, None]:
        logger.debug("Retrieving '%s:%s' from S3", self._bucket, s3_path)
        try:
            # open and unconmpress the s3 stream given the bucket and path
            data = self._client.get_object(Bucket=self._bucket, Key=f"{s3_path}")[
                "Body"
            ].read()
            stream: TextIO = gzip.open(BytesIO(data), mode="rt")
            stream.seek(0)
            return stream

        except Exception as e:
            # catch an exception if the bucket or path is invalid, return None
            logger.debug(
                "Could not receive tsv stream because of exception %s: %s", type(e), e
            )
            return None

    def __len__(self) -> int:
        return len(self._days)

    def __getitem__(self, index: int) -> TextIO:
        # create the s3 path from the current date
        date_str = self._days[index].strftime("%Y/%m/%d")
        path = f"{self._machine_uuid}/days/{date_str}/{self._machine_uuid}.tsv"

        # return the s3 stream
        return self._open_stream(s3_path=path)
