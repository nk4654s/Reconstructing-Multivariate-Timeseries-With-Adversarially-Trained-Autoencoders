"""This module handles the s3 stream."""

import os
import gzip
from datetime import date
from typing import Tuple, Union, TextIO
from io import BytesIO
from uuid import UUID

import boto3

from .tsvio import logger
from .utils import get_day_list


class S3:
    def __init__(
        self,
        machine_uuid: str | UUID,
        interval: Tuple[date, date],
    ) -> None:

        self._bucket = os.getenv("S3_BUCKET", "xplore-measurements")
        self._client = boto3.client("s3")

        self._machine_uuid = machine_uuid
        self._start, self._end = interval
        self._days = get_day_list((self._start, self._end))

        assert self._start < self._end, "Start date must be before end date"

    @property
    def machine_uuid(self):
        return self._machine_uuid

    def set_machine(self, machine_uuid: str | UUID) -> None:
        self._machine_uuid = machine_uuid

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
