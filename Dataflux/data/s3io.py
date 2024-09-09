from enum import Enum
import os
from uuid import UUID
import boto3


class DataType(Enum):
    """This enum contains the bucket types."""

    MEASUREMENTS = "xplore-measurements"
    REPORTS = "xplore-reports"


class S3Bucket:
    def __init__(self, machine_uuid: str | UUID) -> None:
        self._uuid = machine_uuid

        self._bucket = None
        self._bucket_type = None
        self._client = boto3.client("s3")

    @property
    def data_type(self) -> DataType:
        """
        The data type of the bucket to download from.
        """
        return self._bucket_type

    def set_data_type(self, data_type: DataType) -> None:
        """
        Sets the data type of the bucket to download from.

        Args:
            data_type (DataType): The data type of the bucket to download from.
        """
        self._bucket = os.getenv("S3_BUCKET", data_type.value)
        self._bucket_type = data_type

    @property
    def machine_uuid(self) -> str:
        """The machines uuid."""
        return self._uuid

    def set_machine(self, machine_uuid: str | UUID) -> None:
        """Sets the machines uuid.

        Args:
            machine_uuid (str | UUID): The machines uuid
        """
        self._uuid = machine_uuid
