import os
from uuid import UUID
import boto3
import boto3.session

from Dataflux.data.types import DataType


class S3Bucket:
    def __init__(self, machine_uuid: str | UUID) -> None:
        self._uuid: UUID | None = None
        self.set_machine(machine_uuid)

        self._bucket = os.getenv("S3_BUCKET", DataType.MEASUREMENTS.value)
        self._client = boto3.client("s3")

    def set_data_type(self, data_type: DataType) -> None:
        """
        Usually, the bucket type will be retrieved from the environment variables.
        Here, a different bucket can be selected.

        Args:
            data_type (DataType): The data type of the bucket to download from.
        """
        self._bucket = data_type.value

    @property
    def machine_uuid(self) -> UUID:
        """The machines uuid."""
        assert self._uuid
        return self._uuid

    def set_machine(self, uuid: str | UUID) -> None:
        """Sets the machines uuid.

        Args:
            machine_uuid (str | UUID): The machines uuid
        """
        self._uuid = uuid if isinstance(uuid, UUID) else UUID(uuid)
