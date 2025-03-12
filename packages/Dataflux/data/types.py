from enum import Enum


class DataType(Enum):
    """This enum contains the bucket types of Amazon's S3 storage solution."""

    MEASUREMENTS = "xplore-measurements"
    REPORTS = "xplore-reports"
