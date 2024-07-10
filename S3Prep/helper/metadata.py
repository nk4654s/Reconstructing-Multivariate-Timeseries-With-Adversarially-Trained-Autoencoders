"""This module handles the metadata class and content."""
from enum import Enum
import os
from typing import Any
import json

from S3Prep.helper.utils import get_file
from S3Prep.helper.logger import get_logger

# create logger object
logger = get_logger(__name__)


class Meta(Enum):
    """This enum holds the metadata keys.
    """
    DAY = "day"
    MACHINE_UUID = "machine_uuid"
    PATH = "path"
    MODULES = "modules"
    FILES = "files"
    TRAIN_FILES = "train_files"
    VALIDATION_FILES = "validation_files"
    TEST_FILES = "test_files"
    TOTAL_RECORDINGS = "total_recordings"
    CORRELATION_OF_INTEREST = "correlation_of_interest"
    NORMALIZER = "normalizer"
    NORM_METHOD = "normalizer_method"
    NORM_VAL_A = "normalizer_val_a"
    NORM_VAL_B = "normalizer_val_b"

class Metadata:
    """This class holds the metadata of a dataset.
    """
    def __init__(self):
        self._values = {}

    def put(self, value: Any, meta_key: Meta) -> None:
        """Puts the value into a dict under the key meta_key.

        Args:
            value (Any): Value that should be put into dict under key meta_key.
            meta_key (Meta): Key to put the value in.
        """
        self._values[meta_key.name] = value

    def drop(self, meta_key: Meta) -> None:
        """Drops the key meta_key from dict _values.

        Args:
            meta_key (Meta): Key to drop from dict _values.
        """
        del self._values[meta_key.name]

    def __getitem__(self, meta_key: Meta) -> Any:
        """Returns the value of a key in dict _values.

        Args:
            meta_key (Meta): Key that should be retrieved from dict _values.
        
        Returns:
            Any: Value that is stored under key meta_key.
        """
        return self._values[meta_key.name]

    def save(self, base_dir: str) -> None:
        """Saves the metadata to a json-file in `/metadata/info.json`

        Args:
            base_dir (str): Path of the root_folder to save metadata in.
        """
        file_path = os.path.join(base_dir, "metadata", "info.json")
        with open(get_file(file_path, overwrite=True), "w", encoding="utf-8") as f:
            json.dump(self._values, f, indent=2, ensure_ascii=False)

    def load(self, base_dir: str) -> None:
        """Loads the metadata from a json-file. Needs to be in a subfolder 
        `/metadata/info.json`.
        
        Args:
            base_dir (str): Path of the root_folder to load metadata from.
        """
        file_path = os.path.join(base_dir, "metadata", "info.json")
        with open(file_path, "r", encoding="utf-8") as f:
            self._values = json.load(f)
