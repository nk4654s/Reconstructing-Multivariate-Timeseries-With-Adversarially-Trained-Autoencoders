import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict
from uuid import UUID

import pandas as pd


class FileState(Enum):
    """This enum describes the FileMeta's current state."""
    ADDED = auto()
    MODIFIED = auto()
    DELETED = auto()
    SAME = auto()


@dataclass
class FileMeta:
    """
    This class describes a dataset's file object.

    Args:
        path (str): The file name.
        base_path (str): The path to the file (excluding the file name). Defaults to "".
        state (FileState): The file's state. Defaults to `FileState.SAME` which corresponds to no changes.
        day (datetime): The day which will be read from the files appendix automatically.
        uuid (UUID): The UUID which will be read from the file name automatically.
        features (int): The number of columns
    """
    path: str
    base_path: str = ""
    state: FileState = FileState.SAME
    day: datetime = field(init=False)
    uuid: UUID = field(init=False)
    features: int = field(init=False)
    samples: int = field(init=False)

    def __post_init__(self) -> None:
        df = pd.read_feather(os.path.join(self.base_path, self.path))
        self.samples, self.features = df.shape

        file_name = self.path.rsplit(os.path.sep, maxsplit=1)[-1]
        uuid, day_str = file_name.split("_", maxsplit=1)
        day_str = day_str.split("_")[0].split(".")[0]

        self.day = datetime.fromisoformat(day_str)
        self.uuid = UUID(uuid)

    @property
    def full_path(self) -> str:
        """
        Returns:
            str: The full path concatenation of the base path and file name.
        """
        return os.path.join(self.base_path, self.path)

    def as_dict(self) -> Dict:
        return {
            "path": self.path,
            "day": self.day.strptime("YYYY-MM-DD"),
            "features": self.features,
            "samples": self.samples,
        }

    @staticmethod
    def from_dict(d: Dict, base_path: str = "") -> "FileMeta":
        return FileMeta(path=d["path"], base_path=base_path)


if __name__ == "__main__":
    fm = FileMeta(
        "../../../../Comprisk/Comprisk/tests/test-debug-dataset/0b3426b3-94af-4b8c-89c1-da57493c099a_2025-02-03.feather"
    )
    print(fm)
