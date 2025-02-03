import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict
from uuid import UUID

import pandas as pd


class FileState(Enum):
    ADDED = auto()
    MODIFIED = auto()
    DELETED = auto()
    SAME = auto()


@dataclass
class FileMeta:
    path: str
    state: FileState = FileState.SAME
    day: datetime = field(init=False)
    uuid: UUID = field(init=False)
    features: int = field(init=False)
    samples: int = field(init=False)

    def __post_init__(self) -> None:
        df = pd.read_feather(self.path)
        self.samples, self.features = df.shape

        file_name = self.path.rsplit(os.path.sep, maxsplit=1)[-1]
        uuid, day_str = file_name.split("_", maxsplit=1)
        day_str = day_str.split("_")[0].split(".")[0]

        self.day = datetime.fromisoformat(day_str)
        self.uuid = UUID(uuid)

    def as_dict(self) -> Dict:
        return {
            "path": self.path,
            "day": self.day.strptime("YYYY-MM-DD"),
            "features": self.features,
            "samples": self.samples,
        }

    @staticmethod
    def from_dict(d: Dict) -> "FileMeta":
        return FileMeta(path=d["path"])


if __name__ == "__main__":
    fm = FileMeta(
        "../../../../Comprisk/Comprisk/tests/test-debug-dataset/0b3426b3-94af-4b8c-89c1-da57493c099a_2025-02-03.feather"
    )
    print(fm)
