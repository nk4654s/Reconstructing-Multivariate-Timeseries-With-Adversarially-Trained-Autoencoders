"""This module handles the metadata class and content."""
from dataclasses import dataclass, field
import os
from typing import Dict
from uuid import UUID

import yaml

from Dataflux.diskio.meta import FileMeta


@dataclass
class Metadata:
    path: str
    uuid: UUID
    files: Dict[str, FileMeta] = field(default_factory=list)
    total_recordings: int = 0

    def as_dict(self) -> Dict:
        files = {}
        for key, val in self.files.items():
            d = val.as_dict()
            files[d.pop("path")] = d

        return {
            "machine_uuid": self.uuid,
            "path": self.path,
            "files": files,
            "total_recordings": sum([f.samples for f in self.files])
        }

    @property
    def normalizer_path(self):
        return os.path.join(self.path, "metadata", "normalizer.pkl")

    def save(self, base_dir: str) -> None:
        """Saves the metadata to a yaml-file in `/metadata/info.yaml`

        Args:
            base_dir (str): Path of the root_folder to save metadata in.
        """
        file_path = os.path.join(base_dir, "metadata", "info.yaml")
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.as_dict(), f)

    @staticmethod
    def load(base_dir: str) -> "Metadata":
        """Loads the metadata from a yaml-file. Needs to be in a subfolder
        `/metadata/info.yaml`.

        Args:
            base_dir (str): Path of the root_folder to load metadata from.
        """
        file_path = os.path.join(base_dir, "metadata", "info.yaml")
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return Metadata(
                path=data["path"],
                uuid=data["uuid"],
                files={k: FileMeta.from_dict(v) for k, v in data["files"].items()},
                total_recordings=data["total_recordings"],
            )
