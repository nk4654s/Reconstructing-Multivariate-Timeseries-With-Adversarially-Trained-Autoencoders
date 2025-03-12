import os
import shutil
from typing import Dict

from Dataflux import logger
from Dataflux.diskio import CALLABLE_RETURN_TYPE, CallableDependency, FileWorkerCallable, FileWorkerContext
from Dataflux.diskio.meta import FileMeta, Metadata
from Dataflux.helper import deterministic_split_algorithm


class SplitDataset(FileWorkerCallable):
    """This module splits the dataset into two separate datasets given a factor
    between 0 and 1. The split should be executed as final module, as the
    split dataset will not be further processed.
    
    Args:
        percentage (float): The percentage of the dataset to be used for
            the split. Defaults to 0.2.
        name (str): The name of the split dataset.
        overwrite (bool): Whether to overwrite the existing dataset or not.
    """
    def __init__(self, percentage: float = 0.2, name: str = "dataset-split", overwrite: bool = False) -> None:
        self._percentage = percentage
        self._name = name

        # build the dataset folder structure
        self.active = True
        if os.path.exists(name):
            if overwrite:
                shutil.rmtree(name)
                os.makedirs(os.path.join(name, "metadata"))
            else:
                self.active = False
        else:
            os.makedirs(os.path.join(name, "metadata"))

        self._files = []
        self._total = 0


    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {SplitDataset.__name__: CallableDependency.FINAL}

    def run(self, fm: FileMeta, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        return fm

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        if not self.active:
            return

        # split the dataset according to each dataframes metadata
        trim_files, split_files = deterministic_split_algorithm(metadata, self._percentage)

        # update the total recordings and files in the metadata
        metadata.files = {e["path"]: e for e in trim_files}

        # copy the metadata of the original dataset to modify it for the
        # split dataset
        split_metadata = Metadata(self._name, metadata.uuid)

        # finally move all day-files from the trimmed dataset to the split one
        for fm in split_files:
            split_path = fm.path.replace(metadata.path, self._name)
            if os.path.exists(fm.path):
                fm.path = split_path
                shutil.move(fm.path, split_path)
                split_metadata.files[split_path] = fm
            else:
                logger.warning("Tried to move a non-existing file to split dataset (%s).", fm.path)

        # the metadata from the arguments will be saved automatically, but the
        # split_metadata not. Hence, save it from here
        split_metadata.save(self._name)
