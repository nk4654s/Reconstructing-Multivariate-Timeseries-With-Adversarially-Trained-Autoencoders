import os
import shutil
from typing import Dict, List, Tuple
from copy import deepcopy, copy

import pandas as pd

from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency, CALLABLE_RETURN_TYPE
from Dataflux.helper import Metadata, Meta
from Dataflux import logger


def deterministic_split_algorithm(meta: Metadata, percentage: float) -> Tuple[List, List]:
    """
    This function splits the days of a dataset's metadata into two lists. The first list contains
    the days which make up `1 - percentage` of total recordings. The second list contains the remaining days.

    Args:
        meta (Metadata): Metadata object containing the days.
        percentage (float): The percentage of days that should be in the
            splitted list.
    
    Returns:
        Tuple[List]: A tuple containing two lists. The first list contains the first `1 - percentage`
            and the second list contains the remaining days.
    """
    # get the days in a sorted manner
    days = meta[Meta.FILES]
    total_recordings = meta[Meta.TOTAL_RECORDINGS]
    days = sorted(days, key=lambda x: x["recordings"])

    # copy the list of days as the algorithm will remove files from one list and
    # add them to another list
    trimmed_days = copy(days)
    target_recordings = int(total_recordings * percentage)

    # add days to the list until the target_recordings is reached
    selected_days = []
    iterative_recordings = 0
    for i in range(len(days)):
        if iterative_recordings >= target_recordings:
            break

        # also, remove the day from the list of days to be trimmed
        day = trimmed_days.pop(i - len(selected_days))
        selected_days.append(day)
        iterative_recordings += day["recordings"]
    
    # as the list of days is sorted, we can attempt to remove some days from the selected
    # list and put them back to the trimmed list
    while iterative_recordings - selected_days[0]["recordings"] > target_recordings:
        day = selected_days.pop(0)
        iterative_recordings -= day["recordings"]
        trimmed_days.append(day)

    # log some data
    actual_percentage = round(iterative_recordings / total_recordings * 100, 2)
    logger.debug("Targeted %s%% and setteled on %s%%", percentage * 100, actual_percentage)
    logger.debug("Training set has %s days and validation set has %s", len(trimmed_days), len(selected_days))

    return trimmed_days, selected_days


class SplitDataset(FileWorkerCallable):
    """This module splits the dataset into two separate datasets given a factor
    between 0 and 1. The split should be executed as final module, as the
    splitted dataset will not be further processed.
    
    Args:
        percentage (float): The percentage of the dataset to be used for
            the split. Defaults to 0.2.
        name (str): The name of the splitted dataset.
        overwirte (bool): Whether to overwrite the existing dataset or not.
    """
    def __init__(self, percentage: float = 0.2, name: str = "dataset-splitted", overwrite: bool = False) -> None:
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

    def run(self, file_path: str, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        df = pd.read_feather(file_path)
        info = {
            Meta.DAY.name: file_path.split("_", maxsplit=1)[-1][:-3],
            "recordings": df.shape[0],
            "features": df.shape[1]
        }

        self._files.append(info)
        self._total += df.shape[0]

        return None, None

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        if not self.active:
            return
        
        # make sure to have the correct metadata before performing a split
        metadata.put(self._total, Meta.TOTAL_RECORDINGS)
        metadata.put(self._files, Meta.FILES)

        # split the dataset according to each dataframes metadata
        trimmed_days, splitted_days = deterministic_split_algorithm(metadata, self._percentage)
        trimmed_total = sum([d["recordings"] for d in trimmed_days])
        splitted_total = sum([d["recordings"] for d in splitted_days])

        # update the total recordings and files in the metadata
        metadata.put(trimmed_total, Meta.TOTAL_RECORDINGS)
        metadata.put(trimmed_days, Meta.FILES)

        # copy the metadata of the original dataset to modify it for the
        # splitted dataset
        splitted_metadata = deepcopy(metadata)

        # update the metadata of the splitted dataset
        splitted_metadata.put(self._name, Meta.PATH)
        splitted_metadata.put(splitted_total, Meta.TOTAL_RECORDINGS)
        splitted_metadata.put(splitted_days, Meta.FILES)

        # the metadata from the arguments will be saved automatically, but the
        # splitted metadata not. Hence, save it from here
        splitted_metadata.save(self._name)
        logger.debug("Cut produces the sizes %s and %s", metadata[Meta.TOTAL_RECORDINGS], splitted_metadata[Meta.TOTAL_RECORDINGS])

        # finally move all day-files from the trimmed dataset to the splitted one
        uuid = metadata[Meta.MACHINE_UUID]
        path = metadata[Meta.PATH]
        move = [f"{path}{os.path.sep}{uuid}_{day[Meta.DAY.name]}.h5" for day in splitted_days]
        for path in move:
            splitted_path = path.replace(metadata[Meta.PATH], self._name)
            if os.path.exists(path):
                shutil.move(path, splitted_path)
            else:
                logger.warning("Tried to move a non-existing file to splitted dataset (%s).", path)
        
        # do not forget to move the normalizer
        trimmed_norm = os.path.join(metadata[Meta.PATH], "metadata", "normalizer.pkl")
        splitted_norm = os.path.join(self._name, "metadata", "normalizer.pkl")
        shutil.copyfile(trimmed_norm, splitted_norm)


