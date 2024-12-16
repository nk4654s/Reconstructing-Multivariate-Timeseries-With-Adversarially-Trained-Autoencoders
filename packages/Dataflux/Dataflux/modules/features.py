from typing import Dict, List
from threading import Lock
import numpy as np
import pandas as pd
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency, CALLABLE_RETURN_TYPE
from Dataflux.helper import Metadata, Meta
from Dataflux.modules.normalize import NormalizeFit

mutex = Lock()


class Features(FileWorkerCallable):
    """This module removes specified columns from the dataframe.
    
    Args:
        include (str): The list of column names to keep, provided as string
            where each list entry is separated by a ','.
    """
    def __init__(self, include: List[str] = []) -> None:
        self._include = include + ["ts"] if "ts" not in include else include
        self._drop = []

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {NormalizeFit.__name__: CallableDependency.NOT_BEFORE}
    
    def _check_integrity(self, column_names: np.ndarray) -> bool:
        found = sum([1 for name in self._include if name in column_names])
        return found == len(self._include)

    def run(self, file_path: str, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        df = pd.read_hdf(file_path, key="data")
        if not self._check_integrity(df.columns.values):
            return None, [file_path]
        
        df = df[self._include]
        df.to_hdf(file_path, key="data", format="table", mode="w")

        return None, None

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        for file_dict in metadata[Meta.FILES]:
            file_dict["features"] = len(self._include)
