from typing import Any, Dict, List
from itertools import product
from threading import Lock
import pandas as pd
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency, CALLABLE_RETURN_TYPE
from Dataflux.helper import Metadata, Meta
from Dataflux.modules.normalize import NormalizeFit
from Dataflux.modules.rename import ColumnContext
from Dataflux import logger

mutex = Lock()


class Remove(FileWorkerCallable):
    """This module removes specified columns from the dataframe.
    
    Args:
        columns (str): The list of column names to remove, provided as string
            where each list entry is separated by a ','.
    """
    def __init__(self, columns: List[str] = []) -> None:
        self._column_substrings = columns
        self._columns = []

        self._num_columns = 0
        self._operated = []

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {NormalizeFit.__name__: CallableDependency.NOT_BEFORE}

    def _find_corresponding_columns(self, context: FileWorkerContext) -> None:
        if not self._columns and self._column_substrings:
            with mutex:
                header = context.get(ColumnContext.HEADER)
                assert header, "Something went wrong when passing the header to Remove-module."

                combinations = product(self._column_substrings, header)
                self._columns = [name for sub, name in combinations if name.startswith(sub)]
                logger.debug("RemoveColumns will remove %s", self._columns)

    def run(self, file_path: str, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        self._find_corresponding_columns(context)

        df = pd.read_hdf(file_path, key="data")
        df = df.drop(columns=self._columns, errors="ignore")
        df.to_hdf(file_path, key="data", format="table", mode="w")

        self._num_columns = df.columns.size
        self._operated.append(file_path)

        return None, None

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        for file_dict in metadata[Meta.FILES]:
            file_dict["features"] = self._num_columns
