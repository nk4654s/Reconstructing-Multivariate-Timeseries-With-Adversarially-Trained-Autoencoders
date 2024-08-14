from typing import Any, List

import pandas as pd
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency
from Dataflux.helper import Metadata, Meta, get_logger
from Dataflux.modules.postprocess.normalize import NormalizeFit

logger = get_logger(__name__)


class RemoveColumns(FileWorkerCallable):
    """This module removes specified columns from the dataframe.
    
    Args:
        columns (str): The list of column names to remove, provided as string
            where each list entry is separated by a ','.
    """
    def __init__(self, columns: str = "") -> None:
        self._columns = [_.strip() for _ in columns.replace("\"", "").split(",")]
        self._num_columns = 0
        logger.debug("RemoveColumns will remove %s", self._columns)

    def get_dependencies(self) -> List[str]:
        return {NormalizeFit.__name__: CallableDependency.NOT_BEFORE}

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        print("remove_col: " + file_path)
        df = pd.read_hdf(file_path, key="data")
        df = df.drop(columns=self._columns, errors="ignore")
        df.to_hdf(file_path, key="data", format="table", mode="w")
        self._num_columns = df.columns.size

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        for file_dict in metadata[Meta.FILES]:
            file_dict["features"] = self._num_columns
