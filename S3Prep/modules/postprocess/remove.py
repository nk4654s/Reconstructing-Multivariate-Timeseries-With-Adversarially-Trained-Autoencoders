from typing import Any, List

import pandas as pd
from S3Prep.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency
from S3Prep.helper import Metadata, Meta
from S3Prep.modules.postprocess.normalize import NormalizeFit


class RemoveColumns(FileWorkerCallable):
    """This module removes specified columns from the dataframe"""
    def __init__(self, columns: str = "") -> None:
        self._columns = columns.split(" ")
        self._num_columns = 0

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
