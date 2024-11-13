"""This module contains several classes to ensure homogeneous value
distribution across the dataset."""

from typing import Dict, List
from itertools import product
from threading import Lock
import pandas as pd

from Dataflux.data.header import Header
from Dataflux.helper import Meta, Metadata
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency, CALLABLE_RETURN_TYPE
from Dataflux.modules.normalize import NormalizeFit
from Dataflux.modules.rename import Rename, ColumnContext

from Dataflux import logger
mutex = Lock()


class Consistency(FileWorkerCallable):
    """Ensures homogeneous columns."""

    def __init__(self, enable: bool = True, keep: List = []):
        self._header: Header | None = None
        self._keep_substrings = keep
        self._keep = []
        self._enable = enable
        self._stats = {}
        
    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {
                Rename.__name__: CallableDependency.BEFORE,
                NormalizeFit.__name__: CallableDependency.NOT_BEFORE
            }

    def _find_corresponding_columns(self, incoming: Header, context: FileWorkerContext) -> None:
        if not self._keep and self._keep_substrings:
            with mutex:
                header = context.get(ColumnContext.HEADER)
                assert header, "Something went wrong when passing the header to Remove-module."

                combinations = product(self._keep_substrings, header)
                self._keep = [name for sub, name in combinations if name.startswith(sub)]
                logger.debug("Consistency will force keep %s", self._keep)
        
        incoming.drop = [d for d in incoming.drop if d not in self._keep]

    def _drop_empty_cols(self, df: pd. DataFrame) -> pd.DataFrame:
        drop = [c for c in df.columns.values if df[c].isnull().all()]
        return df.drop(columns=drop, errors="ignore")

    def run(self, file_path: str, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        # make sure to remove columns which are completely empty
        df = pd.read_hdf(file_path, key="data")
        if df.empty or isinstance(df, pd.Series):
            return None, None
        
        df = self._drop_empty_cols(df)

        incoming = Header()
        incoming.values = list(df.columns.values)
        incoming.relative_complement(context.get(ColumnContext.HEADER))
        self._find_corresponding_columns(incoming, context)

        if incoming.drop:
            df.drop(columns=incoming.drop, axis=1, inplace=True, errors="ignore")

        df.to_hdf(file_path, key="data")
        return None, None

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        header = context.get(ColumnContext.HEADER)

        for file_dict in metadata[Meta.FILES]:
            file_dict["features"] = len(header)