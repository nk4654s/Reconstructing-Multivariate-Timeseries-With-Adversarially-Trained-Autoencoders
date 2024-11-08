"""This module contains several classes to ensure homogeneous value
distribution across the dataset."""

from typing import Any, Dict
import pandas as pd

from Dataflux.data.header import Header
from Dataflux.helper import Meta, Metadata
from Dataflux.diskio import FileWorkerCallable, FileWorkerContext, CallableDependency
from Dataflux.modules.normalize import NormalizeFit
from Dataflux.modules.rename import Rename, ColumnContext

from Dataflux import logger
        

class Consistency(FileWorkerCallable):
    """Ensures homogeneous columns."""

    def __init__(self, enable: bool = True):
        self._header: Header | None = None
        self._enable = enable
        self._stats = {}
        
    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {
                Rename.__name__: CallableDependency.BEFORE,
                NormalizeFit.__name__: CallableDependency.NOT_BEFORE
            }

    def _drop_empty_cols(self, df: pd. DataFrame) -> pd.DataFrame:
        drop = [c for c in df.columns.values if df[c].isnull().all()]
        return df.drop(columns=drop, errors="ignore")

    def run(self, file_path: str, context: FileWorkerContext) -> Any:
        # make sure to remove columns which are completely empty
        df = pd.read_hdf(file_path, key="data")
        if df.empty or isinstance(df, pd.Series):
            return
        
        df = self._drop_empty_cols(df)

        incoming = Header()
        incoming.values = list(df.columns.values)
        incoming.relative_complement(context.get(ColumnContext.HEADER))

        if incoming.drop:
            df.drop(columns=incoming.drop, axis=1, inplace=True, errors="ignore")
            logger.debug("Removed %s columns", len(incoming.drop))

        df.to_hdf(file_path, key="data")

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        header = context.get(ColumnContext.HEADER)
        context.clear(ColumnContext.HEADER)

        for file_dict in metadata[Meta.FILES]:
            file_dict["features"] = len(header)