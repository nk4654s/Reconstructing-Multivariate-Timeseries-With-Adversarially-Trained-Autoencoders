from typing import Dict, List

import numpy as np

from Dataflux import logger
from Dataflux.diskio import CALLABLE_RETURN_TYPE, CallableDependency, FileWorkerCallable, FileWorkerContext
from Dataflux.diskio.meta import FileMeta, FileState, Metadata
from Dataflux.modules.normalize import NormalizeFit


class Features(FileWorkerCallable):
    """This module removes specified columns from the dataframe.
    
    Args:
        include (str): The list of column names to keep, provided as string
            where each list entry is separated by a ','.
        force_features (bool): If set to True, all columns of "include" will be present, but may contain NaN.
    """
    def __init__(self, include: List[str] = None, force_features: bool = False) -> None:
        include = include if include else []
        self._include = include + ["ts"] if "ts" not in include else include
        self._force_features = force_features

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {NormalizeFit.__name__: CallableDependency.NOT_BEFORE}
    
    def _check_integrity(self, column_names: np.ndarray) -> bool:
        found = sum([1 for name in self._include if name in column_names])
        return found == len(self._include)

    def run(self, fm: FileMeta, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:  # noqa: F821
        df = fm.read()

        # mark file to be deleted if the number of features does not align with the rest of the files
        if not self._force_features and not self._check_integrity(df.columns.values):
            fm.state = FileState.DELETED
            return fm

        # adjust the content by only including certain features
        if self._force_features:
            for feature in self._include:
                if feature not in df.columns:
                    df[feature] = np.nan

        df = df[self._include]
        fm.features = df.shape[1]

        # update the file
        fm.save(df)
        return fm

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        logger.debug("Finished including defined (n=%s) features only", len(self._include))
