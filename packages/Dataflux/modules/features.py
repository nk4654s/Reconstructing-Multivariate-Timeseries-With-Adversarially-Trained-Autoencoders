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

        self._deletion_causes = []

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {NormalizeFit.__name__: CallableDependency.NOT_BEFORE}
    
    def _check_integrity(self, column_names: np.ndarray) -> list:
        return [n for n in self._include if n not in column_names]

    def run(self, fm: FileMeta, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:  # noqa: F821
        df = fm.read()

        # mark file to be deleted if the number of features does not align with the rest of the files
        missing_columns = self._check_integrity(df.columns.values)
        if not self._force_features and len(missing_columns) != 0:
            fm.state = FileState.DELETED
            self._deletion_causes += missing_columns
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
        missing_columns = set(self._deletion_causes)
        logger.debug("Removed dataset files due to %s columns missing.", missing_columns)
