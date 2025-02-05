from typing import Dict
from threading import Lock

from Dataflux import logger
from Dataflux.diskio import (
    FileWorkerCallable,
    FileWorkerContext,
    CallableDependency,
    CALLABLE_RETURN_TYPE,
)
from Dataflux.diskio.meta import FileMeta, Metadata, FileState

mutex = Lock()


class Slice(FileWorkerCallable):
    """This module removes specified columns from the dataframe.

    Args:
        start_ts (float): The start timestep where to slice.
        end_ts (float): The end timestep where to slice.
    """

    def __init__(self, start_ts: float, end_ts: float) -> None:
        self._start_ts = int(start_ts)
        self._end_ts = int(end_ts)

        self._update = {}

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {Slice.__name__: CallableDependency.FINAL}

    def run(self, fm: FileMeta, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        df = fm.read()
        df = df.set_index("ts")

        # check if the df is within the timespan
        if df.index[-1] < self._start_ts or df.index[0] > self._end_ts:
            fm.state = FileState.DELETED
            return fm

        df = df.loc[self._start_ts : self._end_ts]
        df = df.reset_index()
        fm.save(df)

        return fm

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        # requesting the metadata as dict recalculates the total number of recordings
        _ = metadata.as_dict()
        logger.debug("Slicing finished: length=%s", metadata.total_recordings)
