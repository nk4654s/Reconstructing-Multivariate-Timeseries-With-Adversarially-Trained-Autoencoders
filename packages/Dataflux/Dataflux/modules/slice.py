import os
from typing import Dict, List
from threading import Lock
import numpy as np
import pandas as pd

from Dataflux import logger
from Dataflux.diskio import (
    FileWorkerCallable,
    FileWorkerContext,
    CallableDependency,
    CALLABLE_RETURN_TYPE,
)
from Dataflux.helper import Metadata, Meta
from Dataflux.modules.normalize import NormalizeFit

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

    def run(self, file_path: str, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        df = pd.read_feather(file_path)
        df = df.set_index("ts")

        day = os.path.basename(file_path).split("_", maxsplit=1)[-1]
        day = day.split(".")[0]
        os.remove(file_path)

        # check if the df is within the timespan
        if df.index[-1] < self._start_ts or df.index[0] > self._end_ts:
            return None, [file_path]

        df = df.loc[self._start_ts : self._end_ts]
        df = df.reset_index()
        df.to_feather(file_path)

        self._update[day] = len(df)

        return None, None

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        for i, file_info in enumerate(metadata[Meta.FILES]):
            day = file_info[Meta.DAY.name]
            update = self._update.pop(day, file_info)
            if update:
                metadata[Meta.FILES][i]["recordings"] = update

        metadata.put(
            sum([i["recordings"] for i in metadata[Meta.FILES]]), Meta.TOTAL_RECORDINGS
        )
        logger.debug("Total length now %s", metadata[Meta.TOTAL_RECORDINGS])
