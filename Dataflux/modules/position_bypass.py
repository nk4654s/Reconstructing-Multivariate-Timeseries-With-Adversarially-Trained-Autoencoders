"""This module pre-processes a dataframe and insters values to NAN."""
import os
from typing import Any, Dict
import numpy as np
import pandas as pd

from Dataflux.diskio import FileWorkerCallable, FileWorkerContext as Context
from Dataflux.diskio.callable import CallableDependency
from Dataflux.helper import Meta
from Dataflux.helper import Metadata
from Dataflux import logger
from Dataflux.modules.normalize import NormalizeFit
from Dataflux.modules.rename import Rename


class PositionBypassValve(FileWorkerCallable):
    """This mdoule drops bypass ventils from the dataframe. In addition, everytime the bypass ventil is
    opened, a timespan `t` can be removed from the dataset.

    Args:
        remove_timespan (int): The timespan to remove from the dataframe. Defaults to 600 seconds.
    """

    def __init__(self, remove_timespan: int = 600):
        super().__init__()

        self._update = {}
        self._threshold = int(remove_timespan)

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {
            Rename.__name__: CallableDependency.BEFORE,
            NormalizeFit.__name__: CallableDependency.NOT_BEFORE
        }

    def run(self, file_path: str, context: Context) -> Any:
        df = pd.read_hdf(file_path, keys="data")
        cols = [c for c in df.columns.values if "PositionBypassregelventilPCV74381 (%)" in c]

        # copied from Anacision, malfunctions possible
        lower = np.argwhere(df[cols].to_numpy())[:, 0]
        if len(lower) == 0:
            return

        upper = lower + self._threshold

        # Combine lower and upper into a list of tuples and sort by the lower bounds
        intervals = sorted(zip(lower, upper, strict=True))

        # Merge overlapping intervals
        merged_intervals = []
        current_start, current_end = intervals[0]

        for start, end in intervals[1:]:
            if start <= current_end:
                # If intervals overlap, merge them
                current_end = max(current_end, end)
            else:
                # If they don't overlap, add the previous interval and start a new one
                merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end

        # Add the last interval
        merged_intervals.append((current_start, current_end))

        # Create a boolean series to mark data within the specified ranges
        mask = pd.Series(False, index=df.index)
        for l, u in merged_intervals:
            mask |= (df.index >= l) & (df.index <= u)

        # Invert the mask to filter out the data within the specified ranges
        filtered_data = df[~mask]
        filtered_data.to_hdf(file_path, key="data")

        day = os.path.basename(file_path).split("_", maxsplit=1)[-1][:-3]
        self._update[day] = len(filtered_data)

    def finish(self, context: Context, metadata: Metadata) -> None:
        for file_info in metadata[Meta.FILES]:
            day = file_info[Meta.DAY.name]
            recordings = self._update.get(day, None)
            if recordings:
                file_info["recordings"] = recordings
                logger.debug("PositionBypassValve on day %s resulted in %s samples left", day, recordings)
