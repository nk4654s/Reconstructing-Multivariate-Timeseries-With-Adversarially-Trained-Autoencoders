"""This module pre-processes a dataframe and insters values to NAN."""
import numpy as np
from pandas import DataFrame, Series

from Dataflux.helper import simple_graph, Meta
from Dataflux.modules import BasePreProcessor
from Dataflux.helper import Metadata
from Dataflux import logger


GRAPH_HTS_TITLE = "Time Steps Removed after Bypass Ventil"
GRAPH_HTS_XLABEL = "Date"
GRAPH_HTS_YLABEL = "Time Steps Removed (%)"
GRAPH_HTS_FILE_NAME = "position_bypass_ventil_removalpng"


class PositionBypassRemoval(BasePreProcessor):
    """This mdoule drops bypass ventils from the dataframe. In addition, everytime the bypass ventil is
    opened, a timespan `t` can be removed from the dataset.

    Args:
        remove_timespan (int): The timespan to remove from the dataframe. Defaults to 600 seconds.
    """

    def __init__(self, remove_timespan: int = 600):
        super().__init__()

        self._stats = {}
        self._rm_rows = int(remove_timespan)

    def on_download_finished(self, df: DataFrame, metadata: Metadata) -> DataFrame:
        logger.debug("Removing position bypass ventil from dataset file.")
        cols = [c for c in df.columns.values if "PositionBypassregelventilPCV74381 (%)" in c]

        # copied from Anacision, malfunctions possible
        lower = np.argwhere(df[cols].to_numpy())[:, 0]
        if len(lower) == 0:
            self._stats[metadata[Meta.DAY]] = 0
            return df

        upper = lower + self._rm_rows

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
        mask = Series(False, index=df.index)
        for l, u in merged_intervals:
            mask |= (df.index >= l) & (df.index <= u)

        # Invert the mask to filter out the data within the specified ranges
        filtered_data = df[~mask]

        logger.debug("Removing bypass ventils from dataset file resulted in %s dropped columns", len(cols))
        logger.debug("Removed %s rows due to opened bypass ventils.", len(df) - len(filtered_data))

        self._stats[metadata[Meta.DAY]] = round(
            (len(df) - len(filtered_data)) / len(df) * 100, 3
        )

        return filtered_data

    def on_processing_finished(self, metadata: Metadata) -> None:
        days = list(self._stats.keys())
        vals = list(self._stats.values())

        # generate graph
        ax = simple_graph(days, vals)(
            GRAPH_HTS_TITLE, GRAPH_HTS_XLABEL, GRAPH_HTS_YLABEL
        )

        # add graph to metadata
        metadata.put_list((ax, GRAPH_HTS_FILE_NAME), Meta.ADDITIONAL_REPORT_GRAPHS)
