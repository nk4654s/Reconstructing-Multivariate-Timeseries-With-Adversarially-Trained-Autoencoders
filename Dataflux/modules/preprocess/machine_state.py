"""This module removes rows from the dataframe if the machine is off."""

from typing import List, Tuple
from pandas import DataFrame
import numpy as np
from Dataflux.modules import BasePreProcessor
from Dataflux.helper import Meta, Metadata, get_logger, simple_graph

logger = get_logger(__name__)

GRAPH_FILE_NAME = "split_statistics.png"
GRAPH_TITLE = "Data Splits Per Day"
GRAPH_XLABEL = "Date"
GRAPH_YLABEL = "# Splits"


class MachineStateSplit(BasePreProcessor):
    """This class removes rows from the dataframe if the machine is off."""

    STATES = ["on", "off"]

    def __init__(self, on_key: str = "Laeuft", keep_state: str = "on",
                 state_change_threshold: int = 900):
        """The constructor of the class. This will be automatically executed by
        the CLIArgumentParser.

        Args:
            on_key (str): The key of the column that contains the machine state.
                Defaults to "Laeuft".
            keep_state (str): Wether to split when machine is on/off. Defaults to 'on'.
            state_change_threshold (int): Remove the first x seconds of a dataframe, if there
                was a statechange. Defaults to 900 which equals to 15 minutes.
        """
        self._column_key = on_key
        self._state = keep_state
        self._threshold = state_change_threshold
        self._stats = {}

    def _trim(self, df: DataFrame) -> DataFrame:
        running = df[self._column_key].to_numpy()
        if self._state == "on":
            start_idx = np.argmax(running > 0)
        else:
            start_idx = np.argmax(running < 1)
        return df.iloc[start_idx:]

    def _split(self, df: DataFrame) -> Tuple[DataFrame]:
        running = df[self._column_key].to_numpy()

        if self._state == "on":
            end_idx = np.argmax(running < 1)
        else:
            end_idx = np.argmax(running > 0)

        # if the end index is 0, return two empty dataset.
        state = 0 if self._state == "on" else 1
        if end_idx == 0 and running[0] == state:
            return df.iloc[:end_idx], df.iloc[:end_idx]

        return df.iloc[:end_idx], df.iloc[end_idx:]

    def _split_algorithm(self, df: DataFrame) -> List[DataFrame]:
        df_splits = []
        to_split = df.copy()
        trimmed = to_split

        while not trimmed.empty:
            trimmed = self._trim(to_split)
            trimmed, to_split = self._split(trimmed)
            if not trimmed.empty:
                df_splits.append(trimmed)

        if not to_split.empty:
            df_splits.append(to_split)

        return df_splits

    def _apply_threshold(self, dfs: List[DataFrame], state_change: bool) -> List[DataFrame]:
        reduced = []
        # remove the first x datapoints of the first dataframe if there was a state change
        # otherwise, the machine was on and there is just another day recorded. Threshold
        # should not be applied in that case.
        if state_change:
            reduced.append(dfs[0].iloc[self._threshold:])
        else:
            reduced.append(dfs[0])

        if len(dfs) > 0:
            # remove the first x datapoints of all other dataframes, such dataframes
            # must have a state change
            for i in range(1, len(dfs)):
                reduced.append(dfs[i].iloc[self._threshold:])
        return reduced

    def on_download_finished(
        self, df: DataFrame, metadata: Metadata = None
    ) -> List[DataFrame]:
        df_splits = self._split_algorithm(df)

        if len(df_splits) == 0:
            logger.info(
                "Dataset included machine %s data only. Nothing to do.", self._state
            )
            return []

        logger.debug("Splited DataFrame into %s DataFrame(s)", len(df_splits))
        self._stats[metadata[Meta.DAY]] = len(df_splits)

        # inform any handler, that the machine state did not change if there is a single
        # dataframe in the splits array with the same length as the original
        state_change = not (len(df_splits) == 1 and df_splits[0].shape == df.shape)
        reduced_splits = self._apply_threshold(df_splits, state_change)
        state_change = state_change if len(reduced_splits) == len(df_splits) else True

        metadata.put(state_change, Meta.MACHINE_STATE_CHANGE)
        return reduced_splits

    def on_processing_finished(self, metadata: Metadata) -> None:
        days = list(self._stats.keys())
        vals = list(self._stats.values())

        # generate graph
        ax = simple_graph(days, vals)(GRAPH_TITLE, GRAPH_XLABEL, GRAPH_YLABEL)

        # add graph to metadata
        metadata.put_list((ax, GRAPH_FILE_NAME), Meta.ADDITIONAL_REPORT_GRAPHS)
