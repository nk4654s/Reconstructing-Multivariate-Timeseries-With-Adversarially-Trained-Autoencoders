"""This module removes rows from the dataframe if the machine is off."""

from typing import List, Tuple
from pandas import DataFrame
import numpy as np
from Dataflux.modules import BasePreProcessor
from Dataflux.helper import Meta, Metadata, simple_graph
from Dataflux import logger

GRAPH_FILE_NAME = "split_statistics.png"
GRAPH_TITLE = "Data Splits Per Day"
GRAPH_XLABEL = "Date"
GRAPH_YLABEL = "# Splits"


def trim(df: DataFrame, key: str, state_on: bool = True) -> DataFrame:
    """
    This method removes all indicies at the start of the given dataframes,
    until the column for the given key is on/off.
    
    Args:
        df (DataFrame): The DataFrame to be trimmed.
        key (str): The column name which defines the trim.
        state_on (bool): Remove values that are 1 until a value gets 0 when
            setting `state_on` to False. Defaults to True, which means all
            leading 0s are removed.
    """
    if df.empty:
        return df

    running = df[key].to_numpy()
    if state_on:
        start_idx = np.argmax(running > 0)
        return df.iloc[start_idx:]

    start_idx = np.argmax(running < 1)
    return df.iloc[start_idx:]


def split(df: DataFrame, key: str, state_on: bool = True) -> Tuple[DataFrame, DataFrame]:
    """This method splits a DataFrame into two DataFrames where the `state_on`
    variable matches those of the given key.

    Args:
        df (DataFrame): The DataFrame to be splitted.
        key (str): The column name which defines the split.
        state_on (bool): Provide `True` to split at the transition from 1 to 0, else
            give `False`. Defaults to True.
    """
    if df.empty:
        return df, df

    running = df[key].to_numpy()

    if state_on:
        end_idx = np.argmax(running < 1)
    else:
        end_idx = np.argmax(running > 0)

    # if the end index is 0, return two empty dataset.
    if not end_idx:
        return df.iloc[:end_idx], DataFrame({key: []})

    return df.iloc[:end_idx], df.iloc[end_idx:]

def multi_split(df: DataFrame, key: str, state_on: bool = True) -> List[DataFrame]:
    """This method splits a DataFrame into a List of DataFrames where the `state_on`
    variable matches those of the given key.

    Args:
        df (DataFrame): The DataFrame to be splitted.
        key (str): The column name which defines the split.
        state_on (bool): Provide `True` to split at the transition from 1 to 0, else
            give `False`. Defaults to True.
    """
    dfs = []
    state = 1 if state_on else 0
    if df[key].mean() == state:
        return [df]

    while not df.empty:
        df = trim(df, key, state_on)
        df1, df = split(df, key, state_on)
        if not df1.empty:
            dfs.append(df1)
    return dfs


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

    def _apply_threshold(self, dfs: List[DataFrame], state_change: bool) -> List[DataFrame]:
        reduced = []
        # remove the first x datapoints of the first dataframe if there was a state change
        # otherwise, the machine was on and there is just another day recorded. Threshold
        # should not be applied in that case.
        if state_change:
            if len(dfs[0]) > 2 * self._threshold:
                reduced.append(dfs[0].iloc[self._threshold:-self._threshold])
        else:
            reduced.append(dfs[0])

        if len(dfs) > 0:
            # remove the first x datapoints of all other dataframes, such dataframes
            # must have a state change
            for i in range(1, len(dfs)):
                if len(dfs[i]) > 2 * self._threshold:
                    reduced.append(dfs[i].iloc[self._threshold:-self._threshold])
        return reduced

    def on_download_finished(
        self, df: DataFrame, metadata: Metadata = None
    ) -> List[DataFrame]:
        df_splits = multi_split(df.copy(), self._column_key, self._state == "on")

        if len(df_splits) == 0:
            return []

        logger.debug("Splited DataFrame into %s DataFrame(s)", len(df_splits))
        self._stats[metadata[Meta.DAY]] = len(df_splits)

        # inform any handler, that the machine state did not change if there is a single
        # dataframe in the splits array with the same length as the original
        # not (True and )
        state_change = not (len(df_splits) == 1 and df_splits[0].shape == df.shape)
        logger.debug("Detected state change: %s", "yes" if state_change else "no")

        reduced_splits = self._apply_threshold(df_splits, state_change)
        state_change = state_change if len(reduced_splits) == len(df_splits) else True
        logger.debug("Reduced number of DataFrame(s) after applying threshold to: %s",
                     len(reduced_splits))
        logger.debug("Detected state change after thresholding: %s", state_change)

        metadata.put(state_change, Meta.MACHINE_STATE_CHANGE)
        return reduced_splits

    def on_processing_finished(self, metadata: Metadata) -> None:
        days = list(self._stats.keys())
        vals = list(self._stats.values())

        # generate graph
        ax = simple_graph(days, vals)(GRAPH_TITLE, GRAPH_XLABEL, GRAPH_YLABEL)

        # add graph to metadata
        metadata.put_list((ax, GRAPH_FILE_NAME), Meta.ADDITIONAL_REPORT_GRAPHS)
