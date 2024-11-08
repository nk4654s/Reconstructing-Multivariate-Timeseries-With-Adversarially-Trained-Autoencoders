"""This module removes rows from the dataframe if the machine is off."""

import os
from typing import Any, Dict, List, Tuple, Union
from pandas import DataFrame
import pandas as pd
import numpy as np
from Dataflux.diskio import FileWorkerCallable, CallableDependency, CALLABLE_RETURN_TYPE
from Dataflux.diskio import FileWorkerContext as Context
from Dataflux.helper import Meta, Metadata
from Dataflux import logger
from Dataflux.modules.normalize import NormalizeFit
from Dataflux.modules.remove import Remove


def trim(df: DataFrame, key: str, value: int | float, lower: bool = True) -> DataFrame:
    """
    This method removes all indicies at the start of the given dataframes,
    until the column for the given key is on/off.
    
    Args:
        df (DataFrame): The DataFrame to be trimmed.
        key (str): The column name which defines the trim.
        value (int | float): The value above/below which the series is trimmed.
        lower (bool): Remove values that are lower than the provided value. Defaults to
            True.
    """
    if df.empty:
        return df

    running = df[key].to_numpy()
    if lower:
        start_idx = np.argmax(running > value)
        return df.iloc[start_idx:]

    start_idx = np.argmax(running < value)
    return df.iloc[start_idx:]


def split(df: DataFrame, key: str, value: int | float, lower: bool = True) -> Tuple[DataFrame, DataFrame]:
    """This method splits a DataFrame into two DataFrames where the `state_on`
    variable matches those of the given key.

    Args:
        df (DataFrame): The DataFrame to be splitted.
        key (str): The column name which defines the split.
        value (int | float): The value above/below which the series is splitted.
        lower (bool): Remove values that are lower than the provided value. Defaults to
            True.

    Returns:
        Tuple[DataFrame, DataFrame]: Returns the splitted DataFrames, where the scond one can be
            empty, if there was no split possible.
    """
    if df.empty:
        return df, df

    running = df[key].to_numpy()
    lower_vals = running < value
    upper_vals = running > value

    if lower:
        end_idx = np.argmax(lower_vals)
        end_idx = 0 if not end_idx and (lower_vals).all() else end_idx
        end_idx = len(running) - 1 if not end_idx and (upper_vals).all() else end_idx
    else:
        end_idx = np.argmax(upper_vals)
        end_idx = 0 if not end_idx and (upper_vals).all() else end_idx
        end_idx = len(running) - 1 if not end_idx and (lower_vals).all() else end_idx

    # if the end index is 0, return two empty dataset.
    if not end_idx:
        return DataFrame({key: []}), DataFrame({key: []})
    
    # if the end_index equlas the full sequence length, then there is no split and the second DataFrame must be empty
    if end_idx == len(running) - 1:
        return df, DataFrame({key: []})

    return df.iloc[:end_idx], df.iloc[end_idx:]

def multi_split(df: DataFrame, key: str, lower: int | float, upper: int | float) -> List[DataFrame]:
    """This method splits a DataFrame into a List of DataFrames where the `state_on`
    variable matches those of the given key.

    Args:
        df (DataFrame): The DataFrame to be splitted.
        key (str): The column name which defines the split.
        lower (int | float): The value below which the series is splitted.
        lower (int | float): The value above which the series is splitted.
    """
    def internal_split(temp_df, value, v_type):
        df_list = []
        while not temp_df.empty:
            temp_df = trim(temp_df, key, value, v_type)
            df1, temp_df = split(temp_df, key, value, v_type)
            if not df1.empty:
                df_list.append(df1)
        return df_list
    
    # first trim/split with lower border, then apply the same algorithm on every resulting df
    df_final = []
    for d in internal_split(df, lower, True):
        df_final += internal_split(d, upper, False)

    return df_final


def remove_threshold(df: DataFrame | List[DataFrame], threshold: int) -> List[DataFrame]:
    """This method removes a given amount of timesteps from the start and end of a DataFrame.

    Args:
        df (DataFrame | List[DataFrame]): The DataFrame to delete some timesteps from.
        threshold (int): The amount of steps to remove.
    
    Returns:
        List [DataFrame]: Returns the list of modified DataFrames.
    """
    df = df if isinstance(df, List) else [df]
    reduced = []

    # remove the first x datapoints of all dataframes
    if len(df) > 0:
        for i in range(0, len(df)):
            if len(df[i]) > 2 * threshold:
                df[i] = df[i].iloc[threshold:-threshold]
                reduced.append(df[i])
    return reduced


class State(FileWorkerCallable):
    """This class splits a dataset's file according to the provided configuration.
    
        Args:
        on_key (str): The key of the column that contains the machine state.
            Defaults to "Laeuft".
        keep_state (str): Wether to split when machine is on/off. Defaults to 'on'.
        state_change_threshold (int): Remove the first x seconds of a dataframe, if there
            was a statechange. Defaults to 900 which equals to 15 minutes.
    """

    def __init__(self, key: str = "Laeuft", lower: Union[int, float] = 1, upper: Union[int, float] = 1,
                 delete_ts: int = 900):
        self._key = key
        self._lower = lower
        self._upper = upper
        self._threshold = delete_ts
        self._update = {}

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {
                NormalizeFit.__name__: CallableDependency.NOT_BEFORE,
                Remove.__name__: CallableDependency.NOT_BEFORE,
            }
    
    def run(self, file_path: str, context: Context) -> CALLABLE_RETURN_TYPE:
        # read and remove the dataset file, if everyting is fine, it will eb created again
        df = pd.read_hdf(file_path, key="data")
        day = os.path.basename(file_path).split("_", maxsplit=1)[-1][:-3]
        os.remove(file_path)
        self._update[day] = None
        if df.empty or isinstance(df, pd.Series):
            return None, [file_path]

        # split the dataset into multiple smaller ones by removing values
        # smaller or greater than the boundaries
        df_splits = multi_split(df.copy(), self._key, self._lower, self._upper)

        # if the resulting amount of splits is 0, then remove the file
        if len(df_splits) == 0:
            return None, [file_path]
        
        # get the current file's date and create the new filenames
        base_path = context.get(Context.DIRECTORY)
        uuid = context.get(Context.UUID)

        # save all dataframes
        new_files = []
        for i, df in enumerate(remove_threshold(df_splits, self._threshold)):
            postfix = "" if i == 0 else f"_{i}"
            path = os.path.join(base_path, f"{uuid}_{day}{postfix}.h5")
            df.to_hdf(path, key="data")

            self._update[f"{day}{postfix}"] = {
                "recordings": df.shape[0],
                "features": df.shape[1]
            }

            if i > 0:
                new_files.append(path)
        
        return new_files, None

    def finish(self, context: Context, metadata: Metadata) -> None:
        metadata.drop(Meta.TOTAL_RECORDINGS)
        files_update = []
        for file_info in metadata[Meta.FILES]:
            day = file_info[Meta.DAY.name]
            update = self._update.pop(day, file_info)
            if update:
                update[Meta.DAY.name] = day
                files_update.append(update)
        
        logger.debug("State splitter (key=%s) introduced %s splits", self._key, len(self._update))
        logger.debug("Files are %s", self._update)
        for day, info in self._update.items():
            if not info:
                continue

            update = {
                Meta.DAY.name: day,
                "recordings": info["recordings"],
                "features": info["features"]
            }

            files_update.append(update)

        metadata.drop(Meta.FILES)
        metadata.put(files_update, Meta.FILES)
        metadata.put(sum([i["recordings"] for i in metadata[Meta.FILES]]), Meta.TOTAL_RECORDINGS)

