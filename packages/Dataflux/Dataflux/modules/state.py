"""This module removes rows from the dataframe if the machine is off."""

from typing import Dict, Union
import pandas as pd
from Dataflux.diskio import FileWorkerCallable, CallableDependency, CALLABLE_RETURN_TYPE
from Dataflux.diskio import FileWorkerContext as Context
from Dataflux.diskio.meta import FileMeta, FileState, Metadata
from Dataflux import logger
from Dataflux.helper.dataframe_ops import multi_split, remove_threshold
from Dataflux.modules.normalize import NormalizeFit
from Dataflux.modules.features import Features


class State(FileWorkerCallable):
    """This class splits a dataset's file according to the provided configuration.
    
        Args:
        key (str): The key of the column that contains the machine state.
            Defaults to "Laeuft".
        lower (float | int): The lower boundary under which a state is defined as invalid.
        upper (float | int): The upper boundary above which a state is defined as invalid.
        alpha_threshold (int): The amount of steps to remove from the DataFrame's start.
        beta_threshold (int): The amount of trailing steps to remove from the DataFrame.
    """

    def __init__(self, key: str = "Laeuft", lower: Union[int, float] = 1, upper: Union[int, float] = 1,
                 alpha_threshold: int = 10800, beta_threshold: int = 900):
        self._key_substring = key
        self._key = ""
        self._lower = lower
        self._upper = upper
        self._alpha = alpha_threshold
        self._beta = beta_threshold
        self._update = {}

    @property
    def alpha(self) -> int:
        return self._alpha

    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {
                NormalizeFit.__name__: CallableDependency.NOT_BEFORE,
                Features.__name__: CallableDependency.NOT_BEFORE,
            }
    
    def _find_corresponding_key(self, column_names) -> bool:
        if not self._key:
            # find the key matching the substring given
            keys = [k for k in column_names if self._key_substring in k]
            if len(keys) > 0:
                logger.debug("Found %s matching key(s) in header", len(keys))
                self._key = keys[0]
                return True
            else:
                logger.debug("Can not find key %s", self._key_substring)
                return False
        
        return self._key in column_names
    
    def run(self, fm: FileMeta, context: Context) -> CALLABLE_RETURN_TYPE:
        # read and remove the dataset file, if everything is fine, it will eb created again
        df = fm.read()
        if df.empty or isinstance(df, pd.Series):
            fm.state = FileState.DELETED
            return fm
        
        # find similar key
        if not self._find_corresponding_key(column_names=df.columns.values):
            return fm

        # split the dataset into multiple smaller ones by removing values
        # smaller or greater than the boundaries
        df_splits = multi_split(df.copy(), self._key, self._lower, self._upper)

        # if the resulting amount of splits is 0, then remove the file
        if len(df_splits) == 0:
            fm.state = FileState.DELETED
            return fm

        # save all dataframes
        if df_splits[0].shape != df.shape:
            df_splits = remove_threshold(df_splits, self._alpha, self._beta)

        files = []
        base_path = fm.base_path
        for i, df in enumerate(df_splits):
            postfix = "" if i == 0 else f"{i}"
            file_name = fm.build_file_name(postfix)

            if i > 0:
                fm = FileMeta.build_and_save(file_name, df, base_path)
            else:
                fm.save(df)

            files.append(fm)

        return files

    def finish(self, context: Context, metadata: Metadata) -> None:
        # requesting the metadata as dict re-calculates the total number of recordings
        _ = metadata.as_dict()
        logger.debug("Splitting finished: length=%s", metadata.total_recordings)

        metadata.state_alpha[self._key] = {"delay": self._alpha, "upper": self._upper, "lower": self._lower}


