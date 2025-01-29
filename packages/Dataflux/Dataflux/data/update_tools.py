"""This module implements tools supporting the TSV file upload process."""
from typing import List, Tuple

import pandas as pd

from Dataflux.data.download import DataDownloader
from Dataflux.data.types import DataType


def safe_df_avg(df: pd.DataFrame, avg_size: int, delta: int) -> pd.DataFrame:
    # apply the mean as a rolling window with the step size being the window's size
    # the window's size is calculated by the time delta between two datapoints and
    # the resolution to be achieved
    pool = avg_size // delta
    df = df.rolling(window=pool, step=pool, center=True).mean()
    df = df.dropna(how="all")
    df = df.round(3)

    return df


def merge_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """This function merges two DataFrames into one. The method performs an outer merge of `df2` on `df1`, then it
    overwrites overlapping values with a high priority on values from `df2`.

    Examples:
        After the outer merge, the DataFrame could look like:

            ts  A_x A_y
            0	0.0	NaN
            1	1.0	NaN
            2	2.0	3.0
            3	3.0	NaN
            4	4.0	5.0
            6	NaN	7.0
            8	NaN	9.0

        Then, this function performs a cleanup:

            ts  A
            0	0.0
            1	1.0
            2	3.0
            3	3.0
            4	5.0
            6	7.0
            8	9.0

    Args:
        df1 (DataFrame): The first DataFrame.
        df2 (DataFrame): The second DataFrame which will be merged into the first one.

    Returns:
        DataFrame: The merged DataFrame.
    """
    # ensure the correct index type
    for df in [df1, df2]:
        if df is None or df.empty:
            continue

        df = df.reindex(sorted(df.columns), axis=1)
        df.reset_index(inplace=True)
        df = df.astype({"ts": "int64"})
        df.set_index("ts", inplace=True)

    if df1.empty:
        return df2

    if df2.empty:
        return df1

    # duplicate columns will be labeled with an "_x" and "_y" where "_y" contains the update
    df = pd.merge(df1, df2, how="outer", left_index=True, right_index=True)

    # always use the incoming values and overwrite old ones
    # NaN is always replaced by a non-NaN value
    for name in df.columns:
        if name.endswith("_y"):
            other = name[:-1] + "x"

            df[name[:-2]] = df[name].fillna(df[other])
            df = df.drop(columns=[name, other])

    return df


def get_downloaders(uuid: str) -> List[Tuple[DataDownloader, int]]:
    """This method initializes downloaders for al virtual tsv files.

    Returns:
        Tuple[str, List[DataDownloader]]: The uuid from which the data will be downloaded from and a list of
            DataDownloader classes. Each class downloads 'virtual' TSV-files with the appendix
            '', '-average', '-average-10m'.
    """
    appendix = [("", 1), ("average", 120), ("average-10m", 600), ("ai", 1), ("ai-average", 120), ("ai-average-10m", 600)]

    # generate downloaders for file *-virtual.tsv(-average, -average-10m)
    output = []
    for key, seconds in appendix:
        downloader = DataDownloader(uuid, file_postfix="virtual", appendix=key)
        downloader.set_data_type(DataType.MEASUREMENTS)

        output.append((downloader, seconds))

    return output


def safe_download(downloader: DataDownloader, day: str) -> pd.DataFrame:
    df = downloader.get_dataframe(day)
    if not df.empty:
        df.set_index("ts", inplace=True)

    return df


class UpdateCallable:
    """
    This class defines the callable which is executed by the DataUpdater. This should
    generate a series (or a list of series) to be added to the virtual sensor list.
    """

    def get_update(self, start: int, end: int, total_key: str) -> pd.DataFrame:
        """
        This method generates the update for the given timeframe.

        Args:
            start (int): The start time steps of the timeframe to be updated.
            end (int): The end time steps of the timeframe to be updated.
            total_key (str): The key of the column showing the max over all sensor values.

        Returns:
            pd.DataFrame: The updated data.
        """
        raise NotImplementedError()

    def __call__(self, start: int, end: int, total_key: str) -> pd.DataFrame:
        return self.get_update(start, end, total_key)
