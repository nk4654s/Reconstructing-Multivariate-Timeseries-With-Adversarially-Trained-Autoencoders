from typing import List, Tuple

import numpy as np
from pandas import DataFrame
from scipy import stats


def trim(df: DataFrame, key: str, value: int | float, lower: bool = True) -> DataFrame:
    """
    This method removes all indices at the start of the given dataframes,
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
        start_idx = np.argmax(running >= value)
        return df.iloc[start_idx:]

    start_idx = np.argmax(running <= value)
    return df.iloc[start_idx:]


def split(
    df: DataFrame, key: str, value: int | float, lower: bool = True
) -> Tuple[DataFrame, DataFrame]:
    """This method splits a DataFrame into two DataFrames where the `state_on`
    variable matches those of the given key.

    Args:
        df (DataFrame): The DataFrame to be splitted.
        key (str): The column name which defines the split.
        value (int | float): The value above/below which the series is splitted.
        lower (bool): Remove values that are lower than the provided value. Defaults to
            True.

    Returns:
        Tuple[DataFrame, DataFrame]: Returns the splitted DataFrames, where the second one can be
            empty, if there was no split possible.
    """
    if df.empty:
        return df, df

    running = df[key].to_numpy()
    lower_vals = running >= value
    upper_vals = running <= value

    if lower:
        min_idx = np.argmin(lower_vals)
        if lower_vals.all():
            end_idx = running.shape[0] - 1

        else:
            end_idx = min_idx
    else:
        min_idx = np.argmin(upper_vals)
        if upper_vals.all():
            end_idx = running.shape[0] - 1

        else:
            end_idx = min_idx

    # if the end index is 0, return two empty dataset.
    if not end_idx:
        return DataFrame({key: []}), DataFrame({key: []})

    # if the end_index equals the full sequence length, then there is no split and the second DataFrame must be empty
    if end_idx == len(running) - 1:
        return df, DataFrame({key: []})

    return df.iloc[:end_idx], df.iloc[end_idx:]


def multi_split(
    df: DataFrame, key: str, lower: int | float, upper: int | float
) -> List[DataFrame]:
    """This method splits a DataFrame into a List of DataFrames where the `state_on`
    variable matches those of the given key.

    Args:
        df (DataFrame): The DataFrame to be splitted.
        key (str): The column name which defines the split.
        lower (int | float): The value below which the series is splitted.
        upper (int | float): The value above which the series is splitted.
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


def remove_threshold(
    df: DataFrame | List[DataFrame], alpha: int, beta: int = 0
) -> List[DataFrame]:
    """This method removes a given amount of timesteps from the start and end of a DataFrame.

    Args:
        df (DataFrame | List[DataFrame]): The DataFrame to delete some timesteps from.
        alpha (int): The amount of steps to remove from the DataFrame's start.
        beta (int): The amount of trailing steps to remove from the DataFrame.

    Returns:
        List [DataFrame]: Returns the list of modified DataFrames.
    """
    df = df if isinstance(df, List) else [df]
    reduced = []

    # remove the first x datapoints of all dataframes
    if len(df) > 0:
        for i in range(0, len(df)):
            if len(df[i]) > alpha + beta:
                df[i] = df[i].iloc[alpha : len(df[i]) - beta]
                reduced.append(df[i])
    return reduced


def remove_outlier(df: DataFrame) -> DataFrame:
    indices = []
    data = np.random.normal(loc=0, scale=1, size=100)
    for key in df.columns.values:
        series = df[key].to_numpy()

        # get boxplot length of data series
        q75, q25 = np.percentile(series, [75, 25])
        box_length = q75 - q25

        # calculate the outlier boundaries of the boxplot
        lower = q25 - (box_length * 1.5)
        upper = q75 + (box_length * 1.5)

        # calculate the histogram of the series and compare values to
        # a sample normal distribution
        data = np.random.normal(loc=series.mean(), scale=1, size=100)
        _, bin_edges = np.histogram(series, bins=50, density=True)
        kernel_density = stats.gaussian_kde(data)

        # based on the previous math, calculate the lower and upper boundaries
        # to remove outliers. This should make sure to only delete outliers and
        # not delete based on quantiles.
        lower_cut_off = kernel_density.integrate_box_1d(bin_edges[0], upper)
        upper_cut_off = kernel_density.integrate_box_1d(lower, bin_edges[-1])
        lower = lower - lower_cut_off
        upper = upper + upper_cut_off

        # find indices to remove
        indices += list(np.argwhere(series < lower).flatten())
        indices += list(np.argwhere(series > upper).flatten())

    # remove from the dataset, reset the index first to also remove "ts" values
    df.reset_index(inplace=True)
    df.drop(index=list(set(indices)), inplace=True)
    df.drop(columns=["index"], inplace=True, errors="ignore")

    return df
