import numpy as np
import pandas as pd


def decompress(data: pd.DataFrame, copy: bool = False, fill: bool = False) -> pd.DataFrame:
    """
    This function decompresses a dataframe which was compressed before. This is achieved by
    restoring each timeframe for each second and filling the added row with the values of
    the previous row.

    Args:
        data (pd.DataFrame): dataframe to decompress.
        copy (bool): Whether to copy or not the dataframe before decompressing. Decompressing
            changes the input object, as it works on the reference.
        fill (bool): Whether to fill the added row with the values of the previous
            row or not. If not, np.NANs will be set for an added row. Defaults to False.
    
    Returns:
        pd.DataFrame: decompressed dataframe.
    """
    if data.empty:
        return data

    # make sure to not modify the original dataframe, if
    # copy is true
    if copy:
        data = data.copy()

    # get the first and last timestamp value which define the new
    # index range. The step is 1 second for each data point
    t0 = data["ts"].iloc[0]
    tn = data["ts"].max() + 1

    # apply the newly created index to the dataframe, added rows will
    # have NAN values
    new_index = pd.Index(np.arange(t0, tn, 1), name="ts")
    data = data.reindex(new_index)

    # decompress the data by forward filling NAN values
    if fill:
        data = data.ffill()
    return data


def compress(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function compresses the time series data by removing duplicates in a safe way.
    If a value is repeated in the time series, it is replaced by np.NAN until there is
    a change in the value. When there was a change, the previous value is used instead of
    np.NAN to declare the change in the value. E.g.:

    - Input `[1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 5, 4, 6]`
    - Intermediate `[1, N/A, N/A, 1, 3, N/A, N/A, N/A, 3, 4, 5, 4, 6]`
    - Output `[1, 1, 3, 3, 4, 5, 4, 6]`

    Args:
        dataframe (pd.DataFrame): The time series data to compress.
    
    Returns:
        pd.DataFrame: Compressed time series data.
    """
    if dataframe.empty:
        return dataframe

    index = dataframe.index
    names = dataframe.columns.values
    data = dataframe.to_numpy()

    # due to our basic approximation of the first derivative, the first and last value is missing.
    # To handle this, simply pad a 0 to the start and end of the data
    data = np.insert(data, 0, 0, axis=0)
    data = np.insert(data, len(data), 0, axis=0)

    # assume the data is a function: we want to find the changes of the first and second
    # derivative. An approximation is f'(x) = f(x) - f(x - 1)
    deriv_1 = (data[1:, :] - data[:-1, :])
    deriv_2 = (deriv_1[1:, :] - deriv_1[:-1, :])
    interest = abs(deriv_1[:-1]) + abs(deriv_2)

    # if the first and second derivative is NOT zero, then there is a change, we are interested
    # in. Set these value to 1 and others to 0. Finally, multiply the array of 0s and 1s with
    # the original data array, to have the original values at the position
    # where a value change happens.
    interest[interest != 0] = 1
    interest[interest == 0] = np.nan
    compressed = interest * data[1:-1, :]

    # to further compress the data, remove all rows, where each value is np.NAN. This
    # data is not valuable and can be removed.
    compressed = pd.DataFrame(compressed, columns=names, index=index)
    # compressed = compressed.set_index("ts")
    return compressed.dropna(how="all")
