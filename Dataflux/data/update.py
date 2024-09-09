from datetime import datetime
import hashlib

import numpy as np
import pandas as pd

from Dataflux.helper import get_day_list
from Dataflux.data.download import DataDownloader
from Dataflux.data.s3io import S3Bucket


def ts_to_datetime(ts: int) -> str:
    """
    This function converts a time step integer value to the datetime as string in the
    format YYYY-MM-DD.

    Args:
        ts (int): time step integer value.
    
    Returns:
        str: datetime as string in the format YYYY-MM-DD.
    """
    # transforms the time step integer value to the datetime as string
    # in the format YYYY-MM-DD
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')


def compress(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function compresses the time series data by removing duplicates in a safe way.
    If a value is repeated in the time series, it is replaced by np.NAN until there is
    a change in the value. When there was a change, the previous value is used instead of
    np.NAN to declare the change in the value. E.g.:

    - Input `[1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 5, 4, 6]`
    - Intermediate `[1, N/A, N/A, 1, 3, N/A, N/A, N/A, 3, 4, 5, 4, 6]`
    - Output `[1, 1, 3, 3, 4, 5, 4, 6]`

    Args:
        data (pd.DataFrame): The time series data to compress.
    
    Returns:
        pd.DataFrame: Compressed time series data.
    """
    names, data = data.columns.values, data.to_numpy()

    # assume the data is a function and we want to find the changes of the first 
    # derivative. An approximation is f'(x) = f(x) - f(x - 1)
    compressed = (data[1:, :] - data[:-1, :])
    compressed = (data[2:, :] - data[:-2, :])

    # if the first derivative is NOT zero, then there is a change and we are interested
    # in the value. Set these value to 1 and others to 0. Finally, multiply the array
    # of 0s and 1s with the original data array, to have the original values at the position
    # where a value change happens.
    compressed[compressed != 0] = 1
    compressed[compressed == 0] = np.nan
    compressed = compressed * data[1:-1, :]

    # due to our basic approximation of the first derivative, the first and last value is missing.
    # As per definition of this compression method, both values can be added as usual.
    compressed = np.insert(compressed, 0, data[0, :], axis=0)
    compressed = np.insert(compressed, len(compressed), data[-1, :], axis=0)

    # to further compress the data, remove all rows, where each value is np.NAN. This
    # data is not valuable and can be removed.
    compressed = pd.DataFrame(compressed, columns=names)
    return compressed.dropna(how="all")


class UpdateConsistencyError(Exception):
    """
    This class defines the error which occurs, when a TSV-file was changed at positions
    which should not be touched.
    """
    def __init__(self):
        super().__init__("Update is not consistent, only new columns should be added.")


class UpdateCallable:
    """
    This class defines the callable which is executed by the DataUpdater. This should 
    generate a series (or a list of series) to be added to the virtual sensor list.
    """
    def get_update(self, start: int, end: int) -> pd.DataFrame:
        """
        This method generates the update for the given timeframe.

        Args:
            start (int): The start time steps of the timeframe to be updated.
            end (int): The end time steps of the timeframe to be updated.

        Returns:
            pd.DataFrame: The updated data.
        """
        raise NotImplementedError()

    def __call__(self) -> pd.DataFrame:
        return self.modify()


class DataUpdater(S3Bucket):
    """
    This class updates a TSV in the S3Bucket with newly genereated data. To generate the data,
    an `UpdateCallable` is required, which generates the series to add for a given time frame.

    Args:
        machine_uuid (str): The UUID of the machine to be updated.
        update_callable (UpdateCallable): The callable which generates the data to be added.
    """
    def __init__(self, machine_uuid: str, update_callable: UpdateCallable) -> None:
        super().__init__(machine_uuid)
        self._callable = update_callable

    def _hash(self, df: pd.DataFrame) -> int:
        return int(hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest(), 16)

    def update(self, start_ts: int, end_ts: int) -> None:
        """
        This method updates the data in the S3 for the given time frame. This method downloads the
        data from the S3 and then calls the callable to generate the new data. The newly generated
        data is compressed, added to the downloaded TSV-file and uploaded again.

        Args:
            start (int): The start time steps of the timeframe to be updated.
            end (int): The end time steps of the timeframe to be updated.
        """
        # get start and end days
        start = ts_to_datetime(start_ts)
        end = ts_to_datetime(end_ts)
        days = get_day_list((start, end))

        # create the S3 bucket connection and download the files for each day
        downloader = DataDownloader(self._uuid, file_postfix="virtual")
        for day in days:
            # 1. download the DataFrame
            df = downloader.get_dataframe(day)

            # 3. Before modifying, store the current hash
            original_columns = df.columns.values
            original_hash = self._hash(df)

            # 4. modify the dataframe with the callable
            result_df = self._callable(start_ts, end_ts)
            df = pd.merge(df, result_df, how="cross", left_index=True, right_index=True)

            # 5. check the hash of the new dataframe and the original one.
            # the numbers should be identical for the untouched parts of the dataframes.
            new_hash = self._hash(df[original_columns])
            if new_hash != original_hash:
                raise UpdateConsistencyError()

            # 6. sort the columns by uuid-name
            df = df.reindex(sorted(df.columns), axis=1)

            # 7. apply the same compressing rules 
            df = compress(df)

            # 8. export dataframe as tsv-file
            tsv = df.to_csv(sep="\t")
            path = f"{self._uuid}/days/{day}/{self._uuid}.tsv"
            # self._client.set_object(Bucket=self._bucket, Key=f"{path}")["Body"].write()
