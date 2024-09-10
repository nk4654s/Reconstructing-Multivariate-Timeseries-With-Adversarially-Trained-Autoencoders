from datetime import datetime

from tqdm import tqdm
import numpy as np
import pandas as pd

from Dataflux.helper import get_day_list
from Dataflux.data.download import DataDownloader
from Dataflux.data.s3io import S3Bucket, DataType


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

    def __call__(self, start: int, end: int) -> pd.DataFrame:
        return self.get_update(start, end)


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
        start = datetime.fromtimestamp(start_ts)
        end = datetime.fromtimestamp(end_ts)
        days = get_day_list((start, end))

        # create the S3 bucket connection and download the files for each day
        downloader = DataDownloader(self._uuid, file_postfix="virtual")
        downloader.set_data_type(DataType.MEASUREMENTS)
        for day in tqdm(days, desc="Score to Virtual Sensors"):
            # 0. calculate the correct time steps for the day to update
            ts_1 = day.timestamp()
            ts_1 = ts_1 if ts_1 > start_ts else start_ts

            ts_2 = day.timestamp() + 86400
            ts_2 = ts_2 if ts_2 < end_ts else end_ts

            # 1. download the DataFrame and ensure homogeneous time steps
            df = downloader.get_dataframe(day)
            new_index = pd.Index(np.arange(df["ts"].iloc[0], df["ts"].iloc[-1], 1), name="ts")
            df = df.set_index("ts").reindex(new_index).reset_index("ts")
            df["ts"] = df["ts"].astype(np.float64)

            # 2. modify the dataframe with the callable
            result_df = self._callable(ts_1, ts_2).reset_index()
            result_df["ts"] = result_df["ts"].astype(np.float64)
            df = pd.merge(result_df, df, how="outer", on="ts")
            df = df.ffill()

            # 3. round to 3 decimals, as the front-end can not show more, then apply the same compressing rules 
            df = df.round(3)
            df = compress(df)

            # 4. sort the columns by uuid-name
            df = df.set_index("ts")
            df = df.dropna(how="all")
            df = df.reindex(sorted(df.columns), axis=1)
            df = df.reset_index()

            # 5. export dataframe as tsv-file
            tsv = df.to_csv(sep="\t")
            path = f"{self._uuid}/days/{day}/{self._uuid}.tsv"
            # self._client.set_object(Bucket=self._bucket, Key=f"{path}")["Body"].write()
