from datetime import datetime
import gzip

from tqdm import tqdm
import pandas as pd

from Dataflux.helper import get_day_list, compress, decompress, get_sensor_name_to_uuid
from Dataflux.data.download import DataDownloader
from Dataflux.data.s3io import S3Bucket, DataType


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

        score_uuids = get_sensor_name_to_uuid(self._uuid)

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
            df.drop(columns=score_uuids.values(), errors="ignore", inplace=True)
            df = decompress(df.reset_index())

            # 2. modify the dataframe with the callable
            # round to 3 decimals, as the front-end can not show more
            result_df = self._callable(ts_1, ts_2)
            result_df = result_df.round(3)

            # 3. rename human readable column names to be machine readable
            names = {name: score_uuids.get(name, name) for name in list(result_df.columns.values)}
            result_df.rename(columns=names, inplace=True)

            # 4. merge, then apply the same compressing rules 
            df = pd.merge(result_df, df, how="outer", left_index=True, right_index=True)
            df = compress(df.reset_index())

            # 5. sort the columns by uuid-names
            df = df.reindex(sorted(df.columns), axis=1)
            df.reset_index(inplace=True)

            # 6. export dataframe as tsv-file
            tsv = df.to_csv(sep="\t")
            tsv_bytes = gzip.compress(tsv.encode("utf-8"))
            day = day.strftime("%Y/%m/%d")
            path = f"{self._uuid}/days/{day}/{self._uuid}-virtual-debug.tsv"

            # 7. upload dataframe to s3
            self._client.put_object(Bucket=self._bucket, Key=f"{path}", Body=tsv_bytes,
                ContentType="text/tab-separated-values", ContentEncoding="gzip")
