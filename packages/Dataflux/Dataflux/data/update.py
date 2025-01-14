from datetime import datetime, timezone, timedelta
import gzip

from tqdm import tqdm
import pandas as pd
from botocore.exceptions import ClientError

from Dataflux.helper import get_day_list, compress, decompress, get_sensors, SensorRegisterFormat, filter_upload
from Dataflux.data.download import DataDownloader
from Dataflux.data.s3io import S3Bucket, DataType

from Dataflux import logger


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


class DataUpdater(S3Bucket):
    """
    This class updates a TSV in the S3Bucket with newly generated data. To generate the data,
    an `UpdateCallable` is required, which generates the series to add for a given time frame.

    Args:
        machine_uuid (str): The UUID of the machine to be updated.
        update_callable (UpdateCallable): The callable which generates the data to be added.
    """

    def __init__(self, machine_uuid: str, update_callable: UpdateCallable, debug: bool = False) -> None:
        super().__init__(machine_uuid)
        self._callable = update_callable

        self._debug = debug
        self._score_substring = "Malfunction"
        self._sensor_format = SensorRegisterFormat.NAME_TO_UUID

    def update(self, start_ts: int, end_ts: int, time_delta: int) -> None:
        """
        This method updates the data in the S3 for the given time frame. This method downloads the
        data from the S3 and then calls the callable to generate the new data. The newly generated
        data is compressed, added to the downloaded TSV-file and uploaded again.

        Args:
            start_ts (int): The start time steps of the timeframe to be updated.
            end_ts (int): The end time steps of the timeframe to be updated.
            time_delta (int): The delta in seconds between two timestamps.
        """
        # get start and end days
        start = datetime.fromtimestamp(start_ts, timezone.utc)
        end = datetime.fromtimestamp(end_ts, timezone.utc)
        days = get_day_list((start, end))

        sensor_filter = filter_upload(sensor_format=self._sensor_format, include=self._score_substring)
        score_uuids = get_sensors(self._uuid, self._sensor_format, sensor_filter, debug=self._debug)
        if not score_uuids:
            logger.error("No sensors to upload to, is '%s' missing in the sensor names?", self._score_substring)
            return None

        # create the S3 bucket connection and download the files for each day
        downloader = DataDownloader(self._uuid, file_postfix="virtual")
        downloader.set_data_type(DataType.MEASUREMENTS)

        logger.debug(f"Updating TSV-files for {days}")
        for day in tqdm(days, desc="Score to Virtual Sensors"):
            # 0. calculate the correct time steps for the day to update
            ts_1 = int(day.date().strftime("%s"))
            ts_1 = ts_1 if ts_1 > start_ts else start_ts

            ts_2 = int((day.date() + timedelta(days=1)).strftime("%s"))
            ts_2 = ts_2 if ts_2 < end_ts else end_ts

            # 1. download the DataFrame and ensure homogeneous time steps
            day_str = day.strftime("%Y-%m-%d")
            try:
                df = downloader.get_dataframe(day_str)
                if not df.empty:
                    df.set_index("ts", inplace=True)
                    df.drop(columns=score_uuids.values(), errors="ignore", inplace=True)
                    df = decompress(df)
            except ClientError as _:
                logger.debug(f"Could not find a TSV-file for day {day_str}, continue with empty DataFrame.")
                df = pd.DataFrame()

            # 2. modify the dataframe with the callable
            # round to 3 decimals, as the front-end can not show more
            result_df = self._callable(ts_1, ts_2, total_key="Total")
            result_df = result_df.round(3)

            # 3. rename human-readable column names to be machine-readable
            drop = [name for name in result_df.columns.values if
                    not score_uuids.get(name.split(" ")[0] + " (Malfunction AI) (%)", None)]
            if len(drop) > 0:
                logger.debug("Did not expect %s in header, removing from upload.", drop)
                result_df.drop(columns=drop, inplace=True)

            names = {name: score_uuids.get(name.split(" ")[0] + " (Malfunction AI) (%)", name) for name in
                     list(result_df.columns.values)}
            result_df.rename(columns=names, inplace=True)

            # 4. merge, then apply the same compressing rules 
            day = day.strftime("%Y/%m/%d")

            # 5. calculate the averages for the 2- and 10-minutes files
            for avg in [600, 120]:
                if avg < time_delta:
                    continue

                # apply the mean as a rolling window with the step size beeing the window's size
                # the window's size is calculated by the time delta between two datapoints and
                # the resolution to be achieved
                pool = avg // time_delta
                df_avg = result_df.rolling(window=pool, step=pool, center=True).mean()
                df_avg = df_avg.iloc[1:-1]
                df_avg = df_avg.round(3)

                postfix = "average" if avg == 120 else "average-10m"
                path = f"{self._uuid}/days/{day}/{self._uuid}-virtual.tsv-{postfix}"

                df_avg = pd.merge(df_avg, df, how="outer", left_index=True, right_index=True)
                self.upload(df_avg, path)

            # 6. upload the original for the 1 second data
            path = f"{self._uuid}/days/{day}/{self._uuid}-virtual.tsv"
            df = pd.merge(result_df, df, how="outer", left_index=True, right_index=True)
            self.upload(df, path)

    def upload(self, df: pd.DataFrame, path: str) -> None:
        # df = compress(df.reset_index())
        df = compress(df)
        if "ts" in df.columns:
            df.drop(columns="ts", inplace=True)

        # 1. sort the columns by uuid-names
        df = df.reindex(sorted(df.columns), axis=1)
        df.reset_index(inplace=True)
        df = df.astype({"ts": "int64"})
        df.set_index("ts", inplace=True)

        # 2. export dataframe as tsv-file
        tsv = df.to_csv(sep="\t")
        tsv_bytes = gzip.compress(tsv.encode("utf-8"))

        # 3. upload dataframe to s3
        self._client.put_object(Bucket=self._bucket, Key=f"{path}", Body=tsv_bytes,
                                ContentType="text/tab-separated-values", ContentEncoding="gzip")
