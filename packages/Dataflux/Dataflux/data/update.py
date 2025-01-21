from datetime import datetime, timezone
import gzip

from tqdm import tqdm
import pandas as pd

from Dataflux.helper import (
    compress,
    get_sensors,
    SensorRegisterFormat,
    filter_upload,
)

from Dataflux.data.s3io import S3Bucket
from Dataflux.data.update_tools import (
    UpdateCallable,
    get_downloaders,
    merge_dfs,
    safe_download,
)

from Comprisk import logger, DEBUG_MODE
from Dataflux.helper.utils import is_valid_uuid, daterange


class DataUpdater(S3Bucket):
    """
    This class updates a TSV in the S3Bucket with newly generated data. To generate the data,
    an `UpdateCallable` is required, which generates the series to add for a given time frame.

    Args:
        machine_uuid (str): The UUID of the machine to be updated.
        update_callable (UpdateCallable): The callable which generates the data to be added.
        debug (bool): Enable or disable pushing values to the test device. Defaults to False.
    """

    def __init__(
        self, machine_uuid: str, update_callable: UpdateCallable, debug: bool = False
    ) -> None:
        super().__init__(machine_uuid)
        self._callable = update_callable

        self._debug = debug
        self._debug_uuid = "5ac7f03f-f4e6-4b15-9bc6-21db798f9d25"
        self._score_substring = "Malfunction"
        self._sensor_format = SensorRegisterFormat.NAME_TO_UUID

    def update(self, start: datetime, end: datetime, time_delta: int) -> None:
        """
        This method updates the data in the S3 for the given time frame. This method downloads the
        data from the S3 and then calls the callable to generate the new data. The newly generated
        data is compressed, added to the downloaded TSV-file and uploaded again.

        Args:
            start (datetime): The start time of the timeframe to be updated.
            end (datetime): The end time of the timeframe to be updated.
            time_delta (int): The delta in seconds between two timestamps.
        """
        days = [dt for dt in daterange(start, end)]
        sensor_filter = filter_upload(
            sensor_format=self._sensor_format, include=self._score_substring
        )
        score_uuids = get_sensors(
            self._uuid, self._sensor_format, sensor_filter, debug=self._debug
        )
        if not score_uuids:
            logger.error(
                "No sensors to upload to, is '%s' missing in the sensor names?",
                self._score_substring,
            )
            return None

        # create the S3 bucket connection and download the files for each day
        uuid = self._debug_uuid if self._debug else self._uuid
        downloaders = get_downloaders(uuid)

        logger.debug(f"Updating TSV-files for {days}")
        for start_date, end_date in tqdm(
            days, desc="Score to Virtual Sensors", disable=not DEBUG_MODE
        ):
            # 0. get the correct time steps for the day to update
            ts_1 = start_date.timestamp()
            ts_2 = end_date.timestamp()

            # 1. download the DataFrame and decompress it
            day_str = start_date.strftime("%Y-%m-%d")
            df = safe_download(downloaders[0], day_str)

            # 2. modify the dataframe with the callable
            # round to 3 decimals, as the front-end can not show more
            logger.debug(
                "Get scores of time frame %s to %s",
                start_date.strftime("%Y-%m-%d %H:%M:%S"),
                end_date.strftime("%Y-%m-%d %H:%M:%S"),
            )
            result_df = self._callable(ts_1, ts_2, total_key="Total")
            result_df = result_df.round(3)

            # 3. rename human-readable column names to be machine-readable
            drop = [
                name
                for name in result_df.columns.values
                if not score_uuids.get(
                    name.split(" ")[0] + " (Malfunction AI) (%)", None
                )
            ]
            if len(drop) > 0:
                logger.debug("Did not expect %s in header, removing from upload.", drop)
                result_df.drop(columns=drop, inplace=True)

            names = {
                name: score_uuids.get(
                    name.split(" ")[0] + " (Malfunction AI) (%)", name
                )
                for name in result_df.columns.values
            }
            result_df.rename(columns=names, inplace=True)

            # 4. merge, then apply the same compressing rules
            day = start_date.strftime("%Y/%m/%d")
            path = f"{uuid}/days/{day}/{uuid}-virtual.tsv"

            df = merge_dfs(df, result_df)
            self.upload(df, path)
            logger.debug(
                "Upload starting at %s", datetime.fromtimestamp(ts_1, timezone.utc)
            )
            logger.debug("Upload until %s", datetime.fromtimestamp(ts_2, timezone.utc))

            # 5. calculate the averages for the 2- and 10-minutes files
            for i, avg in enumerate([120, 600]):
                if avg < time_delta:
                    continue

                # get the current avg file
                df = safe_download(downloaders[i + 1], day_str)

                # apply the mean as a rolling window with the step size being the window's size
                # the window's size is calculated by the time delta between two datapoints and
                # the resolution to be achieved
                pool = avg // time_delta
                df_avg = result_df.rolling(window=pool, step=pool, center=True).mean()
                df_avg = df_avg.iloc[1:-1]
                df_avg = df_avg.round(3)

                postfix = "average" if avg == 120 else "average-10m"
                path = f"{uuid}/days/{day}/{uuid}-virtual.tsv-{postfix}"

                df = merge_dfs(df, df_avg)
                self.upload(df, path)

    def upload(self, df: pd.DataFrame, path: str) -> None:
        # df = compress(df.reset_index())
        df = compress(df)
        if "ts" in df.columns:
            df.drop(columns="ts", inplace=True)

        # sort the columns by uuid-names
        df = df.reindex(sorted(df.columns), axis=1)
        df.reset_index(inplace=True)
        df = df.astype({"ts": "int64"})
        df.set_index("ts", inplace=True)

        # check if the df is empty for some reason
        if df.empty:
            logger.warning("Attempted to upload empty dataframe.")
            return None

        # check if the header columns contain UUIDs only
        for name in df.columns.values:
            if not is_valid_uuid(name):
                logger.warning(
                    "Upload failed due to malformed TSV header, got invalid UUID=%s",
                    name,
                )
                return None

        # export dataframe as tsv-file
        df = df.round(3)
        tsv = df.to_csv(sep="\t")
        tsv_bytes = gzip.compress(tsv.encode("utf-8"))

        # upload dataframe to s3
        self._client.put_object(
            Bucket=self._bucket,
            Key=f"{path}",
            Body=tsv_bytes,
            ContentType="text/tab-separated-values",
            ContentEncoding="gzip",
        )
