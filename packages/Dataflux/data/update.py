import gzip
from datetime import datetime
from typing import Dict

import pandas as pd
from tqdm import tqdm

from Dataflux import DEBUG_MODE, logger
from Dataflux.data.s3io import S3Bucket
from Dataflux.data.update_tools import (
    UpdateCallable,
    get_downloaders,
    merge_dfs,
    safe_df_avg,
    safe_download,
)
from Dataflux.helper import (
    SensorRegisterFormat,
    compress,
    filter_upload,
    get_sensors,
)
from Dataflux.helper.utils import daterange, is_valid_uuid


class EmptyScoreRegister(RuntimeError):
    pass


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
        self,
        machine_uuid: str,
        update_callable: UpdateCallable,
        force_upload: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(machine_uuid)
        self._callable = update_callable
        self._force = force_upload
        if force_upload and not debug:
            raise ValueError("Force upload can only be enabled if debug=True.")

        self._debug = debug
        self._debug_uuid = "5ac7f03f-f4e6-4b15-9bc6-21db798f9d25"
        self._score_substring = "Malfunction"
        self._sensor_format = SensorRegisterFormat.NAME_TO_UUID

        # get the sensor names and uuids where the scores should be located at
        sensor_filter = filter_upload(None, self._sensor_format, self._score_substring)
        self._score_uuids = get_sensors(
            machine_uuid=self._uuid,
            sensor_format=self._sensor_format,
            sensor_filter=sensor_filter,
            debug=self._debug,
        )

        if not self._score_uuids:
            logger.error(
                "No sensors to upload to, is '%s' missing in the sensor names?",
                self._score_substring,
            )
            raise EmptyScoreRegister()

    def _rename_with_register(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = []
        register = {}
        for name in df.columns.values:
            expected_name = name.split(" ")[0] + " (Malfunction AI) (%)"
            if not self._score_uuids.get(expected_name, None):
                drop.append(name)
            else:
                register[name] = self._score_uuids.get(expected_name, name)

        if len(drop) > 0:
            logger.debug("Did not expect %s in header, removing from upload.", drop)
            df.drop(columns=drop, inplace=True)

        return df.rename(columns=register)

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

        # create the S3 bucket connection and download the files for each day
        uuid = self._debug_uuid if self._debug else self._uuid
        downloaders = get_downloaders(uuid)

        interval = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        logger.debug(f"Updating TSV-files for {interval}")
        for start_date, end_date in tqdm(
            days, desc="Score to Virtual Sensors", disable=not DEBUG_MODE
        ):
            # 0. get the correct time steps for the day to update
            ts_1, ts_2 = start_date.timestamp(), end_date.timestamp()
            day_str = start_date.strftime("%Y-%m-%d")

            df = safe_download(downloaders[0][0], day_str)
            download_size = len(df)
            logger.debug("Downloaded virtual sensors (len=%s)", download_size)

            # 2. modify the dataframe with the callable
            # round to 3 decimals, as the front-end can not show more
            result_df = self._callable(ts_1, ts_2, total_key="Total")
            result_df = result_df.round(3)
            logger.debug("Virtual sensors to append (len=%s)", len(result_df))

            # 3. rename human-readable column names to be machine-readable
            result_df = self._rename_with_register(result_df)
            download_cols = {
                n: hash(df[n].dropna().sum())
                for n in df.columns.values
                if n not in result_df.columns.values
            }

            # 4. merge, then apply the same compressing rules
            df = merge_dfs(df, result_df)
            for downloader, avg in downloaders:
                if avg != 1:
                    continue

                self.upload(
                    df,
                    f"{uuid}/days/{day_str.replace('-', '/')}/{uuid}{downloader.postfix}.tsv"
                    + downloader.appendix,
                    download_size,
                    download_cols,
                )

            # 5. calculate the averages for the 2- and 10-minutes files
            for downloader, avg in downloaders:
                if avg == 1 or avg < time_delta:
                    continue

                # get the current avg file
                df = safe_download(downloader, day_str)
                download_size = len(df)
                download_cols = {
                    n: hash(df[n].dropna().sum())
                    for n in df.columns.values
                    if n not in result_df.columns.values
                }
                logger.debug(
                    "Downloaded virtual sensors (len=%s) [%ss]", download_size, avg
                )

                # calculate the avg of the scores
                df_avg = safe_df_avg(result_df, avg, time_delta)
                logger.debug(
                    "Virtual sensors to append (len=%s) [%ss]", len(df_avg), avg
                )

                # merge the downloaded and generated dfs
                df = merge_dfs(df, df_avg)

                # check and upload the updated/merged df
                path = f"{uuid}/days/{day_str.replace('-', '/')}/{uuid}{downloader.postfix}.tsv{downloader.appendix}"
                self.upload(df, path, download_size, download_cols)

    @staticmethod
    def check(
        df: pd.DataFrame, previous_size: int, previous_columns: Dict[str, int]
    ) -> bool:
        # do not get smaller
        if len(df) < previous_size:
            logger.warning("Attempted to upload smaller dataframe.")
            return False

        if len(previous_columns) > df.shape[1]:
            logger.warning("Attempted to upload less columns than were on AWS.")
            return False

        # make sure to keep previously existing columns
        columns = {
            n: df[n].dropna().sum() for n in df.columns.values if n in previous_columns
        }
        for name, val in previous_columns.items():
            if name not in columns:
                logger.warning("Attempted to remove %s when updating.", name)
                return False

            if name not in previous_columns.values() and columns[name] != val:
                logger.warning("Forbidden modification attempt in column %s", name)
                return False

        # check if the header columns contain UUIDs only, this should work as the 'ts' is the index not a column
        for name in df.columns.values:
            if not is_valid_uuid(name):
                logger.warning(
                    "Upload failed due to malformed TSV header, got invalid UUID=%s",
                    name,
                )
                return False
        return True

    def upload(
        self,
        df: pd.DataFrame,
        path: str,
        previous_size: int,
        previous_columns: Dict[str, int],
    ) -> None:
        logger.debug("Upload new virtual sensors (len=%s)", len(df))
        df = compress(df)
        if not self._force and not DataUpdater.check(df, previous_size, previous_columns):
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

        logger.debug("Finished upload to %s", path)
