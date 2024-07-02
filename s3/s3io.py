import os
from datetime import date, timedelta

from typing import Union, TextIO, Generator
from io import BytesIO
from uuid import UUID

import boto3
import gzip

from tqdm import tqdm

from .tsvio import TsvIO, logger
from .utils import get_day_list

s3 = boto3.client("s3")

S3_BUCKET = os.getenv('S3_BUCKET', 'xplore-measurements')


def try_retrieve_tsv_stream_s3(
        s3_bucket: str, s3_file_path: str
) -> Union[TextIO, None]:
    logger.debug(f"Retrieving '{s3_bucket}:{s3_file_path}' from S3")
    try:
        data = s3.get_object(Bucket=s3_bucket, Key=f"{s3_file_path}")["Body"].read()
        stream: TextIO = gzip.open(BytesIO(data), mode="rt")
        stream.seek(0)
        return stream

    except Exception as e:
        logger.debug(
            f"Could not receive tsv stream because of exception {type(e)}: {e}"
        )
        return None


def fetch_interval(machine_uuid: str | UUID, interval: tuple[date, date],
                   *, s3_bucket: str = None, file_ending: str = "tsv", progress_bar: bool = True,
                   ) -> Generator[TextIO | None, None, None,]:
    """
    A generator which fetches the measurements data in the tsv_streams for the given machine in the given interval.
    This function has a resolution of days, as it currently also only supports aggregation of day data.

    :param machine_uuid: The uuid of the machine to aggregate the data for.
    :param interval: The interval over which to aggregate.
    :param s3_bucket: The s3_bucket to use. Defaults to the env var S3_BUCKET.
    :param file_ending: The file ending to use. Current options are ["tsv-raw", "tsv", "tsv-average"]
    :param progress_bar: Whether to show the tqdm progress bar. Defaults to True.
    :return Yields the TSVReaders (or None if tsv was not found) ordered by day.
    """
    s3_bucket = s3_bucket if s3_bucket else S3_BUCKET
    machine_uuid = str(machine_uuid)

    # Collect data
    logger.info(f"Fetching data for machine {machine_uuid} in interval"
                f" {interval[0].strftime('%Y-%m-%d')} to {interval[1].strftime('%Y-%m-%d')}")
    days = get_day_list(interval)
    for i_date in tqdm(days,
                       desc=f"Fetching TSV daily data for machine {machine_uuid}",
                       disable=not progress_bar
                       ):
        path = f'{machine_uuid}/days/{i_date.strftime("%Y/%m/%d")}/{machine_uuid}.{file_ending}'
        try:
            tsv_stream = try_retrieve_tsv_stream_s3(s3_bucket, path)
            yield tsv_stream
        except Exception as exc:
            logger.info(f'Could not find data for day {i_date} for machine {machine_uuid} under path {path}')
            logger.debug(f's3 failed to retrieve file with the following exception: {type(exc)} {exc}')
            yield None
            continue


def fetch_latest_header(company: str, machine_uuid: str | UUID, interval: tuple[date, date] | date, *,
                        s3_bucket: str = None) -> list[str]:
    """
    Fetches the latest header (list of columns) for the given interval.
    :param company: The company of the machine.
    :param machine_uuid: The uuid of the machine to fetch the latest header for.
    :param interval: The interval to fetch the latest header for (simply uses the end date).
                        Can also be set to a specific date.
    :param s3_bucket: The s3_bucket to use. Defaults to the env var S3_BUCKET.
    :return: The latest header.
    """
    # TODO: This could also be done using SQL and the DB
    s3_bucket = s3_bucket if s3_bucket else S3_BUCKET
    machine_uuid = str(machine_uuid)

    if isinstance(interval, date):
        interval = (interval, interval)
    start_date, end_date = interval
    last_tsv = None
    # Ensure we find the latest header possible
    while last_tsv is None:
        last_tsv = next(fetch_interval(machine_uuid, (end_date, end_date),
                                       s3_bucket=s3_bucket, file_ending="tsv", progress_bar=False))
        # Subtract day from end_date
        end_date = end_date - timedelta(days=1)
        if end_date < start_date:
            raise ValueError("Could not find any data for the given interval")

    header = TsvIO(last_tsv).header
    return header
