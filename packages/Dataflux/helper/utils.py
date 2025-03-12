"""This module contains utilities for handling downloaded data from the s3 bucket."""

import importlib
import os
import uuid
from collections.abc import Callable, Iterator
from copy import copy
from datetime import date, datetime, timedelta
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from Dataflux import logger
from Dataflux.diskio.meta import FileMeta, Metadata

rcParams.update({"figure.autolayout": True})


def fullname(module):
    return str(module).split("'")[1]


def is_valid_uuid(value: str):
    """Returns true if the given string is a valid UUID, otherwise this function returns false."""
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def class_for_name(module_name: str, class_name: str):
    """This function finds the class given the module and class name

    Args:
        module_name (str): The module path where the class is located.
        class_name (str): The name of the class.
    """
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def get_file(
    file_path: str,
    create_dirs: bool = True,
    overwrite: bool = False,
    postfix: str = "",
) -> str:
    """This function checks if a file_path is valid and creates directories if required.
    In case the file_path already exists, the file name receives an increasing postfix.

    Args:
        file_path (str): The file path of the file to be downloaded
        create_dirs (bool, optional): Whether directories should be created if
        required. Defaults to True.
        overwrite (bool, optional): Whether existing files should be
        overwritten. Defaults to False.
        post_fix (str, optional): The postfix to be added (does not replace the
        increasing postfix-number)
    """
    assert file_path.count(".") == 1, (
        "File requires an extension (format: <file-name>.<extension>)"
    )
    # create folder of file if required
    folder_path = os.path.dirname(file_path)
    folder_exists = os.path.exists(folder_path)
    if create_dirs and not folder_exists:
        os.makedirs(folder_path)

    # create file name: add postfix, increase postfix-number (optional)
    file_name_parts = os.path.basename(file_path).split(".")
    file_name = file_name_parts[0]
    file_extension = file_name_parts[-1]
    postfix = f"_{postfix}" if postfix else ""
    name_postfix = f"{file_name}{postfix}.{file_extension}"

    # do not check if the file exists in case of overwrite
    if overwrite:
        return os.path.join(folder_path, name_postfix)

    # check if file exists and increase postfix-number until it does not exist anymore
    num = 1
    while os.path.exists(os.path.join(folder_path, name_postfix)):
        name_postfix = f"{file_name}{postfix}_{num}.{file_extension}"
        num += 1

    return os.path.join(folder_path, name_postfix)


def line_str_to_array(line: str) -> np.ndarray:
    """Multiprocesses a line-str, splits into a list and converts to float.

    Args:
        line (str): The string to convert.

    Returns:
        np.array: The converted float32 array.
    """
    row = line.strip("\n").split("\t")
    row = pd.to_numeric(row, errors="coerce")
    return row


def to_float(value: str) -> float:
    """Safe converts a string value into a float value. If the value cannot be converted, it
    is returned as NaN.

    Args:
        value (str): The value that should be converted into a float.

    Returns:
        float: The converted float value or NaN if the conversion failed.
    """
    try:
        return float(value)
    except ValueError as _:
        return np.nan


def simple_graph(data_x: List[str], data_y: List[float]) -> Callable:
    """Creates a simple graph from the given data points.

    Args:
        data_x (List[str]): The x-axis values for each data point
        data_y (List[float]): The y-axis values for each data point

    Returns:
        Callable: A callable that can be used to build the graph.
    """

    def graph(title: str, xlabel: str = "None", ylabel: str = "None"):
        _, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(title)
        ax.plot(data_x, data_y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if len(data_x) == 0:
            return ax

        # limit label ticks
        x_locs = np.arange(0, len(data_x), max(len(data_x) // 2, len(data_x) // 50))
        x_labels = [data_x[i] for i in x_locs]
        ax.set_xticks(x_locs)
        ax.set_xticklabels(x_labels)
        return ax

    return graph


def parse_datestring(date_str: str, force_utc: bool = True) -> datetime:
    """This function parses a given date into a python datetime object. On default, this method forces the output date
    to be in the UTC format. This means, a given date string will be in UTC even when not explicitly stated.

    Args:
        date_str (str): The date string e.g. '2025-01-22' or '2025-01-22T00:00:00Z'.
        force_utc (bool): Whether to force the output to be in UTC or not. Defaults to true.
    """
    is_utc = date_str[-1].upper() == "Z"
    date_obj = datetime.fromisoformat(date_str)
    if not force_utc:
        return date_obj

    if not is_utc:
        return datetime.fromisoformat(date_str + "Z")

    return date_obj


def daterange(
    start_date: datetime, end_date: datetime
) -> Iterator[Tuple[datetime, datetime]]:
    """This function iterates over the start and end dates and returns the intervals with a timedelta of 1 day.

    Args:
        start_date: The start date.
        end_date: The end date.

    Returns:
        Iterator[Tuple[datetime, datetime]]: An iterator returning tuples of the start time and end time.

    """
    curr_day = start_date
    end_of_day = curr_day.replace(
        hour=23, minute=59, second=59, microsecond=999999
    )  # set time part to 23:59:59

    while end_of_day <= end_date:
        yield curr_day, end_of_day
        curr_day = curr_day.replace(
            hour=0, minute=0, second=0
        )  # set time part to 00:00:00
        curr_day += timedelta(days=1)
        end_of_day = curr_day.replace(hour=23, minute=59, second=59, microsecond=999999)

    if curr_day < end_date:  # handle the last day if it's not a full day
        yield curr_day, min(end_date, end_of_day)


def get_day_list(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Get a list of days from an interval (tuple) of days (inclusive).

    Args:
        start_date: The start date.
        end_date: The end date.

    Returns:
        List[date]: A list of dates within the given interval.
    """
    tz = start_date.tzinfo

    # Initialize an empty list to store the days
    days = [start_date]

    # Loop through each day in the interval
    start_date = datetime(start_date.year, start_date.month, start_date.day, tzinfo=tz)
    end_ = datetime(end_date.year, end_date.month, end_date.day, tzinfo=tz)
    for i in range(1, (end_ - start_date).days):
        # Calculate the date by adding the timedelta to the start date
        day = start_date + timedelta(days=i)
        # Add the day to the list
        days.append(day)

    days.append(end_date)

    # Return the list of days
    return days


def get_week_of_year(day_in_week: date | datetime):
    """
    Get the week number (ISO week number) for a given date or datetime.

    Args:
        day_in_week (date | datetime): The date or datetime to get the week number from.

    Returns:
        int: The ISO week number.
    """

    # Check if the input is of type date or datetime
    if isinstance(day_in_week, date):
        date_obj = day_in_week
    elif isinstance(day_in_week, datetime):
        date_obj = day_in_week.date()
    else:
        raise TypeError("Parameter was not of type date or datetime")

    # Return the week number from the ISO calendar
    return date_obj.isocalendar()[1]


def deterministic_split_algorithm(
    meta: Metadata, percentage: float
) -> Tuple[List, List]:
    """
    This function splits the days of a dataset's metadata into two lists. The first list contains
    the days which make up `1 - percentage` of total recordings. The second list contains the remaining days.

    Args:
        meta (Metadata): Metadata object containing the days.
        percentage (float): The percentage of days that should be in the split list.

    Returns:
        Tuple[List]: A tuple containing two lists. The first list contains the first `1 - percentage`
            and the second list contains the remaining days.
    """
    # requesting the metadata as dict recalculates the total number of recordings
    _ = meta.as_dict()

    # get the days in a sorted manner
    days: List[FileMeta] = [v for v in meta.files.values()]
    days = sorted(days, key=lambda fm: fm.samples)
    total_recordings = meta.total_recordings

    # copy the list of days as the algorithm will remove files from one list and
    # add them to another list
    trimmed_days = copy(days)
    target_recordings = int(total_recordings * percentage)

    # add days to the list until the target_recordings is reached
    selected_days: List[FileMeta] = []
    iterative_recordings = 0
    for i in range(len(days)):
        if iterative_recordings >= target_recordings:
            break

        # also, remove the day from the list of days to be trimmed
        fm: FileMeta = trimmed_days.pop(i - len(selected_days))
        selected_days.append(fm)
        iterative_recordings += fm.samples

    # as the list of days is sorted, we can attempt to remove some days from the selected
    # list and put them back to the trimmed list
    while iterative_recordings - selected_days[0].samples > target_recordings:
        fm = selected_days.pop(0)
        iterative_recordings -= fm.samples
        trimmed_days.append(fm)

    # log some data
    actual_percentage = round(iterative_recordings / total_recordings * 100, 2)
    logger.debug(
        "Targeted %s%% and settled on %s%%", percentage * 100, actual_percentage
    )
    logger.debug(
        "Training set has %s days and validation set has %s",
        len(trimmed_days),
        len(selected_days),
    )

    return trimmed_days, selected_days
