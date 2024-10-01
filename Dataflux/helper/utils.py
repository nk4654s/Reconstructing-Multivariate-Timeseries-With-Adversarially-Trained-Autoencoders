"""This module contains utilities for handling downloaded data from the s3 bucket."""

import importlib
import os
from typing import Callable, Tuple, List
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})


def fullname(module):
    return str(module).split("'")[1]


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
    assert (
        file_path.count(".") == 1
    ), "File requires an extension (format: <file-name>.<extension>)"
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


def get_day_list(interval: Tuple[date, date]) -> List[date]:
    """
    Get a list of days from an interval (tuple) of days (inclusive).

    Args:
        interval (Tuple[date, date]): A tuple containing the start and end dates.

    Returns:
        List[date]: A list of dates within the given interval.
    """
    # Calculate the difference between the start and end dates
    start, end = interval

    # Initialize an empty list to store the days
    days = []

    # Loop through each day in the interval
    for i in range((end - start).days + 1):
        # Calculate the date by adding the timedelta to the start date
        day = start + timedelta(days=i)
        # Add the day to the list
        days.append(day)

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
