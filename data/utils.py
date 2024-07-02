"""This module contains utilities for handling downloaded data from the s3 bucket."""
import logging
from enum import Enum
from typing import Tuple, List
from datetime import datetime, date, timedelta

# create logger object
logger = logging.getLogger(__name__)
logging.basicConfig(filename='s3prep.log', encoding='utf-8', level=logging.DEBUG)


class StorageMode(Enum):
    """
    Enum representing different storage modes.

    Attributes:
        IN_MEMORY (str): Storage in memory.
        ON_DISK (str): Storage on disk.
    """
    IN_MEMORY = "in_memory"
    ON_DISK = "on_disk"


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
        raise TypeError('Parameter was not of type date or datetime')

    # Return the week number from the ISO calendar
    return date_obj.isocalendar()[1]
