"""This module contains utilities for handling downloaded data from the s3 bucket."""

import logging
from typing import Tuple, List
from datetime import datetime, date, timedelta
import requests

# create logger object
logger = logging.getLogger(__name__)
logging.basicConfig(filename="s3prep.log", encoding="utf-8", level=logging.DEBUG)


def get_sensor_uuid_to_name(machine_uuid: str) -> dict:
    """This function returns a dictionary of sensor uuids to sensor names.
    """
    url = "https://dev.graphql.api.xplore.ac/"

    payload = '{"query":"query getMachine {\n  machine {\n    getMachine' + \
        f'(machineId: \\\"{machine_uuid}\\\") ' + \
            '{\n      name\n \t\t\tsensors { properties\n\t\t\t\tid\n\t\t\t\tname\n' + \
                '\t\t\t\ttype\n\t\t\t\tkind {\n\t\t\t\t\tis_plc\n\t\t\t\t\tis_hidden\n\t' + \
                    '\t\t\t\tis_virtual\n\t\t\t\t\tis_setpoint\n\t\t\t\t\tis_plc_alarm\n\t' + \
                        '\t\t\t\tis_high_frequency\n\t\t\t\t\tis_infrastructure\n\t\t\t\t}' + \
                            '\n\t\t\t}\n    }\n  }\n}","operationName":"getMachine"}'
    headers = {
        "Content-Type": "application/json",
        "Authorization": "eyJhbGciOiIiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJ" + \
            "jN2VlN2ZmNi03ZTQ3LTRmMGItYWIxMy0xYjM5NWE1NjZlNmIifQ.",
    }
    try:
        response = requests.request("POST", url, data=payload, headers=headers,
                                    timeout=30)
    except requests.Timeout as e:
        logger.warning("Request to %s timed out: %s", url, e)
        return None

    sensor_list = response.json()["data"]["machine"]["getMachine"]["sensors"]
    sensor_dict = {}
    for sensor in sensor_list:
        unit = sensor["properties"].get("unit", None)
        unit = f" ({unit})" if unit else ""
        sensor_dict[sensor["id"]] = sensor["name"] + unit
    return sensor_dict


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
