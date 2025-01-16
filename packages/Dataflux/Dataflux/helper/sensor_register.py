from enum import Enum, auto
from typing import Dict, List, Callable
from uuid import UUID
import requests

from Toking import Toking


class SensorRegisterFormat(Enum):
    """This enum describes how the sensor dictionary looks like. Either with UUIDs or sensor names as the key."""

    UUID_TO_NAME = auto()
    NAME_TO_UUID = auto()


def filter_upload(
    registered: List = None,
    sensor_format: SensorRegisterFormat = SensorRegisterFormat.UUID_TO_NAME,
    include: str = "Malfunction",
) -> Callable:
    """This filter makes sure to only include sensors which are in the registered list of sensors and have
    the `include` string in their names. The `registered` can contain either UUIDs or sensor names, make sure to
    define the `sensor_format` correctly.

    Args:
        registered (List): The registered sensors as names or UUIDs.
        sensor_format (SensorRegisterFormat): The format in which the registered list is given.
        include (str): The substring a sensor name must include to be in the final result.

    Returns:
        Callable: A function used to filter sensors. Provide this function to `get_sensors()`.
    """

    def _filter(sensor_register: Dict) -> Dict:
        filtered = {}

        if sensor_format == SensorRegisterFormat.UUID_TO_NAME:
            uuids = sensor_register.keys() if not registered else registered
            names = [
                name.split(" ")[0]
                for uuid, name in sensor_register.items()
                if uuid in uuids
            ]

            for uuid, name in sensor_register.items():
                if include in name and name.split(" ")[0] in names:
                    filtered[uuid] = name
        else:
            names = sensor_register.keys() if not registered else registered
            names = [
                name.split(" ")[0]
                for name, _ in sensor_register.items()
                if name in names
            ]

            for name, uuid in sensor_register.items():
                if include in name and name.split(" ")[0] in names:
                    filtered[name] = uuid

        return filtered

    return _filter


def filter_empty(sensor_register: Dict) -> Dict:
    return sensor_register


def get_sensors(
    machine_uuid: UUID,
    sensor_format: SensorRegisterFormat = SensorRegisterFormat.UUID_TO_NAME,
    sensor_filter: Callable = filter_empty,
    debug: bool = False,
) -> Dict[str, str] | None:
    """This function returns a dictionary of sensor.

    Args:
        machine_uuid (UUID): The machine's uuid to download the sensors from.
        sensor_format (SensorRegisterFormat): Retrieve the format either in UUID to name or reversed.
        sensor_filter (Callable): A filter function which receives a `Dict` and returns a filtered `Dict`.
        debug (bool): If debug is enabled, the `machine_uuid` will be overwritten by
            `5ac7f03f-f4e6-4b15-9bc6-21db798f9d25`.

    Returns:
        Dict: A dictionary containing sensors and uuids.
    """
    if debug:
        machine_uuid = UUID("5ac7f03f-f4e6-4b15-9bc6-21db798f9d25")

    url = "https://stage.graphql.api.xplore.ac/"
    payload = f"""
        {{"query":"query getMachine {{
          machine {{
            getMachine(machineId: \\"{str(machine_uuid)}\\") {{
              name
              sensors {{
                properties
                id
                name
                type
                kind {{
                  is_plc
                  is_hidden
                  is_virtual
                  is_setpoint
                  is_plc_alarm
                  is_high_frequency
                  is_infrastructure
                }}
              }}
            }}
          }}
        }}","operationName":"getMachine"}}
        """

    headers = {
        "Content-Type": "application/json",
        "Authorization": Toking().get_access_token(),
    }
    try:
        response = requests.request(
            "POST", url, data=payload, headers=headers, timeout=30
        )
    except requests.Timeout as _:
        return None

    sensor_list = response.json()["data"]["machine"]["getMachine"]["sensors"]
    sensor_dict = {}
    for sensor in sensor_list:
        unit = sensor["properties"].get("unit", None)
        unit = f" ({unit})" if unit else ""
        if sensor_format == SensorRegisterFormat.NAME_TO_UUID:
            sensor_dict[sensor["name"] + unit] = sensor["id"]
        else:
            sensor_dict[sensor["id"]] = sensor["name"] + unit

    return sensor_filter(sensor_dict)
