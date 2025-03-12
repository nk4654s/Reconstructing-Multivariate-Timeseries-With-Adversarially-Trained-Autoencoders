import uuid
from collections.abc import Callable
from enum import Enum, auto
from functools import cache
from typing import Dict, List, Tuple
from uuid import UUID

import requests
from Toking import Toking

from Dataflux import get_logger, logging_mode

logger = get_logger(__name__, "logfile", logging_mode)


class SensorRegisterFormat(Enum):
    """This enum describes how the sensor dictionary looks like. Either with UUIDs or sensor names as the key."""

    UUID_TO_NAME = auto()
    NAME_TO_UUID = auto()


class NamespaceNull(uuid.UUID):
    bytes = b""


def gen_health_score_sensor_ids(
    sensor_id: str | uuid.UUID, values: List[int]
) -> List[uuid.UUID]:
    f"""
    This function generates UUIDs given a sensor id (UUID) and a list of versions, which serve as generation
    seed. The function builds a string ':neax/health-score/{sensor_id}/{values}'. The generated UUID's version
    is V3.

    Args:
        sensor_id (Union[str, uuid.UUID]): The sensor's UUID which is required to create the seed string. 
        values (List[int]): The list of seeds.

    Returns:
        List[uuid.UUID]: The list of generated UUIDs.
    """
    sensor_id = str(sensor_id)
    gen_uuids = []

    base = ":neax/health-score"
    for seed in values:
        suffix = f"/{seed}" if seed != 0 else ""
        data_str = f"{base}/{sensor_id}{suffix}"
        gen_uuids.append(uuid.uuid3(NamespaceNull, data_str))

    return gen_uuids


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


def get_sensors_ai(
    machine_uuid: UUID | str,
    ai_tag: str = "Healthscore",
    sensor_format: SensorRegisterFormat = SensorRegisterFormat.UUID_TO_NAME,
    debug: bool = False,
) -> Dict:
    sensor_filter = filter_upload(sensor_format=sensor_format, include=ai_tag)
    register = get_sensors(
        machine_uuid=machine_uuid, sensor_format=sensor_format, debug=debug
    )
    return sensor_filter(register)


@cache
def score_uuid_mapping(
    score_uuids: Tuple[str], machine_id: uuid.UUID, ai_tag: str
) -> Tuple[Dict[str, str], List[str]]:
    """This function maps sensor UUIDs to their corresponding AI sensor UUID counterpart.

    Args:
        score_uuids: The UUIDs one is interested in. If some were not found in the mapping, then the UUIDs are
            added to the "drop" list in return.
        machine_id: The machine's UUID to get the sensor UUIDs from.
        ai_tag: The substring which identifies a sensor to be an AI sensor.

    Returns:
        Tuple[Dict, List]: The UUID to UUID mapping and the list of sensor UUIDs which do not have a mapping.

    """
    # rename the sensors to upload to "sensor_name + AI_TAG"
    format_ = SensorRegisterFormat.NAME_TO_UUID
    sensor_register = get_sensors(machine_id)
    ai_register = get_sensors_ai(machine_id, ai_tag=ai_tag, sensor_format=format_)

    drop = []
    mapping = {}
    for s_uuid in score_uuids:
        # check if the sensor is registered for the machine
        s_name = sensor_register.get(s_uuid, None)
        if not s_name:
            drop.append(s_uuid)
            continue

        # check if there is an AI sensor registered for the machine
        ai_names = sorted([s for s in ai_register if s_name.split()[0] in s.split()])
        ai_uuid = ai_register.get(ai_names[0]) if ai_names else None
        if not ai_uuid:
            drop.append(s_uuid)
            continue

        # add the sensor uuid to AI sensor uuid mapping
        g_uuid = str(gen_health_score_sensor_ids(s_uuid, [0])[0])
        if g_uuid != ai_uuid:
            logger.warning("Generated AI UUID does not match %s -> %s", s_uuid, g_uuid)
            logger.warning("Expected %s instead", ai_uuid)

        mapping[s_uuid] = ai_uuid

    return mapping, drop
