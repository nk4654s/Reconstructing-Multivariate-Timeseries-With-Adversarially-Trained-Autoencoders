from typing import Dict, Tuple
import requests


def get_sensor_name_to_uuid(machine_uuid: str) -> Dict[str, str]:
    """This function returns a dictionary of sensor uuids to sensor names.
    
    Args:
        machine_uuid (str): The machine uuid to get the sensor names for.

    Returns:
        Dict[str, str]: A dictionary of sensor name to sensor uuid.
    """
    raise NotImplementedError()
    url = "https://dev.graphql.api.xplore.ac/"
    payload = ()

    headers = {
        "Content-Type": "application/json",
        "Authorization": "eyJhbGciOiIiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJ"
        + "jN2VlN2ZmNi03ZTQ3LTRmMGItYWIxMy0xYjM5NWE1NjZlNmIifQ.",
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
        sensor_dict[sensor["id"]] = sensor["name"] + unit
    return sensor_dict


def get_machine_location(machine_uuid: str) -> Tuple[float, float]:
    """This function returns the latitude and logintude of a machine.

    Args:
        machine_uuid (str): The machine's UUID from which the position is
        requested.

    Returns:
        Tuple[float, float]: Latitude and longitude.
    """
    url = "https://dev.graphql.api.xplore.ac/"
    payload = (
        '"query": "query getMachine '
        + '{\n\tmachine {\n\t\tgetMachine(machineId: \\"'
        + machine_uuid
        + '\\") {\n'
        + "\t\t\txrn\n\t\t\tname\n\t\t\tmission {\n\t\t\t\tlocation {\n"
        + '\t\t\t\t\tlat\n\t\t\t\t\tlng\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n}"'
        + ',"operationName":"getMachine"}'
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": "eyJhbGciOiIiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJ"
        + "jN2VlN2ZmNi03ZTQ3LTRmMGItYWIxMy0xYjM5NWE1NjZlNmIifQ.",
    }

    try:
        response = requests.request(
            "POST", url, data=payload, headers=headers, timeout=30
        )
    except requests.Timeout as _:
        return None

    coords = response.json()["data"]["machine"]["getMachine"]["mission"]["locataion"]
    return (coords["lat"], coords["lng"])


def get_sensor_uuid_to_name(machine_uuid: str) -> Dict[str: str]:
    """This function returns a dictionary of sensor uuids to sensor names."""
    url = "https://dev.graphql.api.xplore.ac/"
    payload = (
        '{"query":"query getMachine {\n  machine {\n    getMachine'
        + f'(machineId: \\"{machine_uuid}\\") '
        + "{\n      name\n \t\t\tsensors { properties\n\t\t\t\tid\n\t\t\t\tname\n"
        + "\t\t\t\ttype\n\t\t\t\tkind {\n\t\t\t\t\tis_plc\n\t\t\t\t\tis_hidden\n\t"
        + "\t\t\t\tis_virtual\n\t\t\t\t\tis_setpoint\n\t\t\t\t\tis_plc_alarm\n\t"
        + "\t\t\t\tis_high_frequency\n\t\t\t\t\tis_infrastructure\n\t\t\t\t}"
        + '\n\t\t\t}\n    }\n  }\n}","operationName":"getMachine"}'
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": "eyJhbGciOiIiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJ"
        + "jN2VlN2ZmNi03ZTQ3LTRmMGItYWIxMy0xYjM5NWE1NjZlNmIifQ.",
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
        sensor_dict[sensor["id"]] = sensor["name"] + unit
    return sensor_dict