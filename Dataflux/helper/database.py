from typing import Dict, Tuple
from uuid import UUID
import requests


def get_sensor_name_to_uuid(machine_uuid: UUID) -> Dict[str, str]:
    """This function returns a dictionary of sensor uuids to sensor names.
    
    Args:
        machine_uuid (UUID): The machine uuid to get the sensor names for.

    Returns:
        Dict[str, str]: A dictionary of sensor name to sensor uuid.
    """
    # TODO: get from backend
    return {
        "EnddruckErsteStufePIT74339 (bar)": "0094a079-ac28-3bd5-a694-414679f00a6e",
        "EnddruckZweiteStufePIT74397 (bar)": "be91c533-7376-3633-883b-a17b1d53f9ff",
        "EndtemperaturErsteStufeTETIT74334 (°C)": "3085626d-be6e-3713-9a98-292fac5963c2",
        "EndtemperaturZweiteStufeTETIT74371 (°C)": "e26a5188-7ce4-3b8f-a3bc-1f588d3e08e7",
        "EndtemperaturZweiteStufenachKühlerTETIT74378 (°C)": "f53970b1-45ec-3087-a9c7-5a96695ee7c6",
        "KompressordrehzahlSE74201 (rpm)": "fb083db1-b9c1-3366-968c-823162df9669",
        "KühlwasserdrucknachPumpePIT74425 (bar)": "130e1ca0-a6c5-3678-a68d-f79dcfc2148d",
        "KühlwasserdurchflussmengeFIT74423 (m^3/min)": "f8be2faf-fbc3-3d4c-ae02-fb45ad4e9d95",
        "KühlwassertemperaturnachKühlerTETIT74422 (°C)": "03cad2f9-00df-36dc-aec7-2b7a024cf46b",
        "Motorlager (°C)": "23d9b983-d139-3113-81e7-397f842be938",
        "PackungsleckageFIT74389 (Nm3h)": "0bcf7769-8978-3a1f-94f9-4d9a02d03c36",
        "SaugdruckErsteStufePIT74319 (bar)": "5d023116-ddf9-39aa-a361-3dc583ade6f9",
        "SaugdruckZweiteStufePIT74340 (bar)": "99f00205-629d-318d-b125-03b7f3bc219a",
        "SaugtemperaturErsteStufeTETIT74328 (°C)": "b2c6bf8f-16a6-3069-af92-e150cbf7c336",
        "SaugtemperaturZweiteStufeTETIT74341 (°C)": "76428f28-f1c8-3661-b23d-95cb9f257eb6",
        "SchwingungsmessungTriebwerkVT74280 (mm/s)": "5c8e4594-1131-33be-b766-1834e8659a8a",
        "VentilnesttemperaturZweiteStufeDeckelseiteTETT74367 (°C)": "fb192925-66fe-3d68-a18e-0f1021e3dcf1",
        "VentilnesttemperaturZweiteStufeKurbelseiteTETT74386 (°C)": "cc50d022-df40-33bd-a926-396c1a122faf",
        "Wicklungstemperatur (°C)": "cdfcbe0b-c5d6-3776-b978-ca62a0f57c2a",
        "ÖltemperaturVorlaufTriebwerkTETIT74538 (°C)": "8c3eceb2-2144-35b3-bc33-79a17488624d",
        "Total": "ad127252-e4ff-3fe9-8d7c-630fa4ea1203"
    }


def get_machine_location(machine_uuid: UUID) -> Tuple[float, float] | None:
    """This function returns the latitude and logintude of a machine.

    Args:
        machine_uuid (UUID): The machine's UUID from which the position is
        requested.

    Returns:
        Tuple[float, float]: Latitude and longitude.
    """
    url = "https://dev.graphql.api.xplore.ac/"
    payload = (
        '"query": "query getMachine '
        + '{\n\tmachine {\n\t\tgetMachine(machineId: \\"'
        + str(machine_uuid)
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


def get_sensor_uuid_to_name(machine_uuid: UUID) -> Dict[str, str] | None:
    """This function returns a dictionary of sensor uuids to sensor names."""
    url = "https://dev.graphql.api.xplore.ac/"
    payload = (
        '{"query":"query getMachine {\n  machine {\n    getMachine'
        + f'(machineId: \\"{str(machine_uuid)}\\") '
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