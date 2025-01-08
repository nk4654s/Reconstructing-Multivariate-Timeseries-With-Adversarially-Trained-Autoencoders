from typing import Dict, Tuple
from uuid import UUID
import os
import requests


class MissingEnvironmentVariable(Exception):
    """This class defines the exception to throw when trying to
    access a missing environment variable. Provide the variable's name to
    the constructor for displaying a correct message.

    Args:
        key_name (str): The missing variable's name.
    """
    def __init__(self, key_name: str, *args: object) -> None:
        message = f"Make sure to define the '{key_name}' environment variable according to your system."
        super().__init__(message, *args)


def get_secret() -> str | None:
    """This method loads the authorization key for the database from the ".env" file, which
    should be located next to this code.

    Raises:
        MissingEnvironmentVariable: If the 'DB_SECRET' could not be found in the environment
            variables.
    """
    if "DB_SECRET" not in os.environ:
        raise MissingEnvironmentVariable("DB_SECRET")

    return os.environ["DB_SECRET"]


def get_sensor_name_to_uuid(machine_uuid: UUID) -> Dict[str, str]:
    """This function returns a dictionary of sensor uuids to sensor names.
    
    Args:
        machine_uuid (UUID): The machine uuid to get the sensor names for.

    Returns:
        Dict[str, str]: A dictionary of sensor name to sensor uuid.
    """
    return {"OperatingHours (h)": "9509b21f-60cb-402d-8723-8fefee4dcc4d",
            "Connectivity": "d7c39371-2bff-3995-93f2-802100f2e575",
            "CPU Temp (°C)": "8d753132-769a-33cd-b855-a14ed3a26cfe",
            "CPU Usage (%)": "7ab52395-d5e1-31ac-ad5d-5e6d36d9aa64",
            "Memory Usage (%)": "082b3922-a7d9-3bf3-b015-e5acb19d1dbf",
            "EnddruckErsteStufePIT74339 (Malfunction AI) (%)": "bdd640fb-0667-4ad1-9c80-317fa3b1799d",
            "EndtemperaturErsteStufeTETIT74334 (Malfunction AI) (%)": "bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9",
            "EndtemperaturZweiteStufeTETIT74371 (Malfunction AI) (%)": "972a8469-1641-4f82-8b9d-2434e465e150",
            "EndtemperaturZweiteStufenachKühlerTETIT74378 (Malfunction AI) (%)": "17fc695a-07a0-4a6e-8822-e8f36c031199",
            "KompressordrehzahlSE74201 (Malfunction AI) (%)": "9a1de644-815e-46d1-bb8f-aa1837f8a88b",
            "Storage Usage (%)": "aaa994cf-2260-3832-900d-13c47e6ae870",
            "Data Volume Remaining (MB)": "3c0860fb-b478-376d-a5ab-a5441187277b",
            "KühlwasserdrucknachPumpePIT74425 (Malfunction AI) (%)": "b74d0fb1-32e7-4629-8fad-c1a606cb0fb3",
            "KühlwasserdurchflussmengeFIT74423 (Malfunction AI) (%)": "6b65a6a4-8b81-48f6-b38a-088ca65ed389",
            "KühlwassertemperaturnachKühlerTETIT74422 (Malfunction AI) (%)": "47378190-96da-4dac-b2ff-5d2a386ecbe0",
            "PackungsleckageFIT74389 (Malfunction AI) (%)": "6c307511-b2b9-437a-a8df-6ec4ce4a2bbd",
            "SaugdruckZweiteStufePIT74340 (Malfunction AI) (%)": "1a2a73ed-562b-4f79-8374-59eef50bea63",
            "VentilnesttemperaturZweiteStufeDeckelseiteTETT74367 (Malfunction AI) (%)": "ec1b8ca1-f91e-4d4c-9ff4-9b7889463e85",
            "VentilnesttemperaturZweiteStufeKurbelseiteTETT74386 (Malfunction AI) (%)": "4b0dbb41-8d52-48f1-942c-3fe860e7a113",
            "Wicklungstemperatur (Malfunction AI) (%)": "e2acf72f-9e57-4f7a-a0ee-89aed453dd32",
            "ÖltemperaturVorlaufTriebwerkTETIT74538 (Malfunction AI) (%)": "3139d32c-93cd-49bf-9c94-1cf0dc98d2c1",
            "Biomethandurchfluss (Malfunction AI) (%)": "632f86ac-e987-4ce7-bd79-0c38e034adf8",
            "DifferenzdruckTriebwerksölPDIT74553 (Malfunction AI) (%)": "16ea8594-78da-46e0-b882-99fa5956b541",
            "Total (Malfunction AI) (%)": "a9488d99-0bbb-4599-91ce-5dd2b45ed1f0"}
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
    url = "https://stage.graphql.api.xplore.ac/"
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
        "Authorization": get_secret(),
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
    url = "https://stage.graphql.api.xplore.ac/"
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
        "Authorization": get_secret(),
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
