"""Dataflux/helper/metadata.py"""

import unittest
from uuid import UUID

from Toking import Toking
from Dataflux.helper import get_sensors, SensorRegisterFormat, filter_upload


class TestMetadata(unittest.TestCase):
    def setUp(self):
        Toking()

        self.uuid = UUID("5ac7f03f-f4e6-4b15-9bc6-21db798f9d25")
        self.senors = {
            "8d753132-769a-33cd-b855-a14ed3a26cfe": "CPU Temp (°C)",
            "7ab52395-d5e1-31ac-ad5d-5e6d36d9aa64": "CPU Usage (%)",
            "082b3922-a7d9-3bf3-b015-e5acb19d1dbf": "Memory Usage (%)",
            "bdd640fb-0667-4ad1-9c80-317fa3b1799d": "EnddruckErsteStufePIT74339 (Malfunction AI) (%)",
            "bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9": "EndtemperaturErsteStufeTETIT74334 (Malfunction AI) (%)",
            "972a8469-1641-4f82-8b9d-2434e465e150": "EndtemperaturZweiteStufeTETIT74371 (Malfunction AI) (%)",
            "17fc695a-07a0-4a6e-8822-e8f36c031199": "EndtemperaturZweiteStufenachKühlerTETIT74378 (Malfunction AI) (%)",
            "9509b21f-60cb-402d-8723-8fefee4dcc4d": "OperatingHours (h)",
            "f2faabde-1b93-30c4-a825-817a8b125695": "GatewayOnlineStatus",
            "aaa994cf-2260-3832-900d-13c47e6ae870": "Storage Usage (%)",
            "3c0860fb-b478-376d-a5ab-a5441187277b": "Data Volume Remaining (MB)",
            "af3a8e0a-9be7-3292-b5d2-7ffec24642b2": "Signal Quality (%)",
            "ec1b8ca1-f91e-4d4c-9ff4-9b7889463e85": "VentilnesttemperaturZweiteStufeDeckelseiteTETT74367 (Malfunction AI) (%)",
            "4b0dbb41-8d52-48f1-942c-3fe860e7a113": "VentilnesttemperaturZweiteStufeKurbelseiteTETT74386 (Malfunction AI) (%)",
            "e2acf72f-9e57-4f7a-a0ee-89aed453dd32": "Wicklungstemperatur (Malfunction AI) (%)",
            "3139d32c-93cd-49bf-9c94-1cf0dc98d2c1": "ÖltemperaturVorlaufTriebwerkTETIT74538 (Malfunction AI) (%)",
            "a9488d99-0bbb-4599-91ce-5dd2b45ed1f0": "Total (Malfunction AI) (%)",
            "632f86ac-e987-4ce7-bd79-0c38e034adf8": "Biomethandurchfluss (Malfunction AI) (%)",
            "43b7a3a6-9a8d-4a03-980d-7b71d8f56413": "SaugtemperaturZweiteStufeTETIT74341 (Malfunction AI) (%)",
            "b74d0fb1-32e7-4629-8fad-c1a606cb0fb3": "KühlwasserdrucknachPumpePIT74425 (Malfunction AI) (%)",
            "6b65a6a4-8b81-48f6-b38a-088ca65ed389": "KühlwasserdurchflussmengeFIT74423 (Malfunction AI) (%)",
            "47378190-96da-4dac-b2ff-5d2a386ecbe0": "KühlwassertemperaturnachKühlerTETIT74422 (Malfunction AI) (%)",
            "6c307511-b2b9-437a-a8df-6ec4ce4a2bbd": "PackungsleckageFIT74389 (Malfunction AI) (%)",
            "16ea8594-78da-46e0-b882-99fa5956b541": "DifferenzdruckTriebwerksölPDIT74553 (Malfunction AI) (%)",
        }

    def test_get_sensors(self):
        """Test get sensors from test device."""
        sensors = get_sensors(self.uuid)
        self.assertEqual(sensors, self.senors)

    def test_get_sensors_name_is_key(self):
        """Test get sensors from test device with name as key."""
        sensors = get_sensors(self.uuid, SensorRegisterFormat.NAME_TO_UUID)
        expected = {name: uuid for uuid, name in self.senors.items()}
        self.assertEqual(sensors, expected)

    def test_get_sensors_filter(self):
        """Test get sensors from test device."""
        _filter = filter_upload()
        expected = {uuid: name for uuid, name in self.senors.items() if "Malfunction" in name}

        sensors = get_sensors(self.uuid, sensor_filter=_filter)
        self.assertEqual(sensors, expected)

    def test_get_sensors_name_is_key_filter(self):
        """Test get sensors from test device with name as key."""
        _filter = filter_upload(sensor_format=SensorRegisterFormat.NAME_TO_UUID)
        expected = {name: uuid for uuid, name in self.senors.items() if "Malfunction" in name}

        sensors = get_sensors(self.uuid, SensorRegisterFormat.NAME_TO_UUID, sensor_filter=_filter)
        self.assertEqual(sensors, expected)


if __name__ == "__main__":
    unittest.main()
