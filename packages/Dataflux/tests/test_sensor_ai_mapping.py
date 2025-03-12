import unittest
from unittest.mock import patch

from Dataflux.helper import SensorRegisterFormat, get_sensors_ai


class TestSensorMappingWithAI(unittest.TestCase):
    def setUp(self):
        self.uuid = "mocked"
        # UUIDs generated using `uuid.uuid3(uuid.NAMESPACE_OID, s)`
        self.register = {
            "e06cebdb-5e75-3ba8-a275-f36bc77d9c82": "neax/sensor/0",
            "ba7c42ec-5b2d-3787-94b9-da094ac3a3a8": "neax/sensor/1",
            "beceb583-81db-3cf2-b3a6-9e38852d5bf8": "neax/sensor/2",
            "89e6e56b-d6e4-30e9-83ae-b77c92412690": "neax/sensor/3",
            "64dc5865-36e1-3bf7-9085-74e918df894e": "neax/sensor/4",
            "05dcb92b-8bcb-3506-bbee-26920e687091": "neax/sensor/5",
        }

        # UUIDs generated using the string "f"neax/health-score/{uuid.uuid3(uuid.NAMESPACE_OID, u)}/{n}""
        # where "u" is the original sensor UUID
        self.ai_register = {
            "Healthscore: neax/sensor/0": "e658ac05-79e2-30cd-8fca-1ed4105785f5",
            "Healthscore: neax/sensor/1": "0878ccd3-73cf-3cdc-b4ba-5319e9e40bfa",
            "Healthscore: neax/sensor/2": "f350208e-5c10-35ab-8068-7a039e0af911",
            "Healthscore: neax/sensor/3": "6373d5e7-c2ee-33b0-9a8a-3c38e78206fa",
            "Healthscore: neax/sensor/4": "bfabba72-edb9-3c00-b2ea-edc147ee4bc1",
            "Healthscore: neax/sensor/5": "11e77a68-c8ed-3aa7-a13f-d00a24f8762d",
        }

        self.sensor_names = list(self.register.values())
        self.ai_names = list(self.ai_register.keys())
        self.ai_tag = "Healthscore:"

        self.register |= {v: k for k, v in self.ai_register.items()}

    @patch("Dataflux.helper.sensor_register.get_sensors")
    def test_filter_healthscore_uuid_to_name(self, mock_get_sensors):
        mock_get_sensors.return_value = self.register

        ai_sensors = get_sensors_ai(
            machine_uuid=self.uuid,
            ai_tag=self.ai_tag,
            sensor_format=SensorRegisterFormat.UUID_TO_NAME,
        )
        self.assertDictEqual({v: k for k, v in self.ai_register.items()}, ai_sensors)

    @patch("Dataflux.helper.sensor_register.get_sensors")
    def test_filter_healthscore_name_to_uuid(self, mock_get_sensors):
        mock_get_sensors.return_value = {v: k for k, v in self.register.items()}

        ai_sensors = get_sensors_ai(
            machine_uuid=self.uuid,
            ai_tag=self.ai_tag,
            sensor_format=SensorRegisterFormat.NAME_TO_UUID,
        )
        self.assertDictEqual(self.ai_register, ai_sensors)

    @patch("Dataflux.helper.sensor_register.get_sensors")
    def test_filter_empty(self, mock_get_sensors):
        mock_get_sensors.return_value = {
            v: k for k, v in self.register.items() if self.ai_tag not in v
        }

        ai_sensors = get_sensors_ai(
            machine_uuid=self.uuid,
            ai_tag=self.ai_tag,
            sensor_format=SensorRegisterFormat.NAME_TO_UUID,
        )
        self.assertDictEqual({}, ai_sensors)