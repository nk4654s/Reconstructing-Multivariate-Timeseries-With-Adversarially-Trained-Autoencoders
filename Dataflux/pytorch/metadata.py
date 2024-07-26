"""This module contains the metadata of a pytorch.Dataset which is required for some AI models."""
from enum import Enum
from typing import Any


class MetadataTypes(Enum):
    """
    An enumeration of metadata types.

    Attributes:
        SENSOR (str): The number of features from sensors.
        WEATHER (str): The number of features from weather.
        TARGET (str): The number of target features.
        IN_SEQUENCE (str): The number of input sequence features.
        OUT_SEQUENCE (str): The number of output sequence features.
    """
    SENSOR = "num_features_sensors"
    WEATHER = "num_features_weather"
    TARGET = "num_features_targets"
    IN_SEQUENCE = "num_input_sequence"
    OUT_SEQUENCE = "num_output_sequence"


class Metadata:
    """
    A class to represent metadata of the dataset.

    Args:
        num_features_sensors (int): The number of features from sensors.
        num_features_weather (int): The number of features from weather.
        num_features_targets (int): The number of target features.
        num_input_sequence (int): The number of input sequence features.
        num_output_sequence (int): The number of output sequence features.
    """
    def __init__(self,
                 num_features_sensors: int,
                 num_features_weather: int,
                 num_features_targets: int,
                 num_input_sequence: int,
                 num_output_sequence: int) -> None:
        self._num_features_sensors = num_features_sensors
        self._num_features_weather = num_features_weather
        self._num_features_targets = num_features_targets
        self._num_input_sequence = num_input_sequence
        self._num_output_sequence = num_output_sequence

    @property
    def num_features_sensors(self) -> int:
        """
        Gets the number of features from sensors.

        Returns:
            int: The number of features from sensors.
        """
        return self._num_features_sensors

    @property
    def num_features_weather(self) -> int:
        """
        Gets the number of features from weather.

        Returns:
            int: The number of features from weather.
        """
        return self._num_features_weather

    @property
    def num_features_targets(self) -> int:
        """
        Gets the number of target features.

        Returns:
            int: The number of target features.
        """
        return self._num_features_targets

    @property
    def num_input_sequence(self) -> int:
        """
        Gets the number of input sequence features.

        Returns:
            int: The number of input sequence features.
        """
        return self._num_input_sequence

    @property
    def num_output_sequence(self) -> int:
        """
        Gets the number of output sequence features.

        Returns:
            int: The number of output sequence features.
        """
        return self._num_output_sequence


class MetadataFactory:
    """
    A factory class to create Metadata objects.
    """
    def __init__(self):
        self._metadata = {}

    def set(self, metadata_type: MetadataTypes, value: Any) -> None:
        """
        Sets a metadata value.

        Args:
            metadata_type (MetadataTypes): The metadata type.
            value (Any): The value to set.
        """
        self._metadata[metadata_type.value] = value

    def build(self) -> Metadata:
        """
        Builds the Metadata object from the factory's metadata.

        Returns:
            Metadata: The built Metadata object.
        """
        return Metadata(**self._metadata)
