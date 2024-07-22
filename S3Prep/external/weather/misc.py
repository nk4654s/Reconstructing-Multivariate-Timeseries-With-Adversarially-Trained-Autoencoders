"""This module contains helper classes for the weather dataset."""

import math


class Geo:
    """Helper class for representing geographical coordinates.

    Attributes:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
    """

    def __init__(self, lat: float, lon: float) -> None:
        """Initialize a new `Geo` object with the given latitude and longitude coordinates.

        Args:
            lat (float): Latitude coordinate.
            lon (float): Longitude coordinate.

        Returns:
            None
        """
        self._lat = lat
        self._lon = lon

    @property
    def lat(self) -> float:
        """Get the latitude coordinate of this `Geo` object.
        Returns:
            float: Latitude coordinate.
        """
        return self._lat

    @property
    def lon(self) -> float:
        """Get the longitude coordinate of this `Geo` object.

        Returns:
            float: Longitude coordinate.
        """
        return self._lon

    def __neg__(self):
        """Negate the latitude and longitude coordinates of this `Geo` object.

        Returns:
            Geo: A new `Geo` object with negated coordinates.
        """
        return Geo(-1 * self.lat, -1 * self.lon)

    def __add__(self, other):
        """Add another `Geo` object to this one, combining the latitude and longitude coordinates.

        Args:
            other (Geo): The other `Geo` object to add.

        Returns:
            Geo: A new `Geo` object with the combined coordinates.
        """
        return Geo(self.lat + other.lat, self.lon + other.lon)

    def __sub__(self, other):
        """Subtract another `Geo` object from this one, combining the latitude
        and longitude coordinates.

        Args:
            other (Geo): The other `Geo` object to subtract.

        Returns:
            Geo: A new `Geo` object with the combined coordinates.
        """
        return Geo(self.lat - other.lat, self.lon - other.lon)

    def distance(self, other: "Geo") -> float:
        """Calculate the Euclidean distance between this `Geo` object and another one.

        Args:
            other (Geo): The other `Geo` object to calculate the distance with.

        Returns:
            float: The Euclidean distance between the two `Geo` objects.
        """

        return math.sqrt((other.lat - self._lat) ** 2 + (other.lon - self._lon) ** 2)
