"""This module contains helper classes for the weather dataset."""
import math


class Geo:
    def __init__(self, lat: float, lon: float) -> None:
        self._lat = lat
        self._lon = lon

    @property
    def lat(self) -> float:
        return self._lat

    @property
    def lon(self) -> float:
        return self._lon

    def __neg__(self):
        return Geo(-1 * self.lat, -1 * self.lon)

    def __add__(self, other):
        return Geo(
            self.lat + other.lat, 
            self.lon + other.lon
        )

    def __sub__(self, other):
        return Geo(
            self.lat - other.lat, 
            self.lon - other.lon
        )

    def distance(self, other: 'Geo') -> float:
        return math.sqrt((other._lat - self._lat)**2 + (other._lon - self._lon)**2)
