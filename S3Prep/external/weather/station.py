"""This module contains the weather station's information."""
import requests
import numpy as np
import pandas as pd

from .misc import Geo

URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/historical/zehn_min_tu_Beschreibung_Stationen.txt"


class WeatherStations:
    def __init__(self) -> None:
        # get the description of all weather stations
        register_link = URL
        response = requests.get(register_link)
        register = (
            response.content.decode("cp1252", "ignore").replace("\r", "").split("\n")
        )
        register.pop(1)

        # extract the header
        header = [_ for _ in register.pop(0).split(" ") if len(_) > 0]

        # extract each station and put information into a DataFrame
        data = {}
        for i in range(len(register)):
            cells = [_ for _ in register[i].split(" ") if len(_) > 0]
            if len(cells) == 0:
                continue
            for j, h in enumerate(header):
                row = data.get(h, [])
                row.append(cells[j])
                data[h] = row

        self._df = pd.DataFrame(data)
    
    def get_station_id_by_name(self, name: str) -> str:
        """The function returns the ID of a weather station by its name.

        Args:
            name (str): The name of the weather station
        
        Returns:
            str: The ID of the weather station
        """
        station_names = self._df["Stationsname"]
        row = self._df[station_names == name]
        return row["Stations_id"].to_list()[0]
    
    def closest_station(self, coord: Geo) -> str:
        """The function returns the ID of a weather station that is closest to a given coordinate.

        Args:
            coord (Geo): The coordinate to find the closest weather station to.
        
        Returns:
            str: The ID of the closest weather station.
        """
        dists = [Geo(float(_[0]), float(_[1])).distance(coord) for _ in self._df[["geoBreite", "geoLaenge"]].to_numpy()]
        closest_idx = np.argmin(np.array(dists))

        return self._df["Stations_id"].to_numpy()[closest_idx]
