"""This module downloads and pre-processes weather data."""

import zipfile
import io
from datetime import datetime
import numpy as np
import pandas as pd
import requests


class WeatherData:
    """
    This class represents weather data.
    
    Attributes:
        BASE_URL (str): The base URL for the German Weather Service (DWD) API.
    """
    BASE_URL = (
        "https://opendata.dwd.de/climate_environment/CDC/"
        + "observations_germany/climate/10_minutes/air_temperature/historical/"
    )
    def __init__(self):
        response = requests.get(WeatherData.BASE_URL, timeout=30)
        self._sub_links = []
        for link in response.content.decode("cp1252").split("\n"):
            if link.startswith('<a href="'):
                self._sub_links.append(
                    link.split('">', maxsplit=1)[0].replace('<a href="', "")
                )

    def _extract_to_dataframe(self, link: str, station_id: str) -> pd.DataFrame:
        """Extracts weather data from a zip file and returns it as pandas dataframe.

        Args:
            link (str): The link to the weather data.
            station_id (str): The id of the weather station.

        Returns:
            pd.DataFrame: The weather data as pandas dataframe.
        """
        response = requests.get(link, timeout=30)
        start_date = link.split("_")[7]
        end_date = link.split("_")[8]
        output_str = f"produkt_zehn_min_tu_{start_date}_{end_date}_{station_id}.txt"

        # Download the zip file (assuming you have a response object 'response' from your request)
        buf = io.BytesIO(response.content)

        # Open the zip file in read binary mode
        with zipfile.ZipFile(buf, "r") as z:
            # Extract the text file from the zip archive
            with z.open(output_str, "r") as f:
                # Read the text file into a string
                file_contents = (
                    io.TextIOWrapper(f, encoding="cp1252")
                    .read()
                    .replace(" ", "")
                    .replace("eor", "")
                    .split("\n")
                )
                header = [_ for _ in file_contents.pop(0).split(";") if len(_) > 0]
                data = {h: [] for h in header if h in ["MESS_DATUM", "TT_10"]}

                for line in file_contents:
                    cells = line.split(";")
                    if len(cells) <= 1:
                        continue

                    for j, h in enumerate(header):
                        if h == "MESS_DATUM":
                            date = datetime.strptime(cells[j], "%Y%m%d%H%M")
                            cells[j] = date.timestamp()

                        elif h == "TT_10":
                            v = float(cells[j])
                            cells[j] = v if -100 < v < 100 else np.nan
                        else:
                            continue

                        row = data.get(h, [])
                        row.append(cells[j])
                        data[h] = row

                data["ts"] = data.pop("MESS_DATUM")
                data["Aussentemperatur (°C)"] = data.pop("TT_10")
                df = pd.DataFrame(data, dtype=np.float32)

        return df

    def get_data_by_id(self, station_id: str, start: str, end: str) -> pd.DataFrame:
        """Extracts weather data from a zip file and returns it as pandas dataframe.

        Args:
            station_id (str): ID of the weather station
            start (str): Start year of the data to extract
            end (str): End year of the data to extract

        Returns:
            pd.DataFrame: Pandas dataframe containing weather data.
        """
        start = datetime.strptime(start, "%Y%m%d")
        end = datetime.strptime(end, "%Y%m%d")

        df = pd.DataFrame()
        for link in self._sub_links:
            if link.split("_")[2] != station_id:
                continue

            start_date = datetime.strptime(link.split("_")[3], "%Y%m%d")
            end_date = datetime.strptime(link.split("_")[4], "%Y%m%d")
            if end > start_date:
                if start <= start_date or start <= end_date:
                    new_df = self._extract_to_dataframe(WeatherData.BASE_URL + link, station_id)
                    df = pd.concat([df, new_df])
                    df.interpolate(method="linear", inplace=True)

        return df
