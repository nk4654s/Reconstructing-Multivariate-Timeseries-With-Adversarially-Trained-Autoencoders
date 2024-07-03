"""This module removes rows from the dataframe if the machine is off."""
from pandas import DataFrame
import numpy as np
from . import BasePreProcessor


class DeleteWhenMachineOff(BasePreProcessor):
    """This class removes rows from the dataframe if the machine is off."""
    def __init__(self, on_key: str = "eedce5b6-b341-4c58-bf3a-c6c1f265ba02"):
        """The constructor of the class. This will be automatically executed by
        the CLIArgumentParser.

        Args:
            on_key (str): The key of the column that contains the machine state.
            Defaults to "eedce5b6-b341-4c58-bf3a-c6c1f265ba02".
        """
        super().__init__()
        self._column_key = on_key

    def __call__(self, df: DataFrame) -> DataFrame:
        sensor_val = df[self._column_key].to_numpy()
        sensor_val[sensor_val == 0] = np.nan
        df[self._column_key] = sensor_val
        df.dropna(how="any", inplace=True)

        return df
