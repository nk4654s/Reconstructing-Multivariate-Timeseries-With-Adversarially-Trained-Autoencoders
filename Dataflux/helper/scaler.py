from enum import Enum
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class CustomRobustScaler:
    def __init__(self) -> None:
        self._all_q1 = []
        self._all_q3 = []

    def partial_fit(self, df: pd.DataFrame) -> None:
        self._all_q1.append(df.quantile(0.10))
        self._all_q3.append(df.quantile(0.90))
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        q1_min = pd.concat(self._all_q1, axis=1).min(axis=1)
        q3_max = pd.concat(self._all_q3, axis=1).max(axis=1)

        return (df - q1_min) / (q3_max - q1_min)
    
    def set_output(self, transform: str) -> None:
        pass


class ScalerMethod(Enum):
    """
    This enum is used to distinguish between different scaling methods.
    """
    NORMALIZE = "minmax"
    STANDARDIZE = "standard"
    COSTUM_ROBUST = "custom-robust"


class FluxScaler:
    """
    This class fits and transforms data using an SKLearn scaler. The available methods are defined
    by the `ScalerMethod` Enum.

    Args:
        method (ScalerMethod): The scaling method to use. Defaults to `NormalizerMethod.NORMALIZE`.
        exclude_keys (List[str], optional): A list of keys to exclude from scaling. Defaults to `None`.
    """
    def __init__(self, method = ScalerMethod.NORMALIZE, exclude_keys: List[str] = []) -> None:
        self._exclude = set() if exclude_keys is None else set(exclude_keys)
        self._internal_scaler = MinMaxScaler(feature_range=(-1, 1))
        self._method = method

        if method == ScalerMethod.STANDARDIZE:
            self._internal_scaler = StandardScaler()
        if method == ScalerMethod.COSTUM_ROBUST:
            self._internal_scaler = CustomRobustScaler()

    @property
    def method(self) -> ScalerMethod:
        """
        The scaling method used.
        """
        return self._method

    def overwrite_column(self, index: int, min_: float, max_: float) -> None:
        assert isinstance(self._internal_scaler, MinMaxScaler), "Overwriting columns on MinMax method available only."

        data_range = max_ - min_

        self._internal_scaler.data_min_[index] = min_
        self._internal_scaler.data_max_[index] = max_
        self._internal_scaler.data_range_[index] = data_range
        self._internal_scaler.min_[index] = -1 - min_ * (2 / data_range)

        # scale calculation: (feature_range_max - feature_range_min) / data_range
        # MinMax feature range is always [-1, 1] here, which results in 2
        scale = 2 / data_range if data_range != 0 else 0
        self._internal_scaler.scale_[index] = scale

    
    def fit(self, df: pd.DataFrame) -> "FluxScaler":
        """
        Fits the scaler to the data. Excluded keys are not considered for scaling.

        Args:
            df (pd.DataFrame): The dataframe to fit the scaler on.
        """
        header = sorted(list(set(df.columns.values) - self._exclude))
        self._internal_scaler.partial_fit(df[header])
        return self

    def transform(self, df: pd.DataFrame, dtype: str = "float32") -> pd.DataFrame:
        """
        This method transforms the dataframe using the fitted scaler. Then it converts the
        the transformed values to the provided dtype. Excluded keys are not changed in any way.
        
        Args:
            df (pd.DataFrame): The dataframe to transform.
            dtype (str): The dtype to convert the transformed values into.
        
        Returns:
            pd.DataFrame: The transformed dataframe with the provided dtype.
        """
        self._internal_scaler.set_output(transform="pandas")

        header = sorted(list(set(df.columns.values) - self._exclude))
        normed = df.copy()
        normed[header] = self._internal_scaler.transform(normed[header])
        normed[header] = normed[header].astype(dtype) # type: ignore
        return normed

    def revert(self, array: np.ndarray) -> np.ndarray:
        return self._internal_scaler.inverse_transform(array)
