from enum import Enum
from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class ScalerMethod(Enum):
    """
    This enum is used to distinguish between different scaling methods.
    """
    NORMALIZE = "minmax"
    STANDARDIZE = "standard"
    STANDARDIZE_ROBUST = "robust"


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
        if method == ScalerMethod.STANDARDIZE_ROBUST:
            self._internal_scaler = RobustScaler()

    @property
    def method(self) -> ScalerMethod:
        """
        The scaling method used.
        """
        return self._method
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fits the scaler to the data. Excluded keys are not considered for scaling.

        Args:
            df (pd.DataFrame): The dataframe to fit the scaler on.
        """
        header = list(set(df.columns.values) - self._exclude)
        match self._method:
            case ScalerMethod.STANDARDIZE_ROBUST:
                # for some reason, the robust scaler requires the partial_fit() method to
                # be used, when the provided data does not cover the whole dataset
                self._internal_scaler.partial_fit(df[header]) # type: ignore
            case _:
                self._internal_scaler.fit(df[header])

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

        header = list(set(df.columns.values) - self._exclude)
        normed = df.copy()
        normed[header] = self._internal_scaler.transform(normed[header])
        normed[header] = normed[header].astype(dtype) # type: ignore
        return normed
