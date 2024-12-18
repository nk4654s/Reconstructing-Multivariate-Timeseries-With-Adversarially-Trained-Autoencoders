import pandas as pd


def loss_fn(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.DataFrame:
    return y - y_hat


class BaseTransformer:
    def fit(self, predicted: pd.DataFrame, target: pd.DataFrame) -> "BaseTransformer":
        return self

    def transform(self,predicted: pd.DataFrame, target: pd.DataFrame, **kwargs) -> "BaseTransformer":
        return self
