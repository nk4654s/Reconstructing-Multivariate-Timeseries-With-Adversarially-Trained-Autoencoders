import os
import yaml
import pandas as pd
import numpy as np
from torch import Tensor


def process_df(data: pd.DataFrame, label: dict[str, str]) -> tuple[Tensor, tuple]:
    sub = data[label.keys()].copy()
    sub.rename(columns=label, inplace=True)
    sub = sub.reindex(sorted(sub.columns), axis=1)

    return Tensor([sub.values]), (sub.values.min(axis=0), sub.values.max(axis=0))


def load_name_mapping(path: str) -> tuple[dict, dict]:
    """This method loads the mapping of KO3 names to Xplore's sensor UUIDs.

    Args:
        path (str): Path of the name mapping.

    Returns:
        tuple[dict, dict]: Mapping of the input / output names
    """
    with open(path, "r") as f:
        mapping = yaml.safe_load(f)
        return mapping["inputs"], mapping["outputs"]


def load_support_points_table(path: str) -> pd.DataFrame:
    """This method loads the partitioned KO3 export as pandas DataFrame.

    Returns:
        pd.DataFrame: The KO3 export.
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if ".csv" in f]

    # read the support points export
    df = pd.DataFrame()
    for p in sorted(files):
        if not p.endswith("csv"):
            continue

        csv = pd.read_csv(
            p,
            delimiter=";",
            decimal=",",
            dtype=np.float32,
            skiprows=[1],
            header=0,
            encoding="utf-8",
        )
        df = pd.concat([df, csv], ignore_index=True)

    return df
