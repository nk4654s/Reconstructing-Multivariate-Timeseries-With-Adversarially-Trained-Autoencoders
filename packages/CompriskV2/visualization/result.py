from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from CompriskV2.model.utils import IteratorResult


def visualize_single_result(
    df: pd.DataFrame,
    name: str,
    unit: str = "N/A",
    ts_offset: int = 0,
    show_mse: bool = True,
    std: int = 1,
):
    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.copy()
        df.index = pd.to_datetime(df.index + ts_offset, unit="s")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    y_hat = np.array(df["y_hat"])
    score = np.array(df["score"])

    axs[0].plot(df["y_scaled"], color="teal", label="Data")
    axs[0].plot(df["y_hat"], color="orange", label="Prediction")
    axs[0].set_ylabel(unit)

    x = df["y_hat"].index
    if "var" in df.columns:
        var = np.array(df["y_1"])
        a2 = y_hat - std * (np.sqrt(var) + 1e-6)
        a1 = y_hat + std * (np.sqrt(var) + 1e-6)
        axs[0].fill_between(x, a1, a2, color="orange", alpha=0.3, label="Certainty")

    axs[1].plot(score, color="black", label="Healthscore")
    axs[1].set_ylabel("%")
    axs[1].set_ylim(-0.05, 1.05)

    # Format x-axis as dates
    axs[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    fig.suptitle(name)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.legend()

    return fig


def visualize_results(
    result: IteratorResult,
    units: dict | None = None,
    show_mse: bool = False,
    std: int = 1,
) -> None:
    units = units or {}
    for name in result.names:
        unit, name_str = units.get(name, ("N/A", name))
        df = result.by_name(name).to_dataframe()

        fig = visualize_single_result(df, name_str, unit, result.ts, show_mse, std)
        fig.show()
