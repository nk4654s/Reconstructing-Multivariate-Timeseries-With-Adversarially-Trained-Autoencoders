"""This module handles showing and saving graphs of a dataframe."""
import math
from typing import List
import matplotlib.pyplot as plt
import pandas as pd

colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "olive", "cyan"]


def show_dataframe(
    sensors: str | List[str],
    df: pd.DataFrame | List[pd.DataFrame],
    names: List[str] = None,
    grid_size: int = 4,
    save: str = None,
    show: bool = False,
):
    """Shows and/or saves the visualization of one or multiple pandas DataFrames. In total, 9
    concurrent DataFrames can be visualized in one figure.

    Args:
        sensors (str | List[str]): The name of the sensor or a list of sensor names to be
            visualized.
        df (pd.DataFrame | List[pd.DataFrame]): The pandas DataFrame or a list of pandas
            DataFrames to be visualized.
        names (List[str]): The name of each DataFrame to be visualized.
        grid_size (int): The number of DataFrames per row in the visualization figure.
        save (str): The path where the visualization should be saved.
        show (bool): Whether to show the visualization or not.
    """
    # configure graph
    df = [df] if isinstance(df, pd.DataFrame) else df
    names = [f"DF{_}" for _ in range(len(df))] if names is None else names
    sensors = [sensors] if isinstance(sensors, str) else sensors
    num = len(sensors)
    num_rows = math.ceil(num / grid_size)
    num_cols = grid_size if grid_size / num < 1 else num
    assert len(df) <= len(colors)

    # init figure
    _, axs = plt.subplots(
        num_rows, num_cols, sharex="col", figsize=(grid_size * num_cols, grid_size * num_rows)
    )

    # ensure axis to be 2 dimensional
    if num_rows == 1:
        axs = [axs]

    # iterate over each sensor to plot
    for i, sensor_test in enumerate(sensors):
        axs[i // num_cols][i % num_cols].title.set_text(sensor_test.split(" ")[0])
        axs[i // num_cols][i % num_cols].set_xlabel("Seconds")
        axs[i // num_cols][i % num_cols].set_ylabel(sensor_test.split(" ")[-1])
        for j, _df in enumerate(df):
            axs[i // num_cols][i % num_cols].plot(
                _df[sensor_test], label=names[j], color=colors[j], alpha=0.4
            )

    # plot the legends to the lower right corner
    handles, labels = axs[0][0].get_legend_handles_labels()
    plt.legend(
        handles, labels, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0
    )

    if save:
        plt.savefig(save, bbox_inches="tight")

    if show:
        plt.show()
