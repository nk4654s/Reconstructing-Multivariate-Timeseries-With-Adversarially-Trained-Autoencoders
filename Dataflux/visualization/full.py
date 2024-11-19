"""This module handles showing and saving graphs of a dataframe."""
import math
from typing import List, Tuple
import matplotlib.pyplot as plt
import pandas as pd

colors = ["teal", "orange", "blue", "red", "purple", "brown", "pink", "olive", "cyan"]


def show_dataframe(
    df: pd.DataFrame | List[pd.DataFrame],
    names: List[str] = [],
    grid_size: int = 4,
    y_label: str = "",
    y_limit: Tuple[int, int] | None = None,
    save: str = "",
    show: bool = False,
):
    """Shows and/or saves the visualization of one or multiple pandas DataFrames. In total, 9
    concurrent DataFrames can be visualized in one figure.

    Args:
        df (pd.DataFrame | List[pd.DataFrame]): The pandas DataFrame or a list of pandas
            DataFrames to be visualized.
        names (List[str]): The name of each DataFrame to be visualized.
        grid_size (int): The number of DataFrames per row in the visualization figure.
        y_label (str): The label of the y-axis. Defaults to None, uses the unit of the label.
        y_limit (Tuple): Lowest and highest number of values to be shown.
        save (str): The path where the visualization should be saved.
        show (bool): Whether to show the visualization or not.
    """
    # configure graph
    df = [df] if isinstance(df, pd.DataFrame) else df
    df = [d.reset_index() for d in df]
    names = [f"DF{_}" for _ in range(len(df))] if len(names) == 0 else names
    sensors = [c for c in df[0].columns.values]

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
        if y_label == "":
            axs[i // num_cols][i % num_cols].set_ylabel(sensor_test.split(" ")[-1])
        else:
            axs[i // num_cols][i % num_cols].set_ylabel(y_label)
        for j, _df in enumerate(df):
            axs[i // num_cols][i % num_cols].plot(
                _df[sensor_test], label=names[j], color=colors[j], alpha=0.4
            )

    # plot the legends to the lower right corner
    handles, labels = axs[0][0].get_legend_handles_labels()
    plt.legend(
        handles, labels, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0
    )

    if y_limit:
        plt.setp(axs, ylim=y_limit)

    if save:
        plt.savefig(save, bbox_inches="tight")

    if show:
        plt.show()
