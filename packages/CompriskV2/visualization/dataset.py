import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def visualize_dataset(sensors: pd.DataFrame, units: dict[str, tuple], cols: int = 3) -> None:
    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(sensors.index):
        sensors = sensors.copy()
        sensors.index = pd.to_datetime(sensors.index, unit="s")

    features = sensors.shape[-1]
    rows = max(1, (features + cols - 1) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    axs = axs.flatten()

    for idx, col in enumerate(sensors.columns):
        ax = axs[idx]
        ax.plot(sensors.index, sensors[col], color="teal", linewidth=0.8)

        unit, label = units.get(col, (col, ""))
        ax.set_title(label)
        ax.set_ylabel(unit)
        ax.set_xlim(sensors.index.min(), sensors.index.max())

        data = sensors[col]
        ax.set_ylim(data.min() - 0.05 * abs(data.min()), data.max() + 0.05 * abs(data.max()))

        # Format x-axis as dates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    # Hide unused axes if any
    for idx in range(features, len(axs)):
        fig.delaxes(axs[idx])

    # Auto format x-axis labels to prevent overlap
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    plt.show()
