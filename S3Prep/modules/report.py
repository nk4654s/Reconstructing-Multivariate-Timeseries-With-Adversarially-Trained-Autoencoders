"""This module logs some report about a downloaded dataset."""

import os
from typing import Dict, List, Tuple
import json
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import BasePreProcessor
from S3Prep.modules.module import logger


rcParams.update({"figure.autolayout": True})


class Report(BasePreProcessor):
    """This class logs some report about a downloaded
    dataframe.
    """

    def __init__(self, folder: str = "report"):
        super().__init__()

        self._folder = folder
        self._meta = {}

        self._sum_correlation_matrix = None

    def _check_file_path(self, file_path: str) -> str:
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)

        file_path = self._folder + os.path.sep + file_path
        temp_path = file_path
        num = 1
        while os.path.exists(temp_path):
            temp_path = file_path.replace(".", f"_{num}.")
            num += 1

        return temp_path

    def _highest_correlated_features(self, df: DataFrame) -> Tuple[str, str]:
        # get the feature with the highest (avg) correlation
        corr_idx = np.argmax(df.mean(axis=0))
        corr_name = df.columns.values[corr_idx]

        # get the highest correlated matching feature
        corr_vec = df[corr_name]
        corr_vec.iloc[corr_idx] = 0

        corr_max = corr_vec.index[np.argmax(corr_vec)]
        return corr_name, corr_max

    def _lowest_correlated_features(self, df: DataFrame) -> str:
        # get the feature with the lowest (avg) correlation
        corr_idx = np.argmin(df.mean(axis=0))
        corr_name = df.columns.values[corr_idx]

        return corr_name

    def _visualize_series(
        self, df: DataFrame, col_name: str, avg_window_size: int = 30, path: str = None
    ) -> None:
        # Get the column series
        series = df[col_name]

        # 1. Rolling average and trendline plot
        rolling_window = avg_window_size
        rolling_avg = series.rolling(window=rolling_window).mean()
        trendline = series.expanding(min_periods=rolling_window).mean()

        _, ax = plt.subplots(figsize=(10, 6))
        ax.plot(series.to_numpy(), label="Original")
        ax.plot(
            rolling_avg.to_numpy(), label=f"Rolling Average (window={rolling_window})"
        )
        ax.plot(trendline.to_numpy(), label="Trendline")
        ax.set_title(
            f"Rolling Average and Trendline: {col_name} (window={rolling_window})"
        )
        ax.set_xlabel("Index")
        ax.set_ylabel(col_name)
        ax.grid(True)
        plt.legend()
        plt.savefig(self._check_file_path("trend_" + path))

        # 2. Boxplot graph
        _, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(series.values)
        ax.set_title(f"Boxplot: {col_name}")
        ax.set_xlabel("")
        ax.set_ylabel(col_name)
        ax.grid(True)

        plt.savefig(self._check_file_path("boxplot_" + path))

    def _visualize_avg_corr(self, cmat: DataFrame, path: str = None) -> None:
        cmat = cmat.copy().fillna(0)
        vals = zip(cmat.columns.values, np.mean(cmat, axis=0), np.std(cmat, axis=0))
        vals = sorted(list(vals), key=lambda x: (x[1], x[2], x[0]), reverse=True)

        y, x, yerr = [], [], []
        for _x, _y, _yerr in vals:
            if _y == 0.0:
                continue
            y.append(_y)
            x.append(_x)
            yerr.append(_yerr)

        _, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x, y)
        ax.errorbar(x, y, yerr=yerr, fmt=".")

        ax.set_title("Average Correlation")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Correlation (Avg.)")

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
        plt.savefig(self._check_file_path(path))

    def _visualize_corr_matrix(self, cmat: DataFrame, path: str = None) -> None:
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(cmat, cmap="coolwarm", interpolation="nearest")

        ax.set_title("Correlation Matrix")
        ax.set_xticks(range(len(cmat.columns.values)))
        ax.set_xticklabels(cmat.columns.values)
        ax.set_yticks(range(len(cmat.columns.values)))
        ax.set_yticklabels(cmat.columns.values)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        plt.savefig(self._check_file_path(path))

    def _clean_corr_matrix(self, counts: int) -> DataFrame:
        # remove the self-correlations
        corr_matrix = self._sum_correlation_matrix / counts
        eye = np.eye(corr_matrix.shape[0], corr_matrix.shape[0])
        corr_matrix[eye == 1] = 0

        # only include rows/cols which have some correlation, a perfect 0
        # means, that the previous correlation matrix were empty
        idx, jdx = [], []
        for i, j in enumerate(corr_matrix.columns.values):
            if sum(corr_matrix[j]) != 0:
                jdx.append(j)
                idx.append(i)

        corr_matrix = corr_matrix.iloc[idx]
        corr_matrix = corr_matrix[jdx]

        # add the self-correlation again
        eye = np.eye(corr_matrix.shape[0], corr_matrix.shape[1])
        corr_matrix[eye == 1] = 1
        return corr_matrix

    def on_processing_started(
        self, df: DataFrame, metadata: Dict
    ) -> DataFrame | List[DataFrame]:
        logger.debug("Processing ON_PROCESSING_STARTED in %s", self.__class__.__name__)

        # Calculate correlation matrix
        corr_matrix = abs(df.corr(method="spearman").fillna(0))
        if self._sum_correlation_matrix is None:
            self._sum_correlation_matrix = corr_matrix
        else:
            headers = self._sum_correlation_matrix.columns.values
            self._sum_correlation_matrix += corr_matrix[headers]

        # save the length of the dataset
        day = metadata.get("date", None)
        if not day:
            logger.warning(
                "No date found in metadata, plot 'feature_per_day.png' might be wrong."
            )
            return df

        # find the next empty key for the current date
        if df.shape[0] > 0:
            meta = self._meta.get("files", [])
            num = 0
            date = day
            while len(meta) > 0 and meta[-1]["day"] == date:
                num += 1
                date = day + f"_{num}"

            meta.append(
                {
                    "day": date,
                    "recordings": df.shape[0],
                    "features": len(df.columns.values),
                }
            )
            self._meta["files"] = meta
            total = self._meta.get("total", 0)
            total += df.shape[0]
            self._meta["total"] = total

        return df

    def on_processing_finished(self, metadata: Dict) -> DataFrame | List[DataFrame]:
        logger.debug("Processing ON_PROCESSING_FINISHED in %s", self.__class__.__name__)

        # save the plot of num_features per day, from data gathered before
        num_features = [x["features"] for x in self._meta["files"]]
        days = [x["day"] for x in self._meta["files"]]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Number of Features per Day")
        ax.plot(days, num_features)
        ax.set_xlabel("Day")
        ax.set_ylabel("# of features")

        x_locs = np.arange(0, len(days), max(len(days) // 2, len(days) // 50))
        x_labels = [days[i] for i in x_locs]
        ax.set_xticks(x_locs)
        ax.set_xticklabels(x_labels)

        plt.savefig(self._check_file_path("features_per_day.png"))

        # plot the dataset length per file downloaded
        num_recordings = [x["recordings"] for x in self._meta["files"]]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Number of Recordings per File")
        ax.plot(days, num_recordings)
        ax.set_xlabel("File (latest to newest)")
        ax.set_ylabel("# of Recordings")

        x_locs = np.arange(0, len(days), max(len(days) // 2, len(days) // 50))
        x_labels = [days[i] for i in x_locs]
        ax.set_xticks(x_locs)
        ax.set_xticklabels(x_labels)

        plt.savefig(self._check_file_path("recordings_per_file.png"))

        # plot the average of the correlation matrix, clean 0-only rows and cols
        corr_matrix = self._clean_corr_matrix(len(days))
        self._visualize_corr_matrix(corr_matrix, "correlation_matrix.png")
        self._visualize_avg_corr(corr_matrix, "avg_feature_correlation.png")

        # save the best correlated sensor names
        corr_a, corr_b = self._highest_correlated_features(corr_matrix)
        uncorr = self._lowest_correlated_features(corr_matrix)

        meta = {"highest_correlated": [corr_a, corr_b], "most_uncorrelated": uncorr}
        with open(
            self._check_file_path("correlated_sensors.json"),
            "w",
            encoding="iso-8859-15",
        ) as f:
            json.dump(meta, f, indent=2)

        # save the length of each dataset file and the total dataset length
        with open(self._check_file_path("metadata.json"), "w", encoding="UTF-8") as f:
            json.dump(self._meta, f, indent=2)
