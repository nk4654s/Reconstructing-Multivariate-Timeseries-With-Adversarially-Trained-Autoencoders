"""This module logs some report about a downloaded dataset."""

from typing import List, Tuple
import logging
import os
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from S3Prep.modules import BasePreProcessor
from S3Prep.helper import Metadata, Meta, get_file, get_logger

# create logger object
logger = get_logger(__name__)
rcParams.update({"figure.autolayout": True})


class Report(BasePreProcessor):
    """This class logs some report about a downloaded
    dataframe.
    """

    def __init__(self, sub_folder: str = "report"):
        super().__init__()

        self._sub_folder = sub_folder
        self._folder = None
        self._file_register = []
        self._total_recordings = 0

        self._counts = 0
        self._sum_correlation_matrix = None

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
        plt.savefig(get_file(f"{self._folder}/trend_{path}"))

        # 2. Boxplot graph
        _, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(series.values)
        ax.set_title(f"Boxplot: {col_name}")
        ax.set_xlabel("")
        ax.set_ylabel(col_name)
        ax.grid(True)

        plt.savefig(get_file(f"{self._folder}/boxplot_{path}"))

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
        plt.savefig(get_file(f"{self._folder}/{path}"))

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
        plt.savefig(get_file(f"{self._folder}/{path}"))

    def _clean_corr_matrix(self) -> DataFrame:
        # remove the self-correlations
        corr_matrix = self._sum_correlation_matrix / self._counts
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

    def on_download_finished(
        self, df: DataFrame, metadata: Metadata
    ) -> DataFrame | List[DataFrame]:
        # define the output folder
        if not self._folder:
            self._folder = os.path.join(metadata[Meta.PATH], self._sub_folder)

        # Calculate correlation matrix
        corr_matrix = abs(df.corr(method="spearman").fillna(0))
        if self._sum_correlation_matrix is None:
            self._sum_correlation_matrix = corr_matrix
        else:
            headers = self._sum_correlation_matrix.columns.values
            self._sum_correlation_matrix += corr_matrix[headers]

        # save the length of the dataset
        day = metadata[Meta.DAY]
        if not day:
            logger.warning(
                "No date found in metadata, plot 'feature_per_day.png' might be wrong."
            )
            return df

        # find the next empty key for the current date
        if df.shape[0] > 0:
            # make sure that the day-key has the increasing postfix-number according to the
            # DataFrame's file name
            num = 0
            date = day
            while (
                len(self._file_register) > 0 and self._file_register[-1]["day"] == date
            ):
                num += 1
                date = day + f"_{num}"

            # add the current DataFrame to the file register
            if metadata[Meta.MACHINE_STATE_CHANGE] or len(self._file_register) == 0:
                self._file_register.append(
                    {
                        "day": date,
                        "recordings": df.shape[0],
                        "features": len(df.columns.values),
                    }
                )
            else:
                self._file_register[-1]["recordings"] += df.shape[0]

            self._total_recordings += df.shape[0]
            self._counts += 1

        return df

    def on_processing_finished(self, metadata: Metadata) -> DataFrame | List[DataFrame]:
        # save the plot of num_features per day, from data gathered before
        num_features = [x["features"] for x in self._file_register]
        days = [x["day"] for x in self._file_register]
        logging.debug("Logging feature per day on %s days", len(days))

        _, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Number of Features per Day")
        ax.plot(days, num_features)
        ax.set_xlabel("Day")
        ax.set_ylabel("# of features")

        x_locs = np.arange(0, len(days), max(len(days) // 2, len(days) // 50))
        x_labels = [days[i] for i in x_locs]
        logging.debug("Reduced number of labels to %s", len(x_labels))
        ax.set_xticks(x_locs)
        ax.set_xticklabels(x_labels)

        plt.savefig(get_file(f"{self._folder}/features_per_day.png"))

        # plot the dataset length per file downloaded
        num_recordings = [x["recordings"] for x in self._file_register]
        logger.debug(
            "Plot the number of recordings per file (files: %s)", len(num_recordings)
        )

        _, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Number of Recordings per File")
        ax.plot(days, num_recordings)
        ax.set_xlabel("File (latest to newest)")
        ax.set_ylabel("# of Recordings")

        x_locs = np.arange(0, len(days), max(len(days) // 2, len(days) // 50))
        x_labels = [days[i] for i in x_locs]
        logging.debug("Reduced number of labels to %s", len(x_labels))
        ax.set_xticks(x_locs)
        ax.set_xticklabels(x_labels)

        plt.savefig(get_file(f"{self._folder}/recordings_per_file.png"))

        # plot the outlier removed statistics if available
        if metadata.has(Meta.ADDITIONAL_REPORT_GRAPHS):
            graphs = metadata[Meta.ADDITIONAL_REPORT_GRAPHS]
            for ax, file_name in graphs:
                ax.figure.savefig(os.path.join(self._folder, file_name))

        # plot the average of the correlation matrix, clean 0-only rows and cols
        corr_matrix = self._clean_corr_matrix()
        self._visualize_corr_matrix(corr_matrix, "correlation_matrix.png")
        self._visualize_avg_corr(corr_matrix, "avg_feature_correlation.png")

        # save the best correlated sensor names
        corr_a, corr_b = self._highest_correlated_features(corr_matrix)
        uncorr = self._lowest_correlated_features(corr_matrix)
        coi = {"highest_correlated": [corr_a, corr_b], "most_uncorrelated": uncorr}

        # update the metadata
        metadata.put(self._file_register, Meta.FILES)
        metadata.put(self._total_recordings, Meta.TOTAL_RECORDINGS)
        metadata.put(coi, Meta.CORRELATION_OF_INTEREST)

        metadata.drop(Meta.DAY)
        metadata.drop(Meta.ADDITIONAL_REPORT_GRAPHS)
