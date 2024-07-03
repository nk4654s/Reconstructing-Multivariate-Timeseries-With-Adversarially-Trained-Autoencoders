"""This module logs some report about a downloaded dataset."""
import os
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from S3Prep.modules.module import BasePreProcessor


class Report(BasePreProcessor):
    """This class logs some report about a downloaded
    dataframe.
    """
    def __init__(self, folder: str = "report"):
        super().__init__()

        self._folder = folder

        # let this module run as late as possible
        self._intermediate_run = False

    def _check_file_path(self, file_path: str) -> str:
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)

        temp_path = self._folder + os.path.sep + file_path
        num = 1
        while os.path.exists(temp_path):
            temp_path = file_path.replace(".", f"_{num}.")
            num += 1

        return temp_path

    def _correlation_matrix(self, df: DataFrame) -> DataFrame:
        # Calculate correlation matrix
        corr_matrix = abs(df.corr().dropna(how="all").dropna(how="all", axis="columns"))

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Display the correlation matrix using imshow
        im = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')

        # Set title and labels
        ax.set_title('Correlation Matrix')
        ax.set_xticks(range(len(corr_matrix.columns.values)))
        ax.set_xticklabels(corr_matrix.columns.values)
        ax.set_yticks(range(len(corr_matrix.columns.values)))
        ax.set_yticklabels(corr_matrix.columns.values)

        # Rotate tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        # Add a color bar
        fig.colorbar(im, ax=ax)
        plt.savefig(self._check_file_path("correlation.png"))
        return corr_matrix

    def _highest_correlated_features(self, df: DataFrame) -> None:
        corr_idx = np.argmax(df.mean(axis=0))
        corr_name = df.columns.values[corr_idx]

        corr_vec = df[corr_name]
        corr_vec[corr_idx] =0

        corr_max = corr_vec.index[np.argmax(corr_vec)]
        return corr_name, corr_max

    def _lowest_correlated_features(self, df: DataFrame) -> None:
        corr_idx = np.argmin(df.mean(axis=0))
        corr_name = df.columns.values[corr_idx]

        return corr_name

    def __call__(self, df: DataFrame) -> DataFrame:
        cf = self._correlation_matrix(df)

        f1, f2 = self._highest_correlated_features(cf)
        self._visualize_series(df, f1, path=f1 + ".png")
        self._visualize_series(df, f2, path=f2 + ".png")

        f = self._lowest_correlated_features(cf)
        self._visualize_series(df, f, path=f + ".png")

        return df

    def _visualize_series(self, df: DataFrame, col_name: str, avg_window_size: int = 30,
                          path: str = None) -> None:
        # Get the column series
        series = df[col_name]

        # 1. Rolling average and trendline plot
        rolling_window = avg_window_size
        rolling_avg = series.rolling(window=rolling_window).mean()
        trendline = series.expanding(min_periods=rolling_window).mean()

        _, ax = plt.subplots(figsize=(10, 6))
        ax.plot(series.to_numpy(), label='Original')
        ax.plot(rolling_avg.to_numpy(), label=f"Rolling Average (window={rolling_window})")
        ax.plot(trendline.to_numpy(), label='Trendline')
        ax.set_title(f"Rolling Average and Trendline: {col_name} (window={rolling_window})")
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
