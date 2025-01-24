"""Dataflux/helper/scaler.py"""

import unittest

import numpy as np
import pandas as pd

from Dataflux.data.update_tools import merge_dfs


class TestMachineSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.df_a = pd.DataFrame({"ts": range(5), "A": range(5)}).set_index(keys=["ts"])
        self.df_b = pd.DataFrame({"ts": range(5, 10), "A": range(5, 10)}).set_index(
            keys=["ts"]
        )

        self.df_c = pd.DataFrame({"ts": range(10), "A": range(10)}).set_index(
            keys=["ts"]
        )

    def test_no_ts_index(self):
        df_a = self.df_a.reset_index()
        df_a.drop(columns=["ts"], inplace=True)

        df_b = self.df_b.reset_index()
        df_b.drop(columns=["ts"], inplace=True)

        def fail():
            merge_dfs(df_a, df_b)

        self.assertRaises(KeyError, fail)

    def test_merge_appending_df(self):
        df = merge_dfs(self.df_a, self.df_b)

        self.assertTrue((df == self.df_c).all().all())

    def test_merge_empty_row_inbetween(self):
        df_b = self.df_b.copy()
        df_c = self.df_c.copy()

        df_b = df_b.drop([5])
        df_c = df_c.drop([5])

        df = merge_dfs(self.df_a, df_b)

        self.assertTrue((df == df_c).all().all())

    def test_merge_overlapping_rows(self):
        df_b = self.df_b.copy()
        df_c = self.df_c.copy()

        df_b.loc[4] = 10
        df_b.sort_index()

        df_c.iloc[4] = 10

        df = merge_dfs(self.df_a, df_b)

        self.assertTrue((df == df_c).all().all())

    def test_merge_try_adding_nans(self):
        df_b = self.df_b.copy()
        df_c = self.df_c.copy()

        df_b["A"] = np.nan
        df_c = df_c.drop(range(5, 10))

        df = merge_dfs(self.df_a, df_b)
        df = df.dropna(how="any")

        self.assertTrue((df == df_c).all().all())

    def test_merge_with_unaffected_column(self):
        df_a = pd.DataFrame({"ts": range(10), "A": range(10)}).set_index(keys=["ts"])
        df_c = self.df_c.copy()

        df_a.loc[5:, "A"] = np.nan
        df_a["B"] = [i + 10 for i in range(10)]
        df_c["B"] = df_a["B"]

        df = merge_dfs(df_a, self.df_b)

        self.assertTrue((df["A"] == df_c["A"]).all())
        self.assertTrue((df["B"] == df_c["B"]).all())
        self.assertTrue((df.index == df_c.index).all())

    def test_merge_with_malformed_header(self):
        df_a = pd.DataFrame(
            {"ts": range(10), "A": range(10), "A_x": range(10)}
        ).set_index(keys=["ts"])
        df_c = self.df_c.copy()

        df_a.loc[5:, "A"] = np.nan
        df_a["B"] = [i + 10 for i in range(10)]
        df_c["B"] = df_a["B"]

        def fail():
            merge_dfs(df_a, self.df_b)

        self.assertRaises(pd.errors.MergeError, fail)
