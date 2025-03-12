"""Dataflux/data/utils.py"""
import unittest
import pandas as pd
import numpy as np

from Dataflux.helper.dataframe_ops import trim, split, multi_split

class TestMachineSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.key = "trim_key"
        self.lower = 1
        self.upper = 1

    def test_trim_empty(self) -> None:
        """Tests trim on empty dataset"""
        df = pd.DataFrame({})
        df = trim(df, self.key, self.lower, lower=True)
        self.assertTrue(df.empty, "DataFrame should be empty")

        df = trim(df, self.key, self.upper, lower=False)
        self.assertTrue(df.empty, "DataFrame should be empty")
    
    def test_trim_lower(self) -> None:
        """Tests lower trim removing the first values."""
        series = np.ones(500)
        series[:50] = 0
        series[-50:] = 0
        df = pd.DataFrame({self.key: series})

        df = trim(df, self.key, self.lower, lower=True)
        self.assertEqual(len(df), 450, f"Length {len(df)} should be 450.")
    
    def test_trim_lower_no_effect(self) -> None:
        """Tests lower trim removing no values."""
        series = np.ones(500)
        df = pd.DataFrame({self.key: series})

        df = trim(df, self.key, self.lower, lower=True)
        self.assertEqual(len(df), 500, f"Length {len(df)} should be 500.")

    def test_trim_upper(self) -> None:
        """Tests upper trim removing the first values."""
        series = np.ones(500)
        series[:50] = 2
        series[-50:] = 2
        df = pd.DataFrame({self.key: series})

        df = trim(df, self.key, self.upper, False)
        self.assertEqual(len(df), 450, f"Length {len(df)} should be 450.")
    
    def test_trim_upper_no_effect(self) -> None:
        """Tests upper trim removing no values."""
        series = np.ones(500)
        df = pd.DataFrame({self.key: series})

        df = trim(df, self.key, self.upper, False)
        self.assertEqual(len(df), 500, f"Length {len(df)} should be 500.")

    def test_split_lower(self) -> None:
        """This method tests the split function."""
        series = np.ones(500)
        series[200:250] = 0
        df = pd.DataFrame({self.key: series})

        df1, df2 = split(df, self.key, self.lower, lower=True)

        self.assertEqual(len(df1), 200, f"Length {len(df1)} should be 200")
        self.assertEqual(df1[self.key].mean(), 1, "DataFrame should only contain ones.")
        self.assertEqual(len(df2), 300, f"Length {len(df2)} should be 250")

        df2 = trim(df2, self.key, self.lower, lower=True)
        self.assertEqual(df2[self.key].mean(), 1, "DataFrame should only contain ones.")
    
    def test_split_upper(self) -> None:
        """This method tests the split function."""
        series = np.ones(500)
        series[200:250] = 2
        df = pd.DataFrame({self.key: series})

        df1, df2 = split(df, self.key, self.upper, lower=False)

        self.assertEqual(len(df1), 200, f"Length {len(df1)} should be 200")
        self.assertEqual(df1[self.key].mean(), 1, "DataFrame should only contain ones.")
        self.assertEqual(len(df2), 300, f"Length {len(df2)} should be 250")

        df2 = trim(df2, self.key, self.upper, lower=False)
        self.assertEqual(df2[self.key].mean(), 1, "DataFrame should only contain ones.")

    def test_empty_split(self) -> None:
        """Test trim on empty dataset"""
        df = pd.DataFrame({self.key: []})

        df1, df2 = split(df, self.key, self.lower, True)
        self.assertTrue(df1.empty and df2.empty, "Expected split of empty DF to be two empty DFs.")

        df1, df2 = split(df, self.key, self.upper, False)
        self.assertTrue(df1.empty and df2.empty, "Expected split of empty DF to be two empty DFs.")

    def test_split_to_empty(self) -> None:
        """Test trim on empty dataset"""
        series = np.zeros(500)
        df = pd.DataFrame({self.key: series})

        df1, df2 = split(df, self.key, self.lower, True)
        self.assertTrue(df1.empty and df2.empty, "Expected split of empty DF to be two empty DFs.")

        series = np.ones(500) * 2
        df = pd.DataFrame({self.key: series})

        df1, df2 = split(df, self.key, self.upper, False)
        self.assertTrue(df1.empty and df2.empty, "Expected split of empty DF to be two empty DFs.")

    def test_multi_split(self) -> None:
        """This method test the split algorithm."""
        series = np.ones(500)
        # 0 - 50
        series[50:60] = 2
        # 60 - 100
        series[100:110] = 0
        # 110 - 200
        series[200:205] = 2
        series[205:210] = 0
        # 210 - 300
        series[300:310] = 2
        # 310 - 450
        series[450:] = 2
        series[480:] = 0
        df = pd.DataFrame({self.key: series})

        dfs = multi_split(df, self.key, self.lower, self.upper)
        self.assertEqual(len(dfs), 5, f"Expected multi-split to output 4 but got {len(dfs)} DFs.")

        for df in dfs:
            self.assertEqual(df[self.key].mean(), 1, "DataFrame should only contain ones.")

    def test_empty_mutli_split(self) -> None:
        """Test trim on empty dataset"""
        df = pd.DataFrame({self.key: []})
        dfs = multi_split(df, self.key, self.lower, self.upper)
        self.assertEqual(len(dfs), 0, "Expected multi split of empty DF to be an empy list.")
