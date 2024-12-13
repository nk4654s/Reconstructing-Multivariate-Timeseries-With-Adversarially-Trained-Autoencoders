"""Dataflux/data/utils.py"""
import unittest
import pandas as pd
import numpy as np

from Dataflux.modules.preprocess.machine_state import trim, split, multi_split

class TestMachineSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.key = "trim_key"
        self.state = True
        series = np.ones(1_000)
        series[:50] = 0
        series[450:550] = 0
        series[750:800] = 0
        series[801] = 0
        series[990] = 0
        self.df = pd.DataFrame({self.key: series})

    def test_trim(self) -> None:
        """This method tests the machine_state.trim() method, which removes
        a given state at the start from given data."""
        df = trim(self.df, self.key, self.state)
        self.assertTrue(len(df) == 950, f"Length {len(df)} should be 950")

        df = trim(self.df * -1 + 1, self.key, not self.state)
        self.assertTrue(len(df) == 950, f"Length {len(df)} should be 950")

    def test_split(self) -> None:
        """This method tests the split function."""
        df = trim(self.df, self.key, self.state)
        df1, df2 = split(df, self.key, True)

        self.assertTrue(len(df1) == 400, f"Length {len(df1)} should be 400")
        self.assertTrue(df1[self.key].mean() == 1, "DataFrame should only contain ones.")
        self.assertTrue(len(df2) == 550, f"Length {len(df2)} should be 450")

    def test_multi_split(self) -> None:
        """This method test the split algorithm."""
        dfs = multi_split(self.df, self.key, self.state)
        self.assertEqual(len(dfs), 4, f"Expected multi-split to output 4 but got {len(dfs)} DFs.")

        for df in dfs:
            self.assertTrue(df[self.key].mean() == 1, "DataFrame should only contain ones.")
        self.assertEqual(len(dfs[-1]), 188, f"Expected last DF to have length 188, got {len(dfs[-1])}")

    def test_empty_trim(self) -> None:
        """Test trim on empty dataset"""
        df = pd.DataFrame({self.key: []})
        df = trim(df, self.key, self.state)
        self.assertTrue(df.empty, "Expected trim of empty DF to be an empty DF.")

    def test_empty_split(self) -> None:
        """Test trim on empty dataset"""
        df = pd.DataFrame({self.key: []})
        df1, df2 = split(df, self.key, self.state)
        self.assertTrue(df1.empty and df2.empty, "Expected split of empty DF to be two empty DFs.")

    def test_empty_mutli_split(self) -> None:
        """Test trim on empty dataset"""
        df = pd.DataFrame({self.key: []})
        dfs = multi_split(df, self.key, self.state)
        self.assertEqual(len(dfs), 0, "Expected multi split of empty DF to be an empy list.")
