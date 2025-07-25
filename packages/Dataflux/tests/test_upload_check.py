import unittest
import uuid

import numpy as np
import pandas as pd
from Dataflux.data import DataUpdater

A = str(uuid.uuid4())
B = str(uuid.uuid4())
C = str(uuid.uuid4())


class TestCheckBeforeUpload(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({A: [1, 2], B: [3, 4]})
        self.previous_size = 2
        self.previous_columns = {A: 3}

    def test_smaller_df(self):
        # smaller df should always be False
        self.df = pd.DataFrame({A: [1], B: [3]})
        result = DataUpdater.check(self.df, self.previous_size, self.previous_columns)
        self.assertEqual(False, result)

    def test_smaller_df_additional_column(self):
        # smaller df should always be False
        self.df = pd.DataFrame({A: [1], B: [3], C: [6]})
        result = DataUpdater.check(self.df, self.previous_size, self.previous_columns)
        self.assertEqual(False, result)

    def test_greater_df(self):
        # greater df should be False if previous columns were modified
        self.df = pd.DataFrame({A: [1, 2, 3], B: [3, 4, 5]})
        result = DataUpdater.check(self.df, self.previous_size, self.previous_columns)
        self.assertEqual(False, result)

    def test_greater_df_additional_column(self):
        # greater df should be True if previous columns were not modified
        self.df = pd.DataFrame({A: [1, np.nan, 2], B: [3, np.nan, 4], C: [6, 7, 8]})
        result = DataUpdater.check(self.df, self.previous_size, self.previous_columns)
        self.assertEqual(True, result)

    def test_less_columns(self):
        # should always be False
        self.df = pd.DataFrame({A: [1, 3]})
        result = DataUpdater.check(self.df, len(self.df), self.previous_columns)
        self.assertEqual(False, result)

    def test_invalid_uuid(self):
        self.df = pd.DataFrame({"Not a UUID": [1]})
        result = DataUpdater.check(self.df, len(self.df), self.previous_columns)
        self.assertEqual(False, result)


if __name__ == "__main__":
    unittest.main()
