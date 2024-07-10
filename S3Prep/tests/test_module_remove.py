"""Test modules/remove.py"""

import unittest
from pandas import DataFrame
from S3Prep.modules.remove import RemoveColumns

class TestRemoveColumns(unittest.TestCase):
    """This class contains unit tests for modules/remove.py
    """
    def test_remove_columns(self):
        """This method tests the remove columns method
        """
        # Create a sample dataframe with columns A, B, C, and D
        df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'D': [10, 11, 12]})

        # Create an instance of the RemoveColumns class
        remove_cols = RemoveColumns(columns_to_remove=['A', 'C'])

        # Run the on_download_finished method with our sample dataframe
        result = remove_cols.on_download_finished(df, None)

        # Check if the resulting dataframe has only columns B and D
        self.assertEqual(result.shape[1], 2)
        self.assertTrue('B' in result.columns)
        self.assertTrue('D' in result.columns)

if __name__ == '__main__':
    unittest.main()
