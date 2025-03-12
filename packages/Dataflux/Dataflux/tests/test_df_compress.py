import unittest

import numpy as np
import pandas as pd
from Dataflux.helper import compress, decompress


class TestCompressDecompress(unittest.TestCase):
    """
    This test case tests the compression and decompression of DataFrames found
    in the Updater class.
    """
    def test_compress_decompress_identity(self):
        """
        This method simply test if an increasing sequence can be compressed and
        decompressed without error.
        """
        data = pd.DataFrame({
            "ts": range(1, 11),
            "A": range(10)
        }, dtype=float)
        data = data.set_index("ts")

        compressed_data = compress(data)
        decompressed_data = decompress(compressed_data, fill=True)

        self.assertTrue(decompressed_data.equals(data))

    def test_compress_random_dataset(self):
        """
        This method tests the compression and decompression of a random dataset
        of 10,000 rows. In addition, multiple columns are tested as well.
        """
        np.random.seed(0)
        num_rows = 10_000
        data = pd.DataFrame({
            "ts": range(1, num_rows + 1),
            "A": np.random.randint(1, 100, size=num_rows),
            "B": np.random.randint(1, 100, size=num_rows),
            "C": np.random.randint(1, 100, size=num_rows),
            "D": np.random.randint(1, 100, size=num_rows)
        }, dtype=float)
        data = data.set_index("ts")

        compressed_data = compress(data)
        decompressed_data = decompress(compressed_data, fill=True)

        self.assertTrue(decompressed_data.equals(data))

    def test_compress_decompress_with_duplicates(self):
        """
        This method tests the compression and decompression of a dataset with
        duplicate values in a column.
        """
        duplicates = [2, 7, 5, 6, 5, 5, 5]
        data = pd.DataFrame({
            "ts": range(len(duplicates)),
            "A": duplicates
        }, dtype=float)
        data = data.set_index("ts")

        compressed_data = compress(data)
        decompressed_data = decompress(compressed_data, fill=True)

        self.assertGreater(len(data), len(compressed_data))
        self.assertTrue(decompressed_data.equals(data))

    def test_compress_decompress_with_empty_dataframe(self):
        """
        This method tests the compression and decompression of an empty dataset.
        """
        data = pd.DataFrame({"ts": [], "A": []}, dtype=float)
        data = data.set_index("ts")

        compressed_data = compress(data)
        decompressed_data = decompress(compressed_data, fill=True)

        self.assertTrue(decompressed_data.empty)

    def test_compress_decompress_with_pair(self):
        """
        This method tests the compression and decompression of a dataset with a
        single pair.
        """
        data = pd.DataFrame({"ts": [1, 2], "A": [1, 1]}, dtype=float)
        data = data.set_index("ts")

        compressed_data = compress(data)
        decompressed_data = decompress(compressed_data, fill=True)

        self.assertTrue(decompressed_data.equals(data))

    def test_compress_decompress_with_zeros(self):
        """
        This method tests the compression and decompression of a dataset with a
        single pair.
        """
        data = pd.DataFrame({"ts": [1, 2, 3, 4, 5, 6, 7, 8], "A": [1, 0, 0, 0, 1, 0, 0, 1]}, dtype=float)
        data = data.set_index("ts")

        compressed_data = compress(data)
        decompressed_data = decompress(compressed_data, fill=True)

        self.assertTrue(decompressed_data.equals(data))

if __name__ == "__main__":
    unittest.main()
