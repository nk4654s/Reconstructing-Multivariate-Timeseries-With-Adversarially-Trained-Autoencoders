"""This module contains the thread class to normalize dataset given a folder of files."""

import threading
from typing import List, Tuple
import pandas as pd


class MinMaxFileThread(threading.Thread):
    """This class is used to find the minimum and maximum values of a DataFrame in a file."""

    def __init__(self, file_paths, _min=None, _max=None):
        threading.Thread.__init__(self)
        self.file_paths = file_paths
        self._min, self._max = _min, _max
        self._fit = True if _min is not None else False

    @property
    def result(self):
        """Returns the min/max values"""
        return self._min, self._max

    def run(self):
        for file_path in self.file_paths:
            # Load the file into a pandas DataFrame
            df = pd.read_hdf(file_path, key="data")

            # Calculate minimum and maximum per column
            if not self._fit:
                df_min = df.min()
                if self._min is None:
                    self._min = df_min
                else:
                    self._min[df_min < self._min] = df_min

                df_max = df.max()
                if self._max is None:
                    self._max = df_max
                else:
                    self._max[df_max > self._max] = df_max
            else:
                df = (df - self._min).div(self._max - self._min)
                df.to_hdf(file_path, key="norm")

    @staticmethod
    def get_file_chunks(file_paths: List[str], num_threads: int = 2) -> List[List[str]]:
        """Calculates and returns the chunks of file paths to be processed by each thread.
        
        Args:
            file_paths (List[str]): List of file paths to be processed by each thread.
            num_threads (int): Number of threads to be used for processing the file paths.
        
        Returns:
            List[List[str]]: A list of lists containing the file paths to be processed by each
            thread.
        """
        if len(file_paths) <= num_threads:
            return [file_paths]

        chunks = []
        for i in range(0, len(file_paths), len(file_paths) // num_threads):
            chunks.append(file_paths[i : i + len(file_paths) // num_threads])
        return chunks

    @staticmethod
    def fit_on_files(
        file_paths: List[str], num_threads: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This method finds the minimum and maximum values of each column in the given files.

        Args:
            file_paths (List[str]): List of file paths to be processed by each thread.
            num_threads (int): Number of threads to be used. Defaults to 2.

        Return:
            Tuple[pd.DataFrame, pd.DataFrame]: Returns the minimum and maximum values of each
            column.
        """
        threads = []
        results = []
        file_chunks = MinMaxFileThread.get_file_chunks(file_paths, num_threads)

        # start a thread on each file chunk
        for chunk in file_chunks:
            thread = MinMaxFileThread(chunk)
            thread.start()
            threads.append(thread)

        # wait for all threads to finish and gather theirs results
        for thread in threads:
            thread.join()
            results.append(thread.result)

        # calculate the final result across all thread results
        _min, _max = results[0]
        for _n_min, _n_max in results[1:]:
            _min[_n_min < _min] = _n_min
            _max[_n_max > _max] = _n_max

        return _min, _max

    @staticmethod
    def transform_on_files(
        file_paths: List[str],
        train_min: pd.DataFrame,
        train_max: pd.DataFrame,
        num_threads: int = 2,
    ) -> None:
        """This method normalizes the DataFrames using the min/max method. The normalized
        DataFrame will be stored in the original file under the key-name "norm". Load it
        by using `pd.read_hdf(file_name, key="norm")`.

        Args:
            file_paths (List[str]): List of file paths to be processed by each thread.
            train_min (pd.DataFrame): Minimum values of each column in the training data.
            train_max (pd.DataFrame): Maximum values of each column in the training data.
            num_threads (int): Number of threads to be used. Defaults to 2.
        """
        threads = []
        file_chunks = MinMaxFileThread.get_file_chunks(file_paths, num_threads)

        # start a thread on each file chunk
        for chunk in file_chunks:
            thread = MinMaxFileThread(chunk, train_min, train_max)
            thread.start()
            threads.append(thread)

        # wait for all threads to finish
        for thread in threads:
            thread.join()
