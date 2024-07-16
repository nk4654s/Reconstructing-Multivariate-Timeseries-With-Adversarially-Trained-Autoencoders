"""This module contains the thread class to standardize dataset given a folder of files."""

import threading
from typing import List, Tuple
import pandas as pd


class StandardizeFileThread(threading.Thread):
    """This class is used to find the mean and std values of a DataFrame in a file."""

    def __init__(self, file_paths, mean=None, std=None):
        threading.Thread.__init__(self)
        self.file_paths = file_paths
        self._mean, self._std = mean, std
        self._fit = True if mean is not None else False

    @property
    def result(self):
        """Returns the mean/std/iterations values"""
        return self._mean, self._std, len(self.file_paths)

    def run(self):
        for file_path in self.file_paths:
            # Load the file into a pandas DataFrame
            df = pd.read_hdf(file_path, key="data")

            # Calculate minimum and maximum per column
            if not self._fit:
                if self._mean is None:
                    self._mean = df.mean()
                else:
                    self._mean += df.mean()

                if self._std is None:
                    self._std = df.std()
                else:
                    self._std += df.std()
            else:
                df = (df - self._mean).div(self._std)
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
        """This method finds the mean and std values of each column in the given files.

        Args:
            file_paths (List[str]): List of file paths to be processed by each thread.
            num_threads (int): Number of threads to be used. Defaults to 2.

        Return:
            Tuple[pd.DataFrame, pd.DataFrame]: Returns the mean and std values of each
            column.
        """
        threads = []
        results = []
        file_chunks = StandardizeFileThread.get_file_chunks(file_paths, num_threads)

        # start a thread on each file chunk
        for chunk in file_chunks:
            thread = StandardizeFileThread(chunk)
            thread.start()
            threads.append(thread)

        # wait for all threads to finish and gather theirs results
        for thread in threads:
            thread.join()
            results.append(thread.result)

        # calculate the final result across all thread results
        mean, std, num = results[0]
        for _mean, _std, _num in results[1:]:
            mean += _mean
            std += _std
            num += _num

        return mean.div(num), std.div(num)

    @staticmethod
    def transform_on_files(
        file_paths: List[str],
        train_mean: pd.DataFrame,
        train_std: pd.DataFrame,
        num_threads: int = 2,
    ) -> None:
        """This method standardizes the DataFrames using the mean/std method. The standardized
        DataFrame will be stored in the original file under the key-name "norm". Load it
        by using `pd.read_hdf(file_name, key="norm")`.

        Args:
            file_paths (List[str]): List of file paths to be processed by each thread.
            train_mean (pd.DataFrame): Mean values of each column in the training data.
            train_std (pd.DataFrame): Std values of each column in the training data.
            num_threads (int): Number of threads to be used. Defaults to 2.
        """
        threads = []
        file_chunks = StandardizeFileThread.get_file_chunks(file_paths, num_threads)

        # start a thread on each file chunk
        for chunk in file_chunks:
            thread = StandardizeFileThread(chunk, train_mean, train_std)
            thread.start()
            threads.append(thread)

        # wait for all threads to finish
        for thread in threads:
            thread.join()
