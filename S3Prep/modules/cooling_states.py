"""This module normalizes a dataset and splits it into train, val, test dataset."""
import math
import os
from typing import List, Callable
import numpy as np
import pandas as pd
from scipy.fftpack import rfft, irfft, fftfreq
from S3Prep.modules import BasePreProcessor
from S3Prep.helper import Meta, Metadata, get_logger, FileWorkerCallable, FileWorkerThread

# create logger object
logger = get_logger(__name__)

class CoolingDetectionCallable(FileWorkerCallable):
    """This callable detects the machine on idle state and registers it
    as separate column.
    """
    def __init__(self, cols: List[str] = []) -> None:
        """Initializes this object.

        Args:
            name (str): The name if the column added.
            list (List[str]): The column names required for the detection to work.
        """
        super().__init__()

        self._names = ["KühlungAn", "KühlungAus", "KühlungAnVerlauf", "KühlungAusVerlauf"]
        self._list = cols

    @staticmethod
    def remove_frequencies(data, cut_off: float = 0.001) -> np.array:
        """Transforms the input into a Fourier Transformation and removes the
        low frequency components.

        Args:
            cut_off (float): The cut off frequency to remove.
        
        Returns:
            np.array: The transformed data.
        """
        y = data
        x = np.linspace(0.0, len(y) - 1, len(y))

        # Number of samplepoints
        f_signal = rfft(y)
        w = fftfreq(y.size, d=x[1]-x[0])

        # filter all frequencies above 0.001
        cut_f_signal = f_signal.copy()
        cut_f_signal[(w>cut_off)] = 0
        cut_signal = irfft(cut_f_signal)
        return cut_signal

    @staticmethod
    def interpolate_polynomial(x, y, deg=4) -> Callable:
        """Creates a polynomial iternpolation for the input data. The degree
        is set by the user.

        Args:
            x (np.array): The x values of the data.
            y (np.array): The y values of the data.
            deg (int): The degree of the interpolation.

        Returns:
            Callable: The interpolation function.
        """
        # coefficients of the polynomial
        c = np.polyfit(x, y, deg)

        # first and second degree coefficients
        c_st = np.array([_ for _ in range(deg, 0, -1)]) * c[:-1]
        assert c[-2] == c_st[-1]
        if deg > 1:
            c_nd = np.array([_ for _ in range(deg - 1, 0, -1)]) * c_st[:-1]
            assert c_st[-2] == c_nd[-1]

        # the callable return by this method is a polynomial of degree n
        # if the user defines `coeff=True` then the coefficients are returned
        def execute(_x, derivative=0, coeff=False):
            if derivative == 0:
                if coeff:
                    return c
                return np.vander(_x, deg + 1) @ c
            if derivative == 1:
                if coeff:
                    return c_st
                return np.vander(_x, deg) @ c_st
            if derivative == 2 and deg > 1:
                if coeff:
                    return c_nd
                return np.vander(_x, deg - 1) @ c_nd

        return execute

    @staticmethod
    def get_split_points(inter: Callable, data: np.array, deriv: int,
                         region_size: int = 3 * 60 * 60) -> np.array:
        """Splits the input data by using the second derivative of the interpolation.
        The splits are taken at the steepest ascent and descent of the interpolation
        model.

        Args:
            inter (Callable): The interpolation function.
            data (np.ndarray): The input data.
            deriv (int): The derivative of the interpolation function.
            region_size (int): The size of the region to be considered for each
        
        Returns:
            np.ndarray: The split points.
        """
        # get critical points of f' by finding roots of f''
        roots = np.array(np.round(np.roots(inter(data, deriv, coeff=True))), dtype=np.uint)
        x = np.linspace(0, int(len(data)) - 1, int(len(data)), dtype=np.uint)
        split_points = []
        for root in roots:
            # get a region around the root
            start_idx = int(root if root-region_size < 0 else root-region_size)
            end_idx = int(root if root+region_size >= len(x) else root+region_size)

            x_1 = np.array([_ for _ in range(start_idx, end_idx, 1)])
            y_1 = data[x_1]

            # calculate the interpolation value at the root
            x_2 = np.repeat([x[root]], len(x_1))
            y_2 = inter(x_2)

            # calculate the distance between true function and interpolation
            dist = np.sqrt(np.power(y_1 - y_2, 2))
            offset = np.argmin(dist) - region_size

            root = int(max(root + offset, 0))
            split_points.append(root)

        return np.array(split_points, dtype=np.uint), roots

    @staticmethod
    def get_segments(split_points: np.array, inter: Callable) -> List:
        """Returns the list segments given the list of `split_points`. Only the
        segments which contain a maximum are kept.

        Args:
            split_points (np.array): List of points where the function changes
            inter (Callable): Callable which returns the interpolation value at a given

        Returns:
            (list): List of tuples containing the start and end points of each
        """
        segments = []
        for i in range(len(split_points) - 1):
            x_1 = split_points[i] - 1
            x_2 = split_points[i] + 1
            y = inter([x_2]) - inter([x_1])
            if y < 0:
                segments.append((split_points[i + 1], split_points[i]))
        return segments

    def run(self, file_path: str) -> None:
        # read the file as dataframe, read key 'data'
        store = pd.HDFStore(file_path, mode="r")
        df = store["data"]
        store.close()

        data = df[self._list].to_numpy()
        x = np.linspace(0, int(len(data)) - 1, int(len(data)), dtype=np.uint)
        days = round(len(data) / (60 * 60 * 24))
        estimated_deg = days * 2 + 2

        # find the points where to split
        state_points = []
        trans_points = []
        for i in range(len(self._list)):
            inter = CoolingDetectionCallable.interpolate_polynomial(x, data[x, i], estimated_deg)
            points, _ = CoolingDetectionCallable.get_split_points(inter, data[:, i], 2,
                                                               region_size=1 * 60 * 60)
            state_points.append(points)
            points, _ = CoolingDetectionCallable.get_split_points(inter, data[:, i], 1,
                                                               region_size=1 * 60 * 60)
            trans_points.append(points)

        state_points = np.mean(np.array(state_points), axis=0, dtype=np.uint)
        trans_points = np.mean(np.array(trans_points), axis=0, dtype=np.uint)

        # create segments with data from the split points
        segments_idx = CoolingDetectionCallable.get_segments(state_points, inter)
        state = np.ones(data.shape[0])
        for i, segment in enumerate(segments_idx):
            start, end = segment
            state[start:end] = 0

        df[self._names[0]] = state
        df[self._names[1]] = abs(state - 1)

        segments_idx = CoolingDetectionCallable.get_segments(trans_points, inter)
        trans = np.ones(data.shape[0])
        for i, segment in enumerate(segments_idx):
            start, end = segment
            trans[start:end] = 0

        df[self._names[1]] = trans
        df[self._names[2]] = abs(trans - 1)

        # save the updatted DataFrame
        updated = pd.HDFStore(file_path, mode="w")
        updated.put("data", df, format="table")
        updated.close()


class CoolingStatus(BasePreProcessor):
    """This module adds cooling states to the dataset, as the temperatures show iregular
    behaviour from time to time. This might be due to enabling/disabling the active
    cooling of a compressor.
    """
    def __init__(self, detection_list: List[str] = ["SaugtemperaturZweiteStufeTETIT74341 (°C)",
                                                    "Motorlager (°C)",
                                                    "KühlwassertemperaturnachKühlerTETIT74422 (°C)"],
                 num_threads: int = 6) -> None:
        self._num_threads = num_threads
        self._detection_list = detection_list
        self._callable = CoolingDetectionCallable(cols=detection_list)

    def on_processing_finished(self, metadata: Metadata) -> None:
        # get all DataFrame files
        path = metadata[Meta.PATH]
        files = []
        for f in os.listdir(path):
            if os.path.isdir(os.path.join(path, f)):
                continue
            files.append(os.path.join(path, f))

        # divide list of files into chunks and give one chunk to each worker
        n = len(files) # e.g. 100
        s = max(math.ceil(n / self._num_threads), 1) # e.g. threads=4 -> s=25
        logger.debug("One worker processes %d files", s)
        logger.debug("Using %d workers", self._num_threads)
        logger.debug("Workers try to detect status based on %s", self._detection_list)
        worker = []
        for i in range(0, n, s): # i=0, i=25, i=75, ...
            logger.debug("Adding files %s to worker %d", files[i:i + s], i)
            worker.append(FileWorkerThread(files[i:i + s], self._callable))
            worker[-1].start()

        # wait for all workers to have finished
        for w in worker:
            w.join()
