"""This module handles pandas dataframe input/outut for s3 streams"""

import os
from typing import List, TextIO
import pandas as pd
import numpy as np
from tqdm import trange

from S3Prep.data import S3
from S3Prep.data.header import Header
from S3Prep.helper import Meta, Metadata
from S3Prep.helper import get_sensor_uuid_to_name, get_file, to_float, get_logger
from S3Prep.modules import BasePreProcessor, EntryPoint

# create logger object
logger = get_logger(__name__)


class DFIO:
    """This class handles pandas dataframe input/outut for s3 streams."""

    def __init__(self, s3: S3) -> None:
        """Sets the s3 object to this class for iteration over streams.

        Args:
            s3 (S3): The s3 class.
        """
        self._s3 = s3
        self._meta = Metadata()
        self._sensor_names = get_sensor_uuid_to_name(self._s3.machine_uuid)

        self._header: Header = None
        self._store = None

    def parse_header(self, line: str) -> Header:
        """This method parses a line into a header and checks for validity. If
        there is a change in number of header.values, this method will define a
        relative complement of the previous header and the new header. This set 
        of values will be dropped from the DataFrame.

        Args:
            line (str): The line to parse into a header.
        
        Returns:
            Header: The parsed header.
        """
        # build the header instance
        incoming = Header()
        incoming.values = line.rstrip("\n").split("\t")

        # if no header is registered so far, just use the latest income
        if self._header is None:
            logger.debug("Set initial header with %s values", incoming.length)
            self._header = incoming
            return self._header

        # if the incoming header is longer than the previous one, drop the
        # values which are unknown or missing in the previous one and
        # return it
        if incoming.values == self._header.values:
            logger.debug("No change in header, returning previous one.")
            return self._header

        # calculate the headers to remove
        assert (self._header.length < incoming.length), \
            "Header length mismatch. Incoming header length is smaller than known header length."

        incoming.relative_complement(self._header, fail_safe=True)
        logger.debug("Header length mismatch. Removing %s values.", len(incoming.drop))
        return incoming

    def _create_storage(self, dir_name: str = None, day_index: int = 0) -> None:
        # close a store if there was one opened
        if self._store:
            self._store.close()

        # get a valid file path based on the machine's UUID, the day_index and
        # increasing postfix-numbers (optional)
        day = self._s3.day(day_index).strftime("%Y-%m-%d")
        file_name = self._s3.machine_uuid + ".h5"
        file_name = get_file(os.path.join(dir_name, file_name), postfix=day)
        logger.debug("Saving DF to %s", file_name)

        self._store = pd.HDFStore(file_name, "w")

    def save(self, df: pd.DataFrame) -> None:
        """Saves the dataframe in an HDF5 file and frees memory.

        Args:
            df (pd.DataFrame): dataframe to be saved
            dir_name (str): Path of the HDF5 file where to save
            day_index (int): index of the day to be saved
        """
        assert df is not None, "No dataframe to save"
        assert self._store is not None, "No HDF store opened."
        if not df.empty:
            self._store.put("data", df, format="table", append=True)

    def parse_stream(self, stream: TextIO = "dataset", day_index: int = 0) -> pd.DataFrame:
        """Downloads the dataframe from a S3 stream and puts it into a pandas DataFrame.

        Args:
            stream (TextIO): Stream to be read.
            day_index (int): index of the day to be saved
        
        Returns:
            df (pd.DataFrame): Pandas DataFrame containing the data from the stream
        """
        # check if the dataframe is empty and set it up accordingly
        # also check if the header is consistent with the stream's one
        # if not, save the progress so far and continue on a fresh dataframe instance
        values = []
        lines = stream.readlines()
        if len(lines) <= 1:
            return None
        logger.info("Streaming day %s", self._s3.day(day_index))
        logger.info("Streaming lines %s", len(lines))

        # read all values from the incoming sensor data
        for i, line in enumerate(lines):
            if i == 0:
                header = self.parse_header(line)
                continue

            # extract and format the row
            columns = line.strip("\n").split("\t")
            row = [to_float(s) for s in columns]
            values.append(row)

        values = np.array(values)
        data = {key: val for (key, val) in zip(header.values, values.T)}

        return pd.DataFrame(data)

    def download(
        self, dir_name: str = "dataset", modules: List[BasePreProcessor] = None
    ) -> None:
        """Downloads the dataframe from a S3 stream and downloads the dataframe as file.

        Args:
            dir_name (str): Path of the HDF5 file to be saved. If not provided, the current
            folder will be used. Defaults to "dataset".
            modules (List[BasePreProcessor]): List of modules to be used. Defaults to None.
        """
        # organize modules and build metadata as information for modules
        modules = modules if modules else []
        self._meta.put([module.__class__.__name__ for module in modules], Meta.MODULES)
        self._meta.put(dir_name, Meta.PATH)
        self._meta.put(self._s3.machine_uuid, Meta.MACHINE_UUID)

        # keep track of machine state changes, store a DF after the state changed
        self._meta.put(True, Meta.MACHINE_STATE_CHANGE)

        # iterate over all streams and append their content to the dataframe
        desc = f"Fetching TSV daily data for machine {self._s3.machine_uuid}"
        for i in trange(len(self._s3), desc=desc):
            # get the stream, skip if the stream could not be opened
            stream = self._s3[i]
            if stream is None:
                continue

            # download the TSV file from the S3 bucket and parse it into a DataFrame
            df = self.parse_stream(stream, day_index=i)
            if df is None:
                continue

            # run the modules which require non-processed data
            # as result, df might be a list
            logger.info("Processing data using %s modules.", len(modules))
            self._meta.put(self._s3.day(i).strftime("%Y-%m-%d"), Meta.DAY)
            for module in modules:
                df = module(df, EntryPoint.ON_DOWNLOAD_FINISHED, metadata=self._meta)

            # some modules might return a list of DataFrames, save each of those
            dfs = [df] if isinstance(df, pd.DataFrame) else df
            logger.info("Single TSV download resultet in %s file(s).", len(dfs))

            # save the processed dataframes
            for j, df in enumerate(dfs):
                # check if a new storage needs to be created, only after a machine state
                # change
                if i == 0 or (j > 0 or self._meta[Meta.MACHINE_STATE_CHANGE]):
                    self._create_storage(dir_name, i)
                self.save(df)
            self._meta.put(False, Meta.MACHINE_STATE_CHANGE)
        self._store.close()

        # execute modules which require to be executed after processing
        # hence, there is no dataframe to process
        for module in modules:
            module(metadata=self._meta, entry_point=EntryPoint.ON_PROCESSING_FINISHED)

        # make sure to drop (optional) graphs created by other modules
        self._meta.drop(Meta.ADDITIONAL_REPORT_GRAPHS)

        logger.debug("Report: Saving metadata info to %s", dir_name)
        self._meta.save(dir_name)
        logger.info("Finished downloading and processing.")
