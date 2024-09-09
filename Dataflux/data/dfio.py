"""This module handles pandas dataframe input/outut for s3 streams"""

import os
from typing import List
import pandas as pd
from tqdm import trange

from Dataflux.data.download import ContinuousDataDownloader, EmptyStreamError
from Dataflux.helper import Meta, Metadata
from Dataflux.helper import (
    get_sensor_uuid_to_name,
    get_file,
)
from Dataflux.modules import BasePreProcessor, EntryPoint
from Dataflux import logger


class DFIO:
    """This class handles pandas dataframe input/outut for s3 streams."""

    def __init__(self, downloader: ContinuousDataDownloader) -> None:
        """Sets the s3 object to this class for iteration over streams.

        Args:
            s3 (S3): The s3 class.
        """
        self._donwloader = downloader
        self._meta = Metadata()

        self._sensor_names = get_sensor_uuid_to_name(self._donwloader.machine_uuid)
        self._store = None

    def _create_storage(self, dir_name: str = None, day_index: int = 0) -> None:
        # close a store if there was one opened
        if self._store:
            self._store.close()

        # get a valid file path based on the machine's UUID, the day_index and
        # increasing postfix-numbers (optional)
        day = self._donwloader.day(day_index).strftime("%Y-%m-%d")
        file_name = self._donwloader.machine_uuid + ".h5"
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
        if not df.empty:
            assert self._store is not None, "No HDF store opened."
            self._store.put("data", df, format="table", append=True)

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
        self._meta.put(self._donwloader.machine_uuid, Meta.MACHINE_UUID)

        # keep track of machine state changes, store a DF after the state changed
        self._meta.put(True, Meta.MACHINE_STATE_CHANGE)

        # iterate over all streams and append their content to the dataframe
        init_save = True
        desc = f"Fetching TSV daily data for machine {self._donwloader.machine_uuid}"
        for i in trange(len(self._donwloader), desc=desc):
            # get the DataFrame of the  stream, skip if the stream could not be opened
            try:
                df = self._donwloader[i]
            except EmptyStreamError:
                continue

            # run the modules which require non-processed data
            # as result, df might be a list
            logger.info("Processing data using %s modules.", len(modules))
            self._meta.put(self._donwloader.day(i).strftime("%Y-%m-%d"), Meta.DAY)
            for module in modules:
                df = module(df, EntryPoint.ON_DOWNLOAD_FINISHED, metadata=self._meta)
                shape = 0 if isinstance(df, list) else len(df)
                if isinstance(df, list):
                    for _ in df:
                        shape += len(_)

                logger.debug("Module %s outputs list of dataframes? %s (total=%s)",
                             module.__class__.__name__,
                             isinstance(df, list),
                             shape)

            # some modules might return a list of DataFrames, save each of those
            dfs = [df] if isinstance(df, pd.DataFrame) else df
            if len(dfs) == 0:
                continue

            if init_save:
                init_save = False
                self._create_storage(dir_name, i)

            # save the processed dataframes
            logger.info("Single TSV download resultet in %s file(s).", len(dfs))
            for j, df in enumerate(dfs):
                # check if a new storage needs to be created, only after a machine state
                # change
                if j > 0 or self._meta[Meta.MACHINE_STATE_CHANGE]:
                    self._create_storage(dir_name, i)

                self.save(df)

            self._meta.put(False, Meta.MACHINE_STATE_CHANGE)

        if self._store:
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
