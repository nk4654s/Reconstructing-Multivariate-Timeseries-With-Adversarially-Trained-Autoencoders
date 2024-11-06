"""This module handles pandas dataframe input/outut for s3 streams"""

import os
from typing import List
import pandas as pd
from tqdm import trange

from Dataflux.data.download import ContinuousDataDownloader, EmptyStreamError
from Dataflux.helper import Meta, Metadata, DataWrapper, get_file
from Dataflux.modules import BasePreProcessor, EntryPoint
from Dataflux import logger


def download(
        downloader: ContinuousDataDownloader,
        dir_name: str = "dataset",
        modules: List[BasePreProcessor] = []
    ) -> None:
    """Downloads the dataframe from a S3 stream and downloads the dataframe as file.

    Args:
        dir_name (str): Path of the HDF5 file to be saved. If not provided, the current
        folder will be used. Defaults to "dataset".
        modules (List[BasePreProcessor]): List of modules to be used. Defaults to None.
    """
    filename = str(downloader.machine_uuid) + ".h5"
    store = None

    # organize modules and build metadata as information for modules
    modules = modules if modules else []
    meta = Metadata()
    meta.put([module.__class__.__name__ for module in modules], Meta.MODULES)
    meta.put(dir_name, Meta.PATH)
    meta.put(str(downloader.machine_uuid), Meta.MACHINE_UUID)

    # keep track of machine state changes, store a DF after the state changed
    meta.put(True, Meta.MACHINE_STATE_CHANGE)

    # iterate over all streams and append their content to the dataframe
    for i in trange(len(downloader), desc="Pre-Processing Dataset"):
        day = downloader.day(i).strftime("%Y-%m-%d")
        file_path = get_file(os.path.join(dir_name, filename), postfix=day)

        # get the DataFrame of the  stream, skip if the stream could not be opened
        try:
            df = downloader[i]
        except EmptyStreamError:
            continue

        wrappers = [DataWrapper(df, file_path)]

        # run the modules which require non-processed data
        # as result, df might be a list
        logger.info("Processing data using %s modules.", len(modules))
        meta.put(downloader.day(i).strftime("%Y-%m-%d"), Meta.DAY)
        for module in modules:
            wrappers = module(wrappers, meta, EntryPoint.ON_DOWNLOAD_FINISHED)
            shape = sum([len(part.df) for part in wrappers])
            logger.debug("Module %s outputs list of dataframes? (total=%s)",
                            module.__class__.__name__,
                            shape)

        # some modules might return a list of DataFrames, save each of those
        if len(wrappers) == 0:
            continue

        for wrapper in wrappers:
            logger.debug("Saving DF to %s", wrapper.path)
            store = wrapper.save(store)

    if store is not None and store.is_open:
        store.close()

    # execute modules which require to be executed after processing
    # hence, there is no dataframe to process
    for module in modules:
        module(df=None, metadata=meta, entry_point=EntryPoint.ON_PROCESSING_FINISHED)

    # make sure to drop (optional) graphs created by other modules
    meta.drop(Meta.ADDITIONAL_REPORT_GRAPHS)

    logger.debug("Report: Saving metadata info to %s", dir_name)
    meta.save(dir_name)
    logger.info("Finished downloading and processing.")
