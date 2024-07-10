"""This module normalizes a dataset and splits it into train, val, test dataset."""
import logging
import random
import os
import re
from typing import Dict
from S3Prep.helper import Metadata
from S3Prep.modules.normalizer.minmax_thread import MinMaxFileThread
from S3Prep.modules.normalizer.standard_thread import StandardizeFileThread
from S3Prep.modules import BasePreProcessor
from S3Prep.helper import Meta, get_logger

# create logger object
logger = get_logger(__name__)


class Normalize(BasePreProcessor):
    """This pre-processor module executes after all dataset files were downloaded
    and processed by other modules. Then, it finds the min/max values of each
    column to normalize the dataset. A split into train/val/test dataset is required.
    """

    METHODS = ["normalize", "standardize"]

    def __init__(self, method: str = "normalize", train_split: float = 0.7,
                 val_split: float = 0.2, shuffle: bool = False,
                 threads: int = 6) -> None:
        assert method in Normalize.METHODS, "Invalid normalization method provided."
        self._method = method
        if self._method == Normalize.METHODS[0]:
            self._fit = MinMaxFileThread.fit_on_files
            self._transform = MinMaxFileThread.transform_on_files
        else:
            self._fit = StandardizeFileThread.fit_on_files
            self._transform = StandardizeFileThread.transform_on_files

        self._shuffle = shuffle
        self._train_split = train_split
        self._val_split = val_split
        self._num_thread = threads

    def _split_dataset(self, metadata: Metadata) -> Dict:
        train_len = int(len(metadata[Meta.FILES]) * self._train_split)
        val_len = int(len(metadata[Meta.FILES]) * self._val_split)

        # shuffle the file_path arrays
        if self._shuffle:
            random.shuffle(metadata[Meta.FILES])

        # split the file paths into train/val
        metadata.put([], Meta.TRAIN_FILES)
        metadata.put([], Meta.VALIDATION_FILES)
        metadata.put([], Meta.TEST_FILES)

        for i in range(len(metadata[Meta.FILES])):
            if i < train_len:
                metadata[Meta.TRAIN_FILES].append(metadata[Meta.FILES][i])
            elif i <= (train_len + val_len - 1):
                metadata[Meta.VALIDATION_FILES].append(metadata[Meta.FILES][i])
            else:
                metadata[Meta.TEST_FILES].append(metadata[Meta.FILES][i])

        # drop the old files array
        metadata.drop(Meta.FILES)

    def _get_file_prefix(self, path: str) -> str:
        for f in os.listdir(path):
            if os.path.isfile(path + "/" + f):
                return re.sub(r"(_\d{4}-\d{2}-\d{2}.*\.h5)", "", f)

        return None

    def on_processing_finished(self, metadata: Metadata) -> None:
        # split the dataset into train, val, test sets
        self._split_dataset(metadata)
        logging.debug("Splitted dataset into 'Train', 'Validation', and 'Test' sets.")

        # fit the min max scaler on the train dataset
        path = metadata[Meta.PATH]
        prefix = self._get_file_prefix(path=path)
        files = [os.path.join(path, f"{prefix}_{f['day']}.h5") for f in metadata[Meta.TRAIN_FILES]]
        scaler_a, scaler_b = self._fit(files, self._num_thread)
        logging.debug("Fitted '%s' on 'Train'", self._method)

        # transform all datasets
        for dataset in [Meta.TRAIN_FILES, Meta.VALIDATION_FILES, Meta.TEST_FILES]:
            files = [os.path.join(path, f"{prefix}_{f['day']}.h5") for f in metadata[dataset]]
            self._transform(files, scaler_a, scaler_b, self._num_thread)
            logging.debug("Transformed on %s using '%s'", dataset.name, self._method)

        # add the normalizer info's to the metadata
        metadata.put({
            Meta.NORM_VAL_A.name: scaler_a.to_dict(),
            Meta.NORM_VAL_B.name: scaler_b.to_dict(),
            Meta.NORM_METHOD.name: self._method
        }, Meta.NORMALIZER)

        # save the new dataset metadata
        metadata.save(path)
