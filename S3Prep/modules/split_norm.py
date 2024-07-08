"""This module normalizes a dataset and splits it into train, val, test dataset."""
import json
import random
import os
from typing import Dict
from S3Prep.modules.normalizer.minmax_thread import MinMaxFileThread
from S3Prep.modules.normalizer.standard_thread import StandardizeFileThread
from S3Prep.modules import BasePreProcessor


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

    def _split_dataset(self, path: str) -> Dict:
        # split dataset
        with open(f"{path}/metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        train_len = int(len(metadata["files"]) * self._train_split)
        val_len = int(len(metadata["files"]) * self._val_split)

        # shuffle the file_path arrays
        if self._shuffle:
            random.shuffle(metadata["files"])

        # save the new dataset metadata
        new_json = {
            "train": [m for m in metadata["files"][:train_len]],
            "val": [m for m in metadata["files"][train_len:train_len + val_len]],
            "test": [m for m in metadata["files"][train_len + val_len:]]
        }
        del metadata["files"]
        new_json = new_json | metadata

        return new_json

    def on_processing_finished(self, metadata: Dict) -> None:
        # split the dataset into train, val, test sets
        report_folder = metadata["data_path"] + "/report"
        metadata = self._split_dataset(report_folder)

        # fit the min max scaler on the train dataset
        files = [os.path.join("dataset", f"test_{f['day']}.h5") for f in metadata["train"]]
        scaler_a, scaler_b = self._fit(files, self._num_thread)

        # transform all datasets
        for dataset in ["train", "val", "test"]:
            files = [os.path.join("dataset", f"test_{f['day']}.h5") for f in metadata[dataset]]
            self._transform(files, scaler_a, scaler_b, self._num_thread)

        # add the normalizer info's to the metadata
        metadata["norm"] = {
            "scaler_a": scaler_a.to_dict(),
            "scaler_b": scaler_b.to_dict(),
            "method": self._method
        }

        # save the new dataset metadata
        with open(f"{report_folder}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
