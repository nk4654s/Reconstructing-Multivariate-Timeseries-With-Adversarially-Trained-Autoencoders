"""This module merges downloaded and pre-processed datasets. The post-processing
entrypoint is meant to run after merging datasets."""

import argparse
import os
import shutil

from Dataflux.helper import Meta, Metadata


def main() -> None:
    """
    The main entry point for merging datasets. This function
    initializes the ArgumentParser to handle command-line interface
    arguments.
    """
    parser = argparse.ArgumentParser(description="This entrypoint merges two datasets")
    parser.add_argument("--datasets", nargs="+", type=str, required=True, help="A list of datasets")
    parser.add_argument( "--export", type=str, required=True, help="The output dataset name")
    args = parser.parse_args()

    # remove the export folder if exists
    if os.path.exists(args.export):
        shutil.rmtree(args.export)

    # check if dataset exists
    for dataset in args.datasets:
        if not os.path.exists(dataset):
            print(f"Dataset {dataset} does not exist!")
            exit(-1)

    # copy the first dataset as base dataset to expand with others
    shutil.copytree(args.datasets[0], args.export, dirs_exist_ok=True)
    shutil.rmtree(os.path.join(args.export, "report"))

    # update the metadata
    metadata = Metadata()
    metadata.load(args.datasets[0])
    metadata.put(args.export, Meta.PATH)
    metadata.drop(Meta.CORRELATION_OF_INTEREST)

    machine_uuid = metadata[Meta.MACHINE_UUID]
    files = metadata[Meta.FILES]
    days = [_["day"] for _ in files]
    recordings = metadata[Meta.TOTAL_RECORDINGS]

    # copy the rest of datasets
    for dataset in args.datasets[1:]:
        # copy the dataset files first
        for feather in os.listdir(dataset):
            if feather[-3:] == ".feather":
                shutil.copyfile(
                    os.path.join(dataset, feather), os.path.join(args.export, feather)
                )

        _metadata = Metadata()
        _metadata.load(dataset)
        assert machine_uuid == _metadata[Meta.MACHINE_UUID]

        for day in _metadata[Meta.FILES]:
            if day["day"] not in days:
                files.append(day)
        recordings += _metadata[Meta.TOTAL_RECORDINGS]

    metadata.put(recordings, Meta.TOTAL_RECORDINGS)
    metadata.put(files, Meta.FILES)
    metadata.save(args.export)


if __name__ == "__main__":
    main()
