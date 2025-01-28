import os

from Comprisk.data import DataModule
from Dataflux.cli import FluxConfigManager
from Dataflux.pipeline import pipeline as flux_pipeline

from Comprisk.entrypoints.base import EntrypointContext, EntrypointBase


class FluxEntrypoint(EntrypointBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
        dataset_paths = {
            "train": context.config.data.train,
            "validate": context.config.data.validate
        }

        download_required = False
        datamodule_args_modified = context.config.data
        for key, path in dataset_paths.items():
            if os.path.isdir(path):
                continue

            # use dataflux to download the dataset
            flux = FluxConfigManager(path)
            folder_path = flux_pipeline(flux)

            # update the comprisk context to let it know where to find the dataset
            datamodule_args_modified[key] = folder_path
            download_required = True

        # finalize the context
        if download_required:
            context.config.data.train = datamodule_args_modified["train"]
            context.config.data.validate = datamodule_args_modified["validate"]
            context.datamodule = DataModule(**datamodule_args_modified)

        return context
