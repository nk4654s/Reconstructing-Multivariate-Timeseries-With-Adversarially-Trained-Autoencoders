import os

from Comprisk.data import DataModule
from Dataflux.cli import FluxConfigManager
from Dataflux.pipeline import pipeline as flux_pipeline

from Comprisk.entrypoints.base import EntrypointContext, EntrypointBase


class FluxEntrypoint(EntrypointBase):
    def __init__(self) -> None:
        super().__init__()

    def _get_path(self, path: str, context: EntrypointContext) -> str:
        base_path = os.path.join(context.base_folder, context.environment.value)
        if path.startswith(base_path):
            return path

        return os.path.join(base_path, path)

    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
        dataset_paths = {
            "train": self._get_path(context.config.data.train, context),
            "validate": self._get_path(context.config.data.validate, context)
        }

        if "__path__" in context.config.data:
            del context.config.data.__path__

        datamodule_args_modified = context.config.data
        for key, path in dataset_paths.items():
            datamodule_args_modified[key] = path
            if os.path.isdir(path):
                continue

            # use dataflux to download the dataset
            flux = FluxConfigManager(path)
            folder_path = flux_pipeline(flux)

            # update the comprisk context to let it know where to find the dataset
            datamodule_args_modified[key] = folder_path

        # finalize the context
        context.config.data.train = datamodule_args_modified["train"]
        context.config.data.validate = datamodule_args_modified["validate"]
        context.datamodule = DataModule(**datamodule_args_modified)

        return context
