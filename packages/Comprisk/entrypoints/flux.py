import os
from typing import Dict

import yaml
from Dataflux import get_logger, logging_mode
from Dataflux.cli import FluxConfigManager
from Dataflux.pipeline import pipeline as flux_pipeline
from Dataflux.pytorch import EmptyDatasetError

from Comprisk.lightning import DataModule
from Comprisk.entrypoints.base import EntrypointBase, EntrypointContext

logger = get_logger(__name__, "logfile", logging_mode)


class FluxEntrypoint(EntrypointBase):
    """
    This entrypoint class handles loading a datamodule using `Dataflux.pytorch.dataset`. The required path variables
    are located in the configuration file under `data.train` / `data.validate` and accept both, the path to a
    downloaded dataset or `Dataflux` pipeline configuration.

    When fine-tuning a model using a different dataset, load the model as follows

    ```
    comprisk-train -c path_to_trained_model/config.yaml
    ```

    Then create a new configuration file defining the content of the `data` key and add the path to it to the
    `comprisk-train` entrypoint using `--data`.
    """
    def __init__(self) -> None:
        super().__init__()

    def _get_path(self, context: EntrypointContext, path: str, check_exists: bool = True) -> str | None:
        _original_path = path
        if not path:
            return None

        # make sure the path contains the environment path e.g. environment/dev/{path}
        base_path = os.path.join(context.base_folder, context.environment.value)
        if not path.startswith(base_path):
            path = os.path.join(base_path, path)

        # if the previous path did not exist, do a fallback and check if the path exists in the
        # config folder. This is likely when loading a model.
        if (path.endswith("yaml") or path.endswith("yml")) and check_exists and not os.path.exists(path):
            path = os.path.join(context.config_folder, _original_path)

        # if requested, check if the path exists. If the path does not exist, then return 'None'.
        if check_exists:
            path = path if os.path.exists(path) else None

        logger.debug("Got path-name: %s to %s", _original_path, path)
        return path

    def _adjust_config_paths(self, context: EntrypointContext, path: str) -> Dict:
        if path is None or os.path.isdir(path):
            return {}

        with open(path, mode="r") as f:
            config = yaml.safe_load(f)

            # add the base path prefix depending on the configured environment
            # Do not check if the folder exists, as this method adjusts the flux-pipeline configuration file.
            # The config["folder"] variable points to the dataset, which might not be downloaded yet.
            config["folder"] = self._get_path(context, config["folder"], check_exists=False)

        return config

    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
        dataset_paths = {
            "train": self._get_path(context, context.config.data.train),
            "validate": self._get_path(context, context.config.data.validate)
        }

        if "__path__" in context.config.data:
            del context.config.data.__path__

        datamodule_args_modified = vars(context.config.data)
        for key, path in dataset_paths.items():
            datamodule_args_modified[key] = path
            if path is None or os.path.isdir(path):
                logger.debug("Flux not required: flux-pipeline(%s)", path)
                continue

            # use dataflux to download the dataset
            config = self._adjust_config_paths(context, path)
            flux = FluxConfigManager(path, custom_config=config)

            logger.debug("Executing flux-pipeline(%s)", path)
            logger.debug("Using custom flux-config: %s", "yes" if config else "no")
            folder_path = flux_pipeline(flux)
            logger.debug("Flux loaded path (%s): %s", key, folder_path)

            # update the comprisk context to let it know where to find the dataset
            datamodule_args_modified[key] = folder_path

        # finalize the context
        try:
            context.config.data.train = datamodule_args_modified["train"]
            context.config.data.validate = datamodule_args_modified["validate"]
            context.datamodule = DataModule(**datamodule_args_modified)
        except EmptyDatasetError:
            logger.error("Unable to load dataset: EmptyDatasetError")

        logger.debug("Loaded (train) from %s", context.config.data.train)
        logger.debug("Loaded (val) from %s", context.config.data.validate)
        return context
