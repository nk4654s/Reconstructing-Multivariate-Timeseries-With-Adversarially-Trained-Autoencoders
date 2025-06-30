"""This module creates a model and tunes its hyperparameters."""
import os

import yaml

from Comprisk.tune import ConfigManager, Tuner
from Comprisk.entrypoints import EntrypointBase
from Comprisk.entrypoints.utils import EntrypointContext

from Dataflux import logging_mode
from Dataflux.logger import get_logger


# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class TuneEntrypoint(EntrypointBase):
    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
        config = ConfigManager(context)
        if not config.initialized:
            logger.warning("Tuner not initialized")
            quit()

        # run the hyperparameter optimization with the ConfigManager
        tuner = Tuner(config, context)
        params = tuner.optimize(context.config.tuner_trials, context.config.tag)
        logger.info("Best parameters: %s", params)

        # get the base config for training and modify the parameters with the best ones
        modified = config.modify_config(params)

        # save the modified config to a new file with increasing suffix
        modified_path = os.path.join(config.train_path, "config_tuned_0.yml")
        i = 0
        while os.path.exists(modified_path):
            modified_path = os.path.join(config.train_path, f"config_tuned_{i}.yml")
            i += 1

        with open(modified_path, "w", encoding="utf-8") as f:
            yaml.dump(modified, f)

        quit()

        return context
