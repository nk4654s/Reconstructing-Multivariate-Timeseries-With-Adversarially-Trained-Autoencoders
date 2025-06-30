import os
import shutil
from argparse import Namespace

from lightning.pytorch.cli import SaveConfigCallback

from Dataflux import get_logger, logging_mode

logger = get_logger(__name__, "logfile", logging_mode)


def copy_dependencies(config: Namespace | dict, key: str, log_dir: str) -> Namespace:
    key = key.split(".", maxsplit=1)
    if len(key) > 1:
        sub_config = copy_dependencies(config[key[0]], key[-1], log_dir)
        config[key[0]] = sub_config
        return config

    path = config.get(key[0], "")
    if path:
        config[key[0]] = os.path.join(log_dir, key[0] + ".yaml")
        shutil.copyfile(path, config[key[0]])

    return config


class AdvancedSaveConfigCallback(SaveConfigCallback):
    """ This callback class handles saving the configuration files to the model output folder. Also, it
    adjusts paths of the following keys to match the output folder:

        - normalizer
        - tuner_config
        - data.train
        - data.validate

    """
    def setup(self, trainer, pl_module, stage):
        self._adjust_config(trainer)

        logger.debug("Save custom config files to %s", trainer.log_dir)
        self.config = copy_dependencies(self.config, "normalizer", trainer.log_dir)
        self.config = copy_dependencies(self.config, "tuner_config", trainer.log_dir)
        self.config = copy_dependencies(self.config, "data.train", trainer.log_dir)
        self.config = copy_dependencies(self.config, "data.validate", trainer.log_dir)

        super().setup(trainer, pl_module, stage)

    def _adjust_config(self, trainer) -> None:
        if "ckpt" in self.config.keys():
            del self.config["ckpt_path"]

        self.config["seed_everything"] = self.config["seed_everything"] + 1
