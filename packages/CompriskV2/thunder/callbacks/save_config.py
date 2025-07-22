from argparse import Namespace
import os
import shutil
from lightning.pytorch.cli import SaveConfigCallback

from Dataflux.logger import get_logger

logger = get_logger(__name__)


def copy_dependencies(config: Namespace, key: str, log_dir: str) -> Namespace:
    key = key.split(".", maxsplit=1)
    if len(key) > 1:
        config[key[0]] = copy_dependencies(config[key[0]], key[-1], log_dir)
        return config

    path = config.get(key[0], None)
    if path:
        config[key[0]] = os.path.join(log_dir, key[0] + ".yaml")
        logger.info("Linked %s to new path %s", key[0], config[key[0]])
        shutil.copyfile(path, config[key[0]])

    return config


class AdvancedSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer, pl_module, stage):
        self._adjust_config(trainer)

        logger.debug("Save custom config files to %s", trainer.log_dir)
        self.config = copy_dependencies(self.config, "data.train", trainer.log_dir)
        self.config = copy_dependencies(self.config, "data.validate", trainer.log_dir)

        super().setup(trainer, pl_module, stage)

    def _adjust_config(self, trainer) -> None:
        if "ckpt" in self.config:
            del self.config["ckpt_path"]

        self.config["checkpoints"] = os.path.join(trainer.log_dir, "checkpoints")
        self.config["seed_everything"] = self.config["seed_everything"] + 1
