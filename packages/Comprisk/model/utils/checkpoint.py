"""This module contains helper methods related to the models."""

import os

from lightning import LightningModule

from Comprisk.custom import CompriskCLI
from Comprisk.helper.structure import class_for_name

from Dataflux import logging_mode
from Dataflux.logger import get_logger

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)

CHECKPOINT_FORMAT = ".ckpt"
CHECKPOINT = "checkpoints"


def get_best_checkpoint(path: str, files: list[str]) -> str:
    files = [f for f in files if CHECKPOINT_FORMAT in f]
    files.sort()
    return os.path.join(path, files[-1])


class Checkpoint:
    def __init__(self, path: str):
        config_path = os.path.join(path, "config.yaml")
        if not os.path.exists(config_path):
            logger.error("Given path is not a model output folder: '%s'", path)
            raise ValueError("Could not find 'config.yaml' in path " + path)

        cli = CompriskCLI(
            run=False,
            subclass_mode_model=True,
            parser_kwargs={"default_config_files": [config_path]},
        )

        self._path = os.path.join(path, CHECKPOINT)
        files = os.listdir(self._path)
        ckpt_path = get_best_checkpoint(self._path, files)

        self._model_cls = class_for_name(cli.config.model.class_path)
        self._checkpoint = self._model_cls.load_from_checkpoint(ckpt_path, **cli.config.model.init_args)
        self._checkpoint.eval()

        logger.info("Loaded checkpoint '%s'", ckpt_path)

    @property
    def model(self) -> LightningModule:
        return self._checkpoint

    def apply_score_wrapper(self, cls, data: LightningModule, **kwargs):
        wrapper = cls(self.model)
        wrapper.train_wrapper(data, **kwargs)
        self._checkpoint = wrapper


if __name__ == "__main__":
    import sys
    Checkpoint(sys.argv.pop(1))
