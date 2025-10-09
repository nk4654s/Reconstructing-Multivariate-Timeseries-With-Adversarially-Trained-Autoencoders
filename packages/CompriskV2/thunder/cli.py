import os
from typing import Any

from lightning import Trainer
from lightning.pytorch import loggers as ptl
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
)

from CompriskV2.thunder.callbacks import AdvancedSaveConfigCallback


class CompriskCLI(LightningCLI):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, save_config_callback=AdvancedSaveConfigCallback, **kwargs)

    @property
    def root_dir(self):
        config = self.config.get("config", [])
        assert len(config), f"Expected exactly one configuration, got {len(config)}."
        assert os.path.exists(config[0]), f"Path of config does not exist: {config[0]}"
        assert os.path.isfile(config[0]), f"Expected config path to point to a file: {config[0]}"

        path = os.path.dirname(self.config.config[0])
        path = path.replace(os.getcwd(), "")[1:]

        return path

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)

        parser.add_argument("--checkpoints", default=None)
        parser.add_argument("--report-name", default=None)

        parser.add_argument("--tag", type=str, required=True)
        parser.add_argument("--env", type=str, required=True)

        # Link data arguments directly to the @LinkedArgumentsModel base class
        parser.link_arguments("data.features_in", "model.init_args.features_in", apply_on="parse")
        parser.link_arguments("data.features_out", "model.init_args.features_out", apply_on="parse")

        parser.link_arguments("data.sequence_in", "model.init_args.sequence_in", apply_on="parse")
        parser.link_arguments("data.sequence_out", "model.init_args.sequence_out", apply_on="parse")

        parser.link_arguments("data.standardize","model.init_args.standardize", apply_on="parse")

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        path = f"environment/{self.config.env}/models/{self.model.__class__.__name__}"
        tb_logger = ptl.TensorBoardLogger(save_dir=path, name=self.config.tag)
        kwargs["logger"] = tb_logger
        return super().instantiate_trainer(**kwargs)
