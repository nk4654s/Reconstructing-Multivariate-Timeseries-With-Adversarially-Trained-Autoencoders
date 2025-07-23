from typing import Any

from lightning import Trainer
from lightning.pytorch.cli import (
    LightningArgumentParser,
LightningCLI
)

from Comprisk.custom import CASTrainer
from Comprisk.entrypoints.utils import EntrypointContext
from Comprisk.helper.structure import get_config_folder, deepvars


class CompmeshCLI(LightningCLI):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, save_config_callback=None)
        self._context = EntrypointContext(
            config=self.config,
            config_folder=get_config_folder(self),
            model=self.model,
            datamodule=self.datamodule,
            trainer=self.trainer,
            environment_=self.config.environment,
        )

    @property
    def context(self):
        return self._context

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        # use the custom trainer class with some modifications which are implicitly given in the config
        config = deepvars(self.config)
        config["config"] = str(config["config"][0])
        self.trainer_class = CASTrainer
        return super().instantiate_trainer(config = config, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--tag", type=str, default=None)
        parser.add_argument("--environment", type=str, default="dev")

        parser.add_argument("--prefix", type=str, default=None)
        parser.add_argument("--support-points", type=str, default=None)
        parser.add_argument("--name-mapping", type=str, default=None)

        # link the model's input/output features with the dataset's configuration
        parser.link_arguments("data.features_in", "model.init_args.features_in", apply_on="instantiate")
        parser.link_arguments("data.features_out", "model.init_args.features_out", apply_on="instantiate")
        parser.link_arguments("data.behavior", "model.init_args.behavior", apply_on="instantiate")
        parser.link_arguments("data.sequence_out", "model.init_args.sequence_out", apply_on="instantiate")
