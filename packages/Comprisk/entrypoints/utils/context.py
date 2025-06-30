from argparse import Namespace
from dataclasses import dataclass, InitVar, field
from enum import Enum

from lightning import Trainer, LightningModule
from torch import nn


class Environment(Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "dev"


def environment_from_string(string: str) -> Environment:
    for type_ in Environment:
        if type_.value == string or type_ == string:
            return type_

    raise ValueError(f"Input '{string}' is not part of 'Environment'.")


@dataclass
class EntrypointContext:
    config: Namespace
    config_folder: str
    model: LightningModule | nn.Module
    datamodule: LightningModule
    trainer: Trainer
    base_folder: str = "environment"
    environment_: InitVar[Environment | str] = Environment.DEVELOPMENT
    environment: Environment = field(init=False)

    def __post_init__(self, environment_: Environment | str) -> None:
        environment_ = environment_from_string(environment_) if isinstance(environment_, str) else environment_
        self.environment = environment_
