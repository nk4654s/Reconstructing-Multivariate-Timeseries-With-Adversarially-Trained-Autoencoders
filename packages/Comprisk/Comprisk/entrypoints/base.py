from argparse import Namespace
from dataclasses import dataclass, field, InitVar
from enum import Enum
from typing import List, Tuple
from lightning import Trainer

from Comprisk.data import DataModule
from Comprisk.model import BaseModel


class Environment(Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "dev"


def environment_from_string(string: str) -> Environment:
    for type_ in Environment:
        if type_.value == string:
            return type_
        elif type_ == string:
            return type_

    raise ValueError(f"Input '{string}' is not part of 'Environment'.")


@dataclass
class EntrypointContext:
    config: Namespace
    config_folder: str
    model: BaseModel
    datamodule: DataModule
    trainer: Trainer
    environment_: InitVar[Environment | str] = Environment.DEVELOPMENT
    environment: Environment = field(init=False)

    def __post_init__(self, environment_: Environment | str) -> None:
        environment_ = environment_from_string(environment_) if isinstance(environment_, str) else environment_
        self.environment = environment_


class EntrypointBase:
    def __init__(self) -> None:
        pass

    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
        return context


class Pipeline:
    def __init__(self, entrypoints: List[EntrypointBase]) -> None:
        self._entrypoints = entrypoints

    def run(self, context: EntrypointContext) -> Tuple[str, int]:
        for i, entry in enumerate(self._entrypoints):
            context: EntrypointContext = entry(context)
            yield entry.__class__.__name__, i
