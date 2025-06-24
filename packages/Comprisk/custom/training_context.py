from uuid import UUID

from Dataflux import logging_mode, get_logger
from Dataflux.helper import get_sensors
from Dataflux.pytorch import DatasetArgs

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class TrainingContext:

    __obj: "TrainingContext" = None

    def __init__(self, dataset_args: DatasetArgs, machine: str) -> None:
        if TrainingContext.__obj:
            raise ValueError("Training context already exists, access via TrainingContext.get_instance()")

        self._dataset_args = dataset_args
        self._machine_uuid = machine
        self._sensors = get_sensors(UUID(machine))

    @property
    def feature_names(self) -> list[str]:
        return self._dataset_args.names_in

    def target_names(self) -> list[str]:
        return self._dataset_args.names_out

    def map_sensors(self, names: list[str]) -> list[str]:
        return [self._sensors[_] for _ in names]

    @classmethod
    def __reset__(cls, *args, **kwargs) -> "TrainingContext":
        cls.__obj = TrainingContext(*args, **kwargs)
        return cls.get_instance()

    @classmethod
    def get_instance(cls) -> "TrainingContext":
        if not cls.__obj:
            raise ValueError("Training context not initialized. Did you run CASTrainer.fit()?")

        return cls.__obj
