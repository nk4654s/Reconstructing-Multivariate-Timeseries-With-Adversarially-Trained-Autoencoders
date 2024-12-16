"""This module contains the metadata of a pytorch.Dataset which is required for some AI models."""
from dataclasses import InitVar, asdict, dataclass, field
from enum import Enum
from typing import Dict, List


class DatasetTypes(Enum):
    AUTOENCODER = "autoencoder"
    PREDICTIVE = "future"


def dataset_type_from_string(string: str) -> DatasetTypes:
    for type_ in DatasetTypes:
        if type_.value == string:
            return type_
    
    raise ValueError(f"Input '{string}' is not part of 'DatasetTypes'.")


@dataclass
class DatasetArgs:
    sequence_in: int
    sequence_out: int
    features_in: int
    features_out: int
    skip: int = 1
    stride: int = 1
    key: str = "norm"
    behavior_: InitVar[DatasetTypes | str] = DatasetTypes.AUTOENCODER.value
    behavior: DatasetTypes = field(init=False)
    ts_index: int = 0
    names: List[str] = field(default_factory=list)

    def __post_init__(self, behavior_: DatasetTypes | str) -> None:

        behavior_ = dataset_type_from_string(behavior_) if isinstance(behavior_, str) else behavior_
        self.behavior = behavior_

    def as_dict(self) -> Dict:
        dict_ = asdict(self)
        del dict_["features_in"]
        del dict_["features_out"]
        del dict_["ts_index"]
        del dict_["names"]
        return dict_
    
    def copy(self) -> "DatasetArgs":
        args = asdict(self)
        args["behavior_"] = args.pop("behavior").value

        return DatasetArgs(**args)
