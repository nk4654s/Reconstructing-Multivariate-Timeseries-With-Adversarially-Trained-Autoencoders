import os
import inspect
import yaml
from uuid import uuid4
from CompriskV2.model.ko3 import SupportPointModel
from Dataflux import get_logger

logger = get_logger(__name__)


def get_class_path(cls_name) -> str:
    return f"{inspect.getmodule(cls_name).__name__}.{cls_name.__name__}"


def mock_config(model_cls: type, **kwargs) -> dict:
    f_in = [str(uuid4()) for _ in range(3)]
    f_out = f_in

    seq = 10
    behavior = "future"
    model_args = {"class_path": get_class_path(model_cls)}

    if model_cls == SupportPointModel:
        model = kwargs.get("model", {})
        path = os.path.join(model["ko3_table"], "mapping.yaml")
        with open(path, "r") as f:
            mapping = yaml.safe_load(f)
            f_in = [v for v in mapping["inputs"].values()]
            f_out = [v for v in mapping["outputs"].values()]

        seq = 1
        behavior = "autoencoder"
        model_args.update({"init_args": model})

    data = {
        "env": "test",
        "tag": "test",
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
        },
        "model": model_args,
        "data": {
            "features_in": list(f_in),
            "features_out": list(f_out),
            "sequence_in": seq,
            "sequence_out": 1,
            "behavior": behavior,
        },
    }

    logger.debug("Test config %s", data)
    return data
