import inspect
from uuid import uuid4
from Dataflux import get_logger

logger = get_logger(__name__)


def get_class_path(cls_name) -> str:
    return f"{inspect.getmodule(cls_name).__name__}.{cls_name.__name__}"


def mock_config(model_cls: type, **kwargs) -> dict:
    f_in = [str(uuid4()) for _ in range(3)]
    f_out = f_in

    seq = 60
    behavior = "future"
    model_args = {"class_path": get_class_path(model_cls)}

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
