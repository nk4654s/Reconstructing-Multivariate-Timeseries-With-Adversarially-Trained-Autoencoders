from Comprisk.helper.structure import get_optimizer_by_name, get_loss_by_name, get_weight_init_by_name
from Comprisk.helper.structure import class_for_name


def get_class_by_name(name: str) -> type:
    parts = name.rsplit(".", maxsplit=1)
    if len(parts) < 2:
        raise ValueError(f"Given name could not be parsed: {name}")

    return class_for_name(parts[0], parts[1])


__all__ = ["get_optimizer_by_name", "get_loss_by_name", "get_class_by_name", "get_weight_init_by_name"]
