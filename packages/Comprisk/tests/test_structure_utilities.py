from Comprisk.helper.structure import class_for_name
from Comprisk.model import BaseModel


def test_class_for_name():
    base_class = class_for_name("Comprisk.model.base", "BaseModel")
    assert BaseModel == base_class


def test_class_for_name_not_existent():
    try:
        class_for_name("Comprisk.model.not_available", "Error")
        raise AssertionError()
    except ModuleNotFoundError:
        assert True
