from CompriskV2.model import LinkedParameterModel
from CompriskV2.model.parser import get_class_by_name


def test_class_for_name():
    base_class = get_class_by_name("CompriskV2.model.LinkedParameterModel")
    assert LinkedParameterModel == base_class


def test_class_for_name_not_existent():
    try:
        get_class_by_name("CompriskV2.model.not_available.Error")
        raise AssertionError()
    except ModuleNotFoundError:
        assert True


def test_get_class_by_name_invalid():
    try:
        get_class_by_name("invalid_as_not_dot")
        raise AssertionError()
    except ValueError:
        assert True
