"""Dataflux/data/utils.py"""

import os
import shutil
from contextlib import contextmanager

import numpy as np
from Dataflux.helper import get_file, to_float


@contextmanager
def environment():
    if os.path.exists(".tmp"):
        shutil.rmtree(".tmp")
    os.mkdir(".tmp")
    yield
    shutil.rmtree(".tmp")


def test_to_float():
    value = "123.45"
    result = to_float(value)
    assert result == 123.45

    value = "abc"
    result = to_float(value)
    assert np.isnan(result)


def test_get_file_create_folder():
    name = "file.ext"
    postfix = "post"
    path = ".tmp"
    with environment():
        path = os.path.join(path, name)
        file_name = get_file(path, postfix)

        with open(file_name, "w", encoding="utf-8") as f:
            f.write("test")

        assert os.path.exists(file_name)
        assert os.path.isfile(file_name)
        assert os.path.getsize(file_name) != 0


def test_get_file_increasing_postfix():
    with environment():
        file_names = []

        name = "file.ext"
        postfix = "post"
        path = ".tmp"

        for i in range(10):
            file_name = get_file(os.path.join(path, name), postfix)

            with open(file_name, "w", encoding="utf-8") as f:
                f.write(f"test_{i}")

            file_names.append(file_name)

        assert len(file_names) == len(os.listdir(path))


def test_get_file_overwrite():
    file_name = ""
    name = "file.ext"
    path = ".tmp"

    with environment():
        for i in range(10):
            file_name = get_file(os.path.join(path, name), overwrite=True)

            with open(file_name, "w", encoding="utf-8") as f:
                f.write(f"{i}")

        assert len(os.listdir(path)) == 1
        with open(file_name, "r", encoding="utf-8") as f:
            value = int(f.read()[0])

        assert value == 9
