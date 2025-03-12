import unittest

from Comprisk.helper.structure import class_for_name
from Comprisk.model.dummy import DummyModel


class TestModelLoader(unittest.TestCase):
    def test_class_for_name(self):
        dummy_class = class_for_name("Comprisk.model.dummy", "DummyModel")
        self.assertEqual(DummyModel, dummy_class)

    def test_class_for_name_not_existent(self):
        def fail():
            class_for_name("Comprisk.model.not_available", "Error")

        self.assertRaises(ModuleNotFoundError, fail)