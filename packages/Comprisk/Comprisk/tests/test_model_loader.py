import unittest
from unittest.mock import patch

from Dataflux.pytorch.dummy_dataset import DummyDataset, DummyMode

from Comprisk.helper.structure import class_for_name
from Comprisk.helper import ModelIterator

class TestModelLoader(unittest.TestCase):
    def setUp(self):
        self.features_in = ["A", "B", "C"]
        self.factor = 1.5

        model_class = class_for_name("Comprisk.model.dummy", "DummyModel")
        self.model = model_class(self.factor, self.features_in, self.features_in)

    @patch("Comprisk.model.normalize.NormalizeModule")
    def test_forward_forecast_equal_stride(self, normalize_mock):
        normalize_mock.apply.return_value = lambda x, y: (x, y)
        normalize_mock.reverse.return_value = lambda x, y: (x, y)

        dataset = DummyDataset(
            DummyMode.RANDOM,
            n_samples=100,
            n_features=3,
            n_sequence=40,
            n_future=20,
            stride=20
        )

        model_class = class_for_name("Comprisk.model.dummy", "DummyModel")
        model = model_class(self.factor, self.features_in, self.features_in)
        iterator = ModelIterator(model, dataset)
        prediction, inputs = iterator.run()

        self.assertTrue((abs(prediction - inputs * self.factor) < 1e-5).all().all())

    @patch("Comprisk.model.normalize.NormalizeModule")
    def test_forward_pass_stride_equal_window(self, normalize_mock):
        """Test ModelLoader.predict_loop"""
        normalize_mock.apply.return_value = lambda x, y: (x, y)
        normalize_mock.reverse.return_value = lambda x, y: (x, y)

        dataset = DummyDataset(
            DummyMode.RANDOM,
            n_samples=100,
            n_features=3,
            n_sequence=20,
            stride=20
        )

        model_class = class_for_name("Comprisk.model.dummy", "DummyModel")
        model = model_class(self.factor, self.features_in, self.features_in)
        iterator = ModelIterator(model, dataset)
        prediction, inputs = iterator.run()

        self.assertTrue((abs(prediction - inputs * self.factor) < 1e-5).all().all())

    @patch("Comprisk.model.normalize.NormalizeModule")
    def test_forward_pass_stride_smaller_window(self, normalize_mock):
        """Test ModelLoader.predict_loop"""
        normalize_mock.apply.return_value = lambda x, y: (x, y)
        normalize_mock.reverse.return_value = lambda x, y: (x, y)

        dataset = DummyDataset(
            DummyMode.RANDOM,
            n_samples=100,
            n_features=3,
            n_sequence=20,
            stride=5
        )

        model_class = class_for_name("Comprisk.model.dummy", "DummyModel")
        model = model_class(self.factor, self.features_in, self.features_in)
        iterator = ModelIterator(model, dataset)
        prediction, inputs = iterator.run()

        self.assertTrue((abs(prediction - inputs * self.factor) < 1e-5).all().all())

    @patch("Comprisk.model.normalize.NormalizeModule")
    def test_forward_pass_inhomogeneous(self, normalize_mock):
        """Test ModelLoader.predict_loop"""
        normalize_mock.apply.return_value = lambda x, y: (x, y)
        normalize_mock.reverse.return_value = lambda x, y: (x, y)

        dataset = DummyDataset(
            DummyMode.RANDOM,
            n_samples=100,
            n_features=3,
            n_sequence=20,
            stride=3
        )

        model_class = class_for_name("Comprisk.model.dummy", "DummyModel")
        model = model_class(self.factor, self.features_in, self.features_in)
        iterator = ModelIterator(model, dataset)
        prediction, inputs = iterator.run()

        self.assertTrue((abs(prediction - inputs * self.factor) < 1e-5).all().all())