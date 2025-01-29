import unittest

from Dataflux.pytorch.dummy_dataset import DummyDataset, DummyMode

from Comprisk.helper.structure import class_for_name
from Comprisk.helper import  ModelLoader

class TestModelLoader(unittest.TestCase):
    def setUp(self):
        self.features_in = ["A", "B", "C"]
        self.factor = 1.5

        model_class = class_for_name("Comprisk.model.dummy", "DummyModel")
        self.model = model_class(self.factor, self.features_in, self.features_in)

        self.loader = ModelLoader(model=self.model)

    def test_forward_forecast_equal_stride(self):
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
        loader = ModelLoader(model=model)
        inputs, prediction = loader.predict_loop(dataset, batch_size=4, num_workers=1)
        self.assertTrue((abs(prediction - inputs * self.factor) < 1e-5).all().all())

    def test_forward_pass_stride_equal_window(self):
        """Test ModelLoader.predict_loop"""
        dataset = DummyDataset(
            DummyMode.RANDOM,
            n_samples=100,
            n_features=3,
            n_sequence=20,
            stride=20
        )

        inputs, prediction = self.loader.predict_loop(dataset, batch_size=4, num_workers=1)
        self.assertTrue((abs(prediction - inputs * self.factor) < 1e-5).all().all())

    def test_forward_pass_stride_smaller_window(self):
        """Test ModelLoader.predict_loop"""
        dataset = DummyDataset(
            DummyMode.RANDOM,
            n_samples=100,
            n_features=3,
            n_sequence=20,
            stride=5
        )

        inputs, prediction = self.loader.predict_loop(dataset, batch_size=4, num_workers=1)
        self.assertTrue((abs(prediction - inputs * self.factor) < 1e-5).all().all())

    def test_forward_pass_inhomogeneous(self):
        """Test ModelLoader.predict_loop"""
        dataset = DummyDataset(
            DummyMode.RANDOM,
            n_samples=100,
            n_features=3,
            n_sequence=20,
            stride=3
        )

        inputs, prediction = self.loader.predict_loop(dataset, batch_size=4, num_workers=1)
        self.assertTrue((abs(prediction - inputs * self.factor) < 1e-5).all().all())