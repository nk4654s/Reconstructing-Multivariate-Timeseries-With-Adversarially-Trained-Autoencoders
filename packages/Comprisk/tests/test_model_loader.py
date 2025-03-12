import os
import unittest
from unittest.mock import patch

from Comprisk.helper import ModelIterator
from Dataflux.pytorch.dummy_dataset import DummyDataset, DummyMode


class TestModelLoader(unittest.TestCase):
    def setUp(self):
        self.features_in = ["A", "B", "C"]
        self.factor = 1.5

        test_folder = os.path.dirname(__file__)
        self.model_path = os.path.join(test_folder, "dummy.onnx")

    @patch("Comprisk.model.normalize.NormalizeModule")
    def test_forward_forecast_equal_stride(self, normalize_mock):
        dataset = DummyDataset(
            DummyMode.RANDOM,
            n_samples=200,
            n_features=22,
            n_sequence=60,
            n_future=0,
            stride=1
        )

        iterator = ModelIterator(self.model_path, dataset)
        result = iterator.run(batch_size=1)

        self.assertTrue((abs(result.scores - result.expected * 1.5) < 1e-5).all().all())
