import unittest
import numpy as np
from torch import tensor
from Comprisk.helper.structure import class_for_name


class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.sequence = 10
        self.batch_size = 4

        self.features_in = ["A", "B", "C"]
        self.factor = 1.5

        model_class = class_for_name("Comprisk.model.dummy", "DummyModel")
        self.model = model_class(self.factor, self.features_in, self.features_in)

        inputs = np.random.rand(
            self.batch_size, self.sequence, len(self.features_in)
        )

        self.torch_inputs = tensor(inputs)
        self.torch_outputs = self.torch_inputs * self.factor

    def test_forward_pass(self):
        """Test BaseModel.forward"""
        outputs = self.model(self.torch_inputs)
        self.assertTrue((self.torch_outputs == outputs).all())

    def test_predict_call(self):
        """Test BaseModel.predict"""
        outputs, _ = self.model.predict_step(self.torch_inputs, self.torch_inputs)
        self.assertTrue((self.torch_outputs == outputs).all())
