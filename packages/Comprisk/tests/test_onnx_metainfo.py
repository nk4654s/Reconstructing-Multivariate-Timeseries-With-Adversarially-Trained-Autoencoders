import random
import unittest

from Comprisk.helper import ONNXArgs


class TestONNXArgs(unittest.TestCase):
    def setUp(self):
        self.args = ONNXArgs()
        self.args.machine_uuid = "some-uuid"
        self.args.version = random.randint(0, 10)
        self.args.producer = "me"
        self.args.creation_date = "yesterday"
        self.args.input_names = ["a", "b"]
        self.args.output_names = ["b", "c"]
        self.args.type = "autoencoder"
        self.args.delays = {
            "some-other-uuid": {
                "lower": 0,
                "upper": 1
            }
        }

    def test_loading_args(self):
        self.args.save_to_model("dummy.onnx")
        args = ONNXArgs.load_from_model("dummy.onnx")
        self.assertTrue(self.args.equals(args))
