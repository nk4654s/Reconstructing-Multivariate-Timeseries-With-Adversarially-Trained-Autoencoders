import unittest
import torch
from torch import tensor
from Compmesh.model import RBFGridInterpolator

class TestRBFLinearInterpolator(unittest.TestCase):
    def test_simple_case(self):
        x = tensor([[2.0], [3.0], [4.0]], dtype=torch.float)
        y = tensor([[1.0], [4.0], [7.0]], dtype=torch.float)

        interpolator = RBFGridInterpolator(x, y)

        # Test the interpolator on new data
        test_input = tensor([[2.5], [3.5]], dtype=torch.float)
        expected_output = tensor([[2.5], [5.5]])

        test_output = interpolator(test_input)
        self.assertTrue(torch.allclose(test_output, expected_output), f"Expected {expected_output}, got {test_output}")

    def test_single_point(self):
        x = tensor([[1.0, 2.0]], dtype=torch.float)
        y = tensor([[3.0]], dtype=torch.float)

        def interpolator_init():
            return RBFGridInterpolator(x, y)

        self.assertRaises(AssertionError, interpolator_init)

    def test_multiple_dimensions(self):
        x = tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)
        y = tensor([[5.0], [6.0]], dtype=torch.float)

        interpolator = RBFGridInterpolator(x, y)

        test_input = tensor([[2.0, 3.0]], dtype=torch.float)
        expected_output = tensor([[5.5]])

        test_output = interpolator(test_input)
        self.assertTrue(torch.allclose(test_output, expected_output), f"Expected {expected_output}, got {test_output}")

    def test_high_dimensions(self):
        x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float)
        y = tensor([[7.0], [8.0]], dtype=torch.float)

        interpolator = RBFGridInterpolator(x, y)

        test_input = tensor([[1.0, 2.0, 3.0]], dtype=torch.float)
        expected_output = tensor([[7.0]])

        test_output = interpolator(test_input)
        self.assertTrue(torch.allclose(test_output, expected_output), f"Expected {expected_output}, got {test_output}")

if __name__ == '__main__':
    unittest.main()
