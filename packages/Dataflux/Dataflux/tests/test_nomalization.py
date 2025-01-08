"""Dataflux/helper/scaler.py"""
import unittest
import pandas as pd
import numpy as np

from Dataflux.helper import FluxScaler
from Dataflux.helper.scaler import ScalerMethod

TOLERANCE = 1e-5

class TestMachineSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.output = np.random.rand(5, 2)
        self.input = pd.DataFrame(self.output)

        self.output_list = [np.random.rand(5, 2) for _ in range(5)]
        self.input_list = [pd.DataFrame(o) for o in self.output_list]

    def test_minmax(self) -> None:
        """Function tests the internal MinMax scaler of the FluxScaler."""
        scaler = FluxScaler(method=ScalerMethod.NORMALIZE)
        normed = scaler.fit(self.input).transform(self.input)
        reverted = scaler.revert(normed.to_numpy())

        self.assertTrue((np.min(normed.to_numpy(), axis=0) == -1).all(), "Minimum after normalization is not `-1`")
        self.assertTrue((np.max(normed.to_numpy(), axis=0) == 1).all(), "Maximum after normalization is not `+1`")
        np.testing.assert_allclose(self.output, reverted, rtol=TOLERANCE, err_msg="MinMax normalization/revert does not match input")

    def test_minmax_partial_fit(self) -> None:
        """Function tests fitting the FluxScaler internal MinMax scaler on multiple inputs
        to properly normalize many inputs."""
        scaler = FluxScaler(method=ScalerMethod.NORMALIZE)

        # train
        for df in self.input_list:
            scaler.fit(df)
        
        # normalize/revert
        normed = [scaler.transform(df) for df in self.input_list]
        reverted = [scaler.revert(n.to_numpy()) for n in normed]

        self.assertTrue((np.min([np.min(n.to_numpy(), axis=0) for n in normed], axis=0) == -1).all(), "Minimum after normalization is not `-1`")
        self.assertTrue((np.max([np.max(n.to_numpy(), axis=0) for n in normed], axis=0) == 1).all(), "Maximum after normalization is not `1`")

        for i, r in enumerate(reverted):
            np.testing.assert_allclose(self.output_list[i], r, rtol=TOLERANCE, err_msg="MinMax normalization/revert does not match input")
