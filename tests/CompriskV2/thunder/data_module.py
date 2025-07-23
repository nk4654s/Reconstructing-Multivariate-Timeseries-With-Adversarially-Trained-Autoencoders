from Dataflux.pytorch.dummy_dataset import DummyDataset, DummyMode
from CompriskV2.thunder import DataModule


def test_standardization_equal_features():
    dm = DataModule()
    dm._train = DummyDataset(DummyMode.RANDOM, 100, 3, 10, 1)
    mean, scale = dm.standardizer_params()
