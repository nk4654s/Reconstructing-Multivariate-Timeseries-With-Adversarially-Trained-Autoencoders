import os
import shutil
import unittest
from argparse import Namespace
from pyexpat import features

import yaml

from Comprisk.entrypoints import EntrypointContext, FluxEntrypoint, GeneratorEntrypoint
from Comprisk.model.dummy import DummyModel


def get_namespace(val_dir, f_in, f_out) -> Namespace:
    data = Namespace()
    setattr(data, "train", None)
    setattr(data, "validate", val_dir)
    setattr(data, "features_in", f_in)
    setattr(data, "features_out", f_out)
    config = Namespace()
    setattr(config, "data", data)
    return config


def mock_model(f_in, f_out):
    model = DummyModel(factor=1.5,
                       features_in=f_in,
                       features_out=f_out)
    return model


class TestEntrypointFlux(unittest.TestCase):
    def setUp(self):
        self.val_dir = "test-debug-dataset"
        self.prefix = "temp/dev"
        os.makedirs(self.prefix, exist_ok=True)

        self.val_dir = os.path.join(self.prefix, self.val_dir)
        shutil.copytree("test-debug-dataset", self.val_dir)
        self.config_file = os.path.join(self.val_dir, "metadata", "config.yaml")

        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)
            for module in config["modules"]:
                if module["class_path"] == "Dataflux.modules.features.Features":
                    features = module["init_args"]["include"]

        self.flux = FluxEntrypoint()
        self.context = EntrypointContext(
            config=get_namespace(self.val_dir, features, features),
            config_folder=None,
            model=None,
            datamodule=None,
            trainer=None,
            base_folder="temp",
            environment_="dev",
        )

    def test_generate_full(self):
        flux = FluxEntrypoint()
        flux(self.context)

        self.context.model = mock_model(
            f_in=self.context.datamodule.valset._features_in,
            f_out=self.context.datamodule.valset._features_out,
        )

        generate = GeneratorEntrypoint()
        generate._updater_cls = None
        generate(self.context)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.context.base_folder)
