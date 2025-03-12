import os
import shutil
import unittest
from argparse import Namespace

import yaml
from Comprisk.entrypoints import EntrypointContext, FluxEntrypoint
from Comprisk.model.dummy import DummyModel

DIRECTORY = "Comprisk/tests"


def get_namespace(val_dir, f_in, f_out) -> Namespace:
    data = Namespace()
    data.train = None
    data.validate = val_dir
    data.features_in = f_in
    data.features_out = f_out
    config = Namespace()
    config.data = data
    config.force_upload = False
    return config


def mock_model(f_in, f_out) -> DummyModel:
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
        shutil.copytree(os.path.join(DIRECTORY, "test-debug-dataset"), self.val_dir)
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
            model=mock_model(features, features),
            datamodule=None,
            trainer=None,
            base_folder="temp",
            environment_="dev",
        )

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.context.base_folder)
