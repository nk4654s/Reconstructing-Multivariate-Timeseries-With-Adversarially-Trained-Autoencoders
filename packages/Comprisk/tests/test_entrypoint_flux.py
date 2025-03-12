import os
import shutil
import unittest
from argparse import Namespace
from copy import deepcopy

from Comprisk.entrypoints import EntrypointContext, FluxEntrypoint
from Dataflux.pytorch import Dataset

DIRECTORY = "tests"


def get_namespace(train_dir) -> Namespace:
    data = Namespace()
    data.train = train_dir
    data.validate = None
    config = Namespace()
    config.data = data
    return config


class TestEntrypointFlux(unittest.TestCase):
    def setUp(self):
        self.train_dir = "test-debug-dataset"

        self.flux = FluxEntrypoint()
        self.context = EntrypointContext(
            config=get_namespace(self.train_dir),
            config_folder=None,
            model=None,
            datamodule=None,
            trainer=None,
            base_folder="temp",
            environment_="dev",
        )
        self.prefix = "temp/dev"
        os.makedirs(self.prefix, exist_ok=True)

        shutil.copytree(os.path.join(DIRECTORY, self.train_dir), os.path.join(self.prefix, self.train_dir))
        self.config_file = os.path.join(self.prefix, self.train_dir, "metadata", "config.yaml")

    def test_get_paths__no_prefix_exists(self):
        path = self.flux._get_path(self.context, self.train_dir)
        self.assertEqual(os.path.join(self.prefix, self.train_dir), path)

    def test_get_paths_no_prefix_not_existent(self):
        path = self.flux._get_path(self.context, "random_name")
        self.assertEqual(None, path)

    def test_get_paths_no_prefix_not_existent_force(self):
        expected = os.path.join(self.prefix, "random_name")
        path = self.flux._get_path(self.context, "random_name", check_exists=False)
        self.assertEqual(expected, path)

    def test_get_paths_prefix_exists(self):
        expected = os.path.join(self.prefix, self.train_dir)
        got = self.flux._get_path(self.context, expected)
        self.assertEqual(expected, got)

    def test_get_paths_prefix_not_existent(self):
        expected = os.path.join(self.prefix, "random_name")
        got = self.flux._get_path(self.context, expected)
        self.assertEqual(None, got)

    def test_get_paths_prefix_not_existent_force(self):
        expected = os.path.join(self.prefix, "random_name")
        path = self.flux._get_path(self.context, expected, check_exists=False)
        self.assertEqual(expected, path)

    def test_adjust_config_from_invalid_path(self):
        # no errors if path is None or a folder
        got = self.flux._adjust_config_paths(self.context, None)
        self.assertEqual({}, got)

        got = self.flux._adjust_config_paths(
            self.context, self.flux._get_path(self.context, self.train_dir)
        )
        self.assertEqual({}, got)

        # error if path does not exist at all
        def fail():
            self.flux._adjust_config_paths(self.context, "random-name")

        self.assertRaises(FileNotFoundError, fail)

    def test_adjust_config_runs(self):
        path = self.flux._get_path(self.context, self.config_file)
        got = self.flux._adjust_config_paths(self.context, path)
        expected_folder = self.flux._get_path(self.context, self.train_dir)
        self.assertEqual(expected_folder, got["folder"])

    def test_flux_entrypoint_trainset(self):
        context = deepcopy(self.context)
        context.config.data.train = "test-debug-dataset"
        train_path = self.flux._get_path(self.context, self.context.config.data.train)
        val_path = self.flux._get_path(self.context, self.context.config.data.validate)
        context = self.flux(self.context)

        # the dataset's path must have changed
        self.assertEqual(context.config.data.train, train_path)
        self.assertEqual(context.config.data.validate, val_path)
        self.assertEqual(self.context, context)

        # dataset was empty and could not be loaded, values must be None
        self.assertIsInstance(context.datamodule.trainset, Dataset)
        self.assertEqual(None, context.datamodule.valset)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.context.base_folder)
