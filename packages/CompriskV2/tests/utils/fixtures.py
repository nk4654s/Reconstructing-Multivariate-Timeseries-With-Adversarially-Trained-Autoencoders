import os
import shutil
import yaml
import sys
from contextlib import ContextDecorator

from CompriskV2.tests.utils.config import mock_config
from CompriskV2.model import LinkedParameterModel
from CompriskV2.thunder import DataModule, CompriskCLI
from Dataflux.pytorch.dummy_dataset import DummyDataset, DummyMode


class CliContext(ContextDecorator):
    def __init__(self, model_cls: type, **kwargs) -> None:
        self._model_cls = model_cls
        self._data = kwargs or {}

    def __enter__(self):
        self.path = ".tmp"
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

        os.mkdir(self.path)
        config_path = os.path.join(self.path, "config.yaml")
        with open(config_path, "w") as f:
            data = mock_config(self._model_cls, **self._data)
            yaml.safe_dump(data, f)

        app_name = sys.argv.pop(0)
        sys.argv = [app_name]
        self.cli = CompriskCLI(
            LinkedParameterModel,
            DataModule,
            run=False,
            subclass_mode_model=True,
            parser_kwargs={"default_config_files": [config_path]},
        )

        self.dataset = DummyDataset(DummyMode.RANDOM, 100, 3, 10, 1)
        self.dataset._args.names_in = self.cli.datamodule.features_in
        self.dataset._args.names_out = self.cli.datamodule.features_out
        names = self.cli.datamodule.features_in + self.cli.datamodule.features_out
        self.dataset._args.names = list(set(names))
        self.cli.datamodule._train = self.dataset
        self.cli.datamodule._val = self.dataset

        self.ckpt_path = os.path.join(self.path, "checkpoints")
        os.mkdir(self.ckpt_path)
        self.cli.trainer.strategy.connect(self.cli.model)
        self.cli.trainer.save_checkpoint(os.path.join(self.ckpt_path, "loss=99.ckpt"))

        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.path)
        if os.path.exists("environment/test"):
            shutil.rmtree("environment/test")


if __name__ == "__main__":
    with CliContext(LinkedParameterModel) as c:
        print(c.cli.model.__class__)
