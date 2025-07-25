import os
import shutil
import yaml
import uuid
import sys
import inspect
from contextlib import ContextDecorator

from CompriskV2.model import LinkedParameterModel
from CompriskV2.thunder import DataModule, CompriskCLI
from Dataflux.pytorch.dummy_dataset import DummyDataset, DummyMode


def get_class_path(cls_name) -> str:
    return f"{inspect.getmodule(cls_name).__name__}.{cls_name.__name__}"


class CliContext(ContextDecorator):
    def __init__(self, model_cls) -> None:
        self._model_cls = model_cls

    def __enter__(self):
        self.path = ".tmp"
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

        os.mkdir(self.path)
        config_path = os.path.join(self.path, "config.yaml")
        with open(config_path, "w") as f:
            sensors = [str(uuid.uuid4()) for _ in range(3)]
            data = {
                "env": "test",
                "tag": "test",
                "trainer": {
                    "max_epochs": 1,
                },
                "model": {
                    "class_path": get_class_path(self._model_cls),
                },
                "data": {
                    "features_in": [_ for _ in sensors],
                    "features_out": [_ for _ in sensors],
                    "sequence_in": 10,
                    "sequence_out": 1,
                },
            }
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
        args = self.dataset.get_args()
        self.cli.datamodule._train = self.dataset
        self.cli.datamodule._val = self.dataset
        self.cli.datamodule.features_in = args.names_in
        self.cli.datamodule.features_out = args.names_out

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
