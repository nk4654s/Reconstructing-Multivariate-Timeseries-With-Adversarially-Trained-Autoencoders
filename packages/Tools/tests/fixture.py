import os
import shutil
import yaml
import uuid
from contextlib import ContextDecorator

from Dataflux.pytorch.dummy_dataset import DummyDataset, DummyMode
from CompriskV2.model.autoregressive import AutoregressiveAutoencoder
from CompriskV2.model.utils import OnnxArgs
from CompriskV2.model.utils.export import export


class ProductionFolder(ContextDecorator):
    def __init__(self) -> None:
        self.path = ".tmp"
        self.prod = os.path.join(self.path, "production")
        self.dev = os.path.join(self.path, "dev")

    def _generate_onnx(self, model_types: list[str], env: str) -> list[OnnxArgs]:
        names = [str(uuid.uuid4()) for _ in range(3)]
        dataset = DummyDataset(DummyMode.RANDOM, 100, names, names, 10, 10)
        m_uuid = str(dataset.UUID)
        args = dataset.get_args()

        model = AutoregressiveAutoencoder(
            features_in=args.names_in,
            features_out=args.names_out,
            sequence_in=10,
            sequence_out=1,
        )

        models = []
        for model_type in model_types:
            path = os.path.join(env, "models", f"{m_uuid}_{model_type}.onnx")
            export(model, path, dataset)
            models.append(OnnxArgs.load_from_model(path))

        return models

    def _generate_output_folder(self) -> OnnxArgs:
        names = [str(uuid.uuid4()) for _ in range(3)]
        dataset = DummyDataset(DummyMode.RANDOM, 100, names, names, 10, 10)
        m_uuid = str(dataset.UUID)
        args = dataset.get_args()

        model = AutoregressiveAutoencoder(
            features_in=args.names_in,
            features_out=args.names_out,
            sequence_in=10,
            sequence_out=1,
        )

        folder = os.path.join(self.dev, m_uuid)
        os.mkdir(folder)
        export(model, os.path.join(folder, "model.onnx"), dataset)

        with open(os.path.join(folder, "config.yaml"), "w") as f:
            yaml.safe_dump(
                {
                    "env": "dev",
                    "data": {
                        "train": os.path.join(folder, "train.yaml"),
                        "validate": os.path.join(folder, "validate.yaml"),
                    },
                },
                f,
            )

        with open(os.path.join(folder, "train.yaml"), "w") as f:
            f.write("test")

        with open(os.path.join(folder, "validate.yaml"), "w") as f:
            f.write("test")

        return OnnxArgs.load_from_model(os.path.join(folder, "model.onnx")), folder

    def __enter__(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

        os.mkdir(self.path)
        os.mkdir(self.dev)
        os.makedirs(os.path.join(self.prod, "models"))
        os.makedirs(os.path.join(self.prod, "configs"))

        self.deployed = self._generate_onnx(["model"], self.prod)
        for _ in range(2):
            self.deployed += self._generate_onnx(["model", "support"], self.prod)

        self.undeployed = self._generate_output_folder()

        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.path)
