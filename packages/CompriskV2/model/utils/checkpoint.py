import os
from datetime import datetime

import torch

from Comprisk.helper import ONNXArgs
from CompriskV2.model import BaseModel, LinkedParameterModel
from CompriskV2.thunder import DataModule, CompriskCLI
from Dataflux import get_logger
from Dataflux.pytorch import Dataset

logger = get_logger(__name__)


def get_best_checkpoint(path: str, files: list[str]) -> str:
    files = [f for f in files if ".ckpt" in f]
    files.sort()
    return os.path.join(path, files[-1])


def import_from_string(class_path: str):
    components = class_path.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Checkpoint:
    def __init__(self, path: str):
        config_path = os.path.join(path, "config.yaml")
        if not os.path.exists(config_path):
            logger.error("Given path is not a model output folder: '%s'", path)
            raise ValueError("Could not finde 'config.yaml' in path")

        self._cli = CompriskCLI(
            LinkedParameterModel,
            DataModule,
            run=False,
            subclass_mode_model=True,
            parser_kwargs={"default_config_files": [config_path]},
        )

        self._embedding = (self._cli.datamodule.trainset[0][0].shape, self._cli.datamodule.trainset[0][1].shape)

        self._path = os.path.join(path, "checkpoints")
        files = os.listdir(self._path)
        ckpt_path = get_best_checkpoint(self._path, files)

        self._model_cls = import_from_string(self._cli.config.model.class_path)
        self._checkpoint = self._model_cls.load_from_checkpoint(ckpt_path, **self._cli.config.model.init_args)
        self._checkpoint.eval()

        logger.info("Loaded checkpoint '%s'", ckpt_path)

    @property
    def model(self) -> BaseModel:
        return self._checkpoint

    def export(self) -> None:
        path = os.path.dirname(self._path)
        path = os.path.join(path, "model.onnx")

        model = self.model.to("cpu")
        embedding = (
            torch.rand(*self._embedding[0], device="cpu").unsqueeze(0),
            torch.rand(*self._embedding[1], device="cpu").unsqueeze(0),
        )

        output = model.predict_step(*embedding)[1:]

        if not isinstance(output, tuple):
            output = output

        logger.info("Start export with embedding %s", self._embedding)
        model.to_onnx(
            file_path=path,
            input_sample=embedding,
            input_names=["x", "y"],
            output_names=["y_scaled"] + [f"y_{i}" for i, _ in enumerate(output)],
            export_params=True,
            dynamic_axes={
                "x": {0: "batch_size"},
                "y": {0: "batch_size"},
            }
            | {f"y_{i}": {0: "batch_size"} for i, _ in enumerate(output)},
        )

        dataset = self._cli.datamodule.trainset
        data_args = dataset.get_args()
        onnx_args = ONNXArgs.load_from_model(path)

        onnx_args.machine_uuid = str(dataset.UUID)
        onnx_args.version = self._cli.config.seed_everything
        onnx_args.creation_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        onnx_args.input_names = data_args.names_in
        onnx_args.output_names = data_args.names_out
        onnx_args.type = data_args.behavior.value

        onnx_args.set_value_corridor(dataset)
        onnx_args.save_to_model(path)

        logger.info("Exported checkpoint to '%s'", path)


if __name__ == "__main__":
    import sys

    Checkpoint(sys.argv.pop(1)).export()
