import os
import sys

from CompriskV2.model import LinkedParameterModel
from CompriskV2.thunder import DataModule, CompriskCLI
from CompriskV2.model.utils.export import export
from CompriskV2.model.parser import get_class_by_name
from Dataflux import get_logger

logger = get_logger(__name__)


def get_best_checkpoint(path: str, files: list[str]) -> str:
    files = [f for f in files if ".ckpt" in f]
    files = sorted(files, key=lambda x: x.split("loss=")[-1].replace(",", "."))
    return os.path.join(path, files[0])


class Checkpoint:
    """This class loads the best checkpoint of a model given its output folder path.

    Args:
        path (Str): The model's output directory (should contain a `config.yaml` file).
    """

    def __init__(self, path: str):
        config_path = os.path.join(path, "config.yaml")
        if not os.path.exists(config_path):
            logger.error("Given path is not a model output folder: '%s'", path)
            raise ValueError("Could not find 'config.yaml' in path")

        sys.argv = [sys.argv[0]]
        self._cli = CompriskCLI(
            LinkedParameterModel,
            DataModule,
            run=False,
            subclass_mode_model=True,
            parser_kwargs={"default_config_files": [config_path]},
        )

        self._path = os.path.join(path, "checkpoints")
        files = os.listdir(self._path)
        ckpt_path = get_best_checkpoint(self._path, files)

        self._model_cls = get_class_by_name(self._cli.config.model.class_path)
        self._checkpoint = self._model_cls.load_from_checkpoint(ckpt_path, **self._cli.config.model.init_args)
        self._checkpoint.eval()

        logger.info("Loaded checkpoint '%s'", ckpt_path)

    @property
    def model(self) -> LinkedParameterModel:
        return self._checkpoint

    def export(self) -> None:
        path = os.path.dirname(self._path)

        export(
            model=self.model,
            path=path,
            dataset=self._cli.datamodule.trainset,
            version=self._cli.config.seed_everything,
        )


if __name__ == "__main__":
    import sys

    Checkpoint(sys.argv.pop(1)).export()
