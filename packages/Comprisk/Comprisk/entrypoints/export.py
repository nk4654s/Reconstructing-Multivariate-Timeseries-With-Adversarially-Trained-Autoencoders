import os
import shutil

import torch
import yaml

from Comprisk.entrypoints.base import EntrypointContext, EntrypointBase
from Comprisk.helper import ModelLoader
from Dataflux import get_logger, logging_mode

logger = get_logger(__name__, "logfile", logging_mode)


class ExportEntrypoint(EntrypointBase):
    def __init__(self) -> None:
        super().__init__()

        self._folder_name = "export"
        self._model_name = "model.onnx"
        self._normalizer_name = "normalizer.yaml"
        self._transformer_name = "transformer.yaml"
        self._features_name = "features.yaml"

    def create_export_folder(self, context: EntrypointContext) -> str:
        path = os.path.join(context.config_folder, self._folder_name)
        if os.path.exists(path):
            shutil.rmtree(path)

        os.makedirs(path)
        return path

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        folder = self.create_export_folder(context)

        # load the (best) trained model back which is the one to export
        loader = ModelLoader(path=context.config_folder, model=context.model)
        if not loader.from_disk:
            logger.warning("Attempted to fit ScoreTransformer on an untrained model.")
            return context
        model = loader.model
        model.eval()

        # get a sample input first which is used to trace the model's grap
        input_sample, _ = context.datamodule.trainset[0]
        input_sample = torch.rand((1, *input_sample.shape))

        # then save the model into the ONNX format
        onnx_path = os.path.join(folder, self._model_name)
        model.to_onnx(
            file_path=onnx_path,
            input_sample=input_sample,
            export_params=True,
        )

        # save the ScoreTransformer parameters
        transformer = loader.get_score_transformer()
        transformer_export = transformer.export()
        with open(os.path.join(folder, self._transformer_name), "w") as f:
            yaml.safe_dump(transformer_export, f)

        # save the input features expected
        input_features = context.datamodule.valset.get_args().names_in
        output_features = context.datamodule.valset.get_args().names_out
        with open(os.path.join(folder, self._features_name), "w") as f:
            yaml.safe_dump(
                {"input_features": input_features, "output_features": output_features},
                f,
            )

        # copy the normalizer yaml
        data_dir = context.datamodule.valset.directory
        normalizer_file = os.path.join(data_dir, "metadata", "normalizer.yaml")
        shutil.copyfile(normalizer_file, os.path.join(folder, self._normalizer_name))

        logger.info("Exported model to %s", onnx_path)
        return context
