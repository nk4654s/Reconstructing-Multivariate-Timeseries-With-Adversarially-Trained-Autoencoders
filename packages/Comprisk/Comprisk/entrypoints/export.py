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

        self.folder_name = "export"
        self.model_name = "model.onnx"
        self._transformer_name = "transformer.yaml"

    def create_export_folder(self, context: EntrypointContext) -> str:
        path = os.path.join(context.config_folder, self.folder_name)
        if os.path.exists(path):
            shutil.rmtree(path)

        os.makedirs(path)
        return path

    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
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
        onnx_path = os.path.join(folder, self.model_name)
        model.to_onnx(
            file_path=onnx_path,
            input_sample=input_sample,
            export_params=True,
        )

        # save the ScoreTransformer parameters
        transformer = loader.get_score_transformer()
        transformer_export = transformer.export()
        with open(os.path.join(folder, self._transformer_name)) as f:
            yaml.safe_dump(transformer_export)

        # export the sensor uuid to ai uuid mapping


        logger.info("Exported model to %s", onnx_path)
        return context
