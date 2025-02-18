import os

import torch

from Comprisk.anomaly import ScoreWrapper
from Comprisk.entrypoints.base import EntrypointContext, EntrypointBase
from Dataflux import get_logger, logging_mode

logger = get_logger(__name__, "logfile", logging_mode)


class ExportEntrypoint(EntrypointBase):
    def __init__(self) -> None:
        super().__init__()

        self._model_name = "model.onnx"
        self._graph_image = "model_graph"

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        model: ScoreWrapper = context.model
        model.eval()

        # get a sample input first which is used to trace the model's grap
        input_sample, output_sample = context.datamodule.trainset[0]
        input_sample = torch.rand((1, *input_sample.shape), device=context.model.device)
        output_sample = torch.rand(
            (1, *output_sample.shape), device=context.model.device
        )

        # then save the model into the ONNX format
        onnx_path = os.path.join(context.config_folder, self._model_name)
        model.to_onnx(
            file_path=onnx_path,
            input_sample=(input_sample, output_sample),
            input_names=["x", "y"],
            export_params=True,
        )

        logger.info("Exported model to %s", onnx_path)
        return context
