import os

import torch
import yaml

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
        model.sub_model.eval()
        model.eval()

        # get a sample input first which is used to trace the model's grap
        input_sample, output_sample = context.datamodule.trainset[0]
        input_sample = torch.Tensor([input_sample])
        output_sample = torch.Tensor([output_sample])

        # then save the model into the ONNX format
        onnx_path = os.path.join(context.config_folder, self._model_name)
        model.to_onnx(
            file_path=onnx_path,
            input_sample=(input_sample, output_sample),
            input_names=["x", "y"],
            export_params=True,
        )

        # expand the model.yaml
        with open(os.path.join(context.config_folder, "model.yaml"), "r") as f:
            meta = yaml.safe_load(f)
            meta["cache_size"] = context.datamodule.trainset.get_args().sequence_in

        with open(os.path.join(context.config_folder, "model.yaml"), "w") as f:
            yaml.safe_dump(meta, f)

        logger.info("Exported model to %s", onnx_path)
        return context
