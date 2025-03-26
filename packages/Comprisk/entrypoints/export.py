import json
import os
from datetime import datetime

import onnx
import torch
from Dataflux import get_logger, logging_mode

from Comprisk.anomaly import ScoreWrapper
from Comprisk.entrypoints.base import EntrypointBase, EntrypointContext

logger = get_logger(__name__, "logfile", logging_mode)


class ExportEntrypoint(EntrypointBase):
    def __init__(self) -> None:
        super().__init__()

        self._model_name = "model.onnx"
        self._graph_image = "model_graph"

    def modify_model_metadata(self, context: EntrypointContext):
        dataset = context.datamodule.valset
        args = dataset.get_args()

        onnx_path = os.path.join(context.config_folder, self._model_name)
        model = onnx.load(onnx_path)

        enc = json.encoder.JSONEncoder()

        model.domain = str(dataset.UUID)
        model.model_version = context.config.seed_everything
        model.producer_name = "NEA-X"
        model.doc_string = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        meta = model.metadata_props.add()
        meta.key = "input_names"
        meta.value = enc.encode(args.names_in)

        meta = model.metadata_props.add()
        meta.key = "output_names"
        meta.value = enc.encode(args.names_out)

        meta = model.metadata_props.add()
        meta.key = "type"
        meta.value = args.behavior.value

        meta = model.metadata_props.add()
        meta.key = "delays"
        dataset = context.datamodule.trainset or context.datamodule.valset
        delays = {}
        for key, value in dataset.metadata.state_alpha.items():
            delays[key] = value
            delays[key]["delay"] *= 1_000 # transform to milliseconds

        meta.value = enc.encode(delays)

        onnx.save(model, onnx_path)

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
            output_names=["score", "y_hat"],
            export_params=True,
            dynamic_axes={
                "x": [0],
                "y": [0],
                "score": [0],
                "y_hat": [0],
            },
        )

        self.modify_model_metadata(context)

        logger.info("Exported model to %s", onnx_path)
        return context
