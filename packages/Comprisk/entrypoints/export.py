import os
from datetime import datetime

import torch

from Comprisk.helper import ONNXArgs
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
        dataset = context.datamodule.trainset or context.datamodule.valset
        data_args = dataset.get_args()

        onnx_path = os.path.join(context.config_folder, self._model_name)
        args = ONNXArgs.load_from_model(onnx_path)
        args.machine_uuid = str(dataset.UUID)
        args.version = context.config.seed_everything
        args.creation_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        args.input_names = data_args.names_in
        args.output_names = data_args.names_out
        args.type = data_args.behavior.value
        args.set_delays(dataset)

        args.save_to_model(onnx_path)

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
