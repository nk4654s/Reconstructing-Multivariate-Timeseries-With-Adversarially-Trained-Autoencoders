import os
from datetime import datetime

import torch

from Comprisk.helper import ONNXArgs
from Dataflux import get_logger, logging_mode

from Comprisk.entrypoints.base import EntrypointBase, EntrypointContext

logger = get_logger(__name__, "logfile", logging_mode)


class ExportEntrypoint(EntrypointBase):
    def __init__(self, name: str = "model") -> None:
        super().__init__()

        self._model_name = f"{name}.onnx"

    def modify_model_metadata(self, context: EntrypointContext):
        """This function modifies the metadata of the exported ONNX file."""
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
        args.set_value_corridor(dataset)

        args.save_to_model(onnx_path)

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        # export finished model as ONNX file
        context.model.sub_model.eval()
        context.model.eval()

        # get a sample input first which is used to trace the model's grap
        input_sample, output_sample = context.datamodule.trainset[0]
        input_sample, output_sample = input_sample, output_sample
        input_sample = torch.Tensor([input_sample])
        output_sample = torch.Tensor([output_sample])

        # match devices
        context.model.to(input_sample.device)
        logger.debug("Running model export on '%s'", input_sample.device)
        logger.debug("Export input  sample: %s", input_sample.shape)
        logger.debug("Export output sample: %s", output_sample.shape)

        # then save the model into the ONNX format
        onnx_path = os.path.join(context.config_folder, self._model_name)
        context.model.to_onnx(
            file_path=onnx_path,
            input_sample=(input_sample, output_sample),
            input_names=["x", "y"],
            output_names=["score", "y_hat"],
            export_params=True,
            dynamic_axes={
                "x": {0: "batch_size"},
                "y": {0: "batch_size"},
                "score": {0: "batch_size"},
                "y_hat": {0: "batch_size"},
            },
            #opset_version=13,
        )

        self.modify_model_metadata(context)
        logger.info("Exported model to %s", onnx_path)
        return context
