import os
from datetime import datetime

import torch

from Comprisk.anomaly import ScoreWrapper
from Comprisk.entrypoints import EntrypointBase, EntrypointContext
from Comprisk.helper import ONNXArgs


class SPExportEntrypoint(EntrypointBase):
    """This entrypoint exports the torch model into the ONNX format used by Xplore's backend."""
    def __init__(self):
        super().__init__()

        self._model_name = "support"

    def modify_model_metadata(self, context: EntrypointContext):
        """This function modifies the metadata of the exported ONNX file."""
        dataset = context.datamodule.trainset or context.datamodule.valset
        data_args = dataset.get_args()

        onnx_path = os.path.join(context.config_folder, self._model_name + ".onnx")
        args = ONNXArgs.load_from_model(onnx_path)
        args.machine_uuid = str(dataset.UUID)
        args.version = context.config.seed_everything
        args.creation_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        args.input_names = data_args.names_in
        args.output_names = data_args.names_out
        args.type = data_args.behavior.value
        args.set_delays(dataset)

        args.save_to_model(onnx_path)

    def __call__(self, context: EntrypointContext, *args, **kwargs) -> EntrypointContext:
        if os.path.exists(os.path.join(context.config_folder, "support.onnx")):
            return context

        # train the score wrapper
        valset = context.datamodule.val_dataloader()
        wrapper = ScoreWrapper(context.model, threshold=0.8, threshold_deviation=3)
        wrapper.train_wrapper(valset, verbose=True)
        wrapper.export_to(context.config_folder)

        # export finished model as ONNX file
        wrapper.sub_model.eval()
        wrapper.eval()

        # get a sample input first which is used to trace the model's grap
        input_sample, output_sample = context.datamodule.trainset[0]
        input_sample, output_sample = input_sample[0:1], output_sample[0:1]
        input_sample = torch.Tensor([input_sample])
        output_sample = torch.Tensor([output_sample])

        # then save the model into the ONNX format
        onnx_path = os.path.join(context.config_folder, "support.onnx")
        wrapper.to_onnx(
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
            },  # Support dynamic batch size
            opset_version=13,  # Choose an appropriate ONNX opset version
        )

        self.modify_model_metadata(context)
        return context
