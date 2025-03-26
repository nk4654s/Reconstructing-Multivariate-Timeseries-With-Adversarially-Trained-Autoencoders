import json
import os
from datetime import datetime

import onnx
import torch

from Comprisk.anomaly import ScoreWrapper
from Comprisk.entrypoints import EntrypointBase, EntrypointContext


class SPExportEntrypoint(EntrypointBase):
    """This entrypoint exports the torch model into the ONNX format used by Xplore's backend."""
    def __init__(self):
        super().__init__()

        self._model_name = "support"

    def modify_model_metadata(self, context: EntrypointContext):
        """This function modifies the metadata of the exported ONNX file."""
        dataset = context.datamodule.valset
        args = dataset.get_args()

        # load the exported model
        onnx_path = os.path.join(context.config_folder, self._model_name + ".onnx")
        model = onnx.load(onnx_path)

        enc = json.encoder.JSONEncoder()

        # store the machine UUID, seed and company name
        model.domain = str(dataset.UUID)
        model.model_version = context.config.seed_everything
        model.producer_name = "NEA-X"
        model.doc_string = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        # store the input and output sensor UUIDs required by this model as an input
        meta = model.metadata_props.add()
        meta.key = "input_names"
        meta.value = enc.encode(args.names_in)

        meta = model.metadata_props.add()
        meta.key = "output_names"
        meta.value = enc.encode(args.names_out)

        onnx.save(model, onnx_path)

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
