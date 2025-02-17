import os
import shutil

import torch
import yaml
import graphviz
from torchview import draw_graph

from Comprisk.wrapper import ScoreWrapper
from Comprisk.entrypoints.base import EntrypointContext, EntrypointBase
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
        self._graph_image = "model_graph"

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
        model: ScoreWrapper = context.model
        model.eval()

        # get a sample input first which is used to trace the model's grap
        input_sample, output_sample = context.datamodule.trainset[0]
        input_sample = torch.rand((1, *input_sample.shape), device=context.model.device)
        output_sample = torch.rand(
            (1, *output_sample.shape), device=context.model.device
        )

        onnx_path = os.path.join(folder, "deprecated_" + self._model_name)
        model.sub_model.to_onnx(
            file_path=onnx_path,
            input_sample=input_sample,
            input_names=["x"],
            export_params=True,
        )

        # try to save the model graph
        try:
            x_shape = (1, 3, input_sample.shape[-1])
            y_shape = (1, 3, output_sample.shape[-1])

            model_graph = draw_graph(
                context.model, input_size=(x_shape, y_shape), expand_nested=True
            )
            model_graph.visual_graph.render(
                os.path.join(folder, self._graph_image), format="png"
            )
        except graphviz.backend.execute.ExecutableNotFound as e:
            logger.warning("Unable to save model graph as image.")
            logger.warning("Graph render crashed due to %s", e)

        # then save the model into the ONNX format
        onnx_path = os.path.join(folder, self._model_name)
        model.to_onnx(
            file_path=onnx_path,
            input_sample=(input_sample, output_sample),
            input_names=["x", "y"],
            export_params=True,
        )

        # save the input features expected
        input_features = context.datamodule.valset.get_args().names_in
        output_features = context.datamodule.valset.get_args().names_out
        with open(os.path.join(folder, self._features_name), "w") as f:
            yaml.safe_dump(
                {"input_features": input_features, "output_features": output_features},
                f,
            )

        logger.info("Exported model to %s", onnx_path)
        return context
