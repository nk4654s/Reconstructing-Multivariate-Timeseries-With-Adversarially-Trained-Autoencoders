import os
import shutil

import torch
import yaml

from Comprisk.entrypoints.base import EntrypointContext, EntrypointBase
from Comprisk.helper import ModelLoader
from Dataflux import get_logger, logging_mode
from Dataflux.helper.sensor_register import (
    score_uuid_mapping,
    gen_health_score_sensor_ids,
)

logger = get_logger(__name__, "logfile", logging_mode)


class ExportEntrypoint(EntrypointBase):
    def __init__(self) -> None:
        super().__init__()

        self.folder_name = "export"
        self.model_name = "model.onnx"
        self._transformer_name = "transformer.yaml"
        self._mapping_name = "mapping.yaml"
        self._features_name = "features.yaml"

    def create_export_folder(self, context: EntrypointContext) -> str:
        path = os.path.join(context.config_folder, self.folder_name)
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
        onnx_path = os.path.join(folder, self.model_name)
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

        # export the sensor uuid to AI uuid mapping
        output_features = context.datamodule.valset.get_args().names_out
        mapping, drop = score_uuid_mapping(
            score_uuids=tuple(output_features),
            machine_id=context.datamodule.valset.UUID,
            ai_tag="Healthscore",
        )

        # maybe there are not all UUIDs defined yet, but still include them
        for s_uuid in drop:
            ai_uuid = str(gen_health_score_sensor_ids(s_uuid, [0])[0])
            mapping[s_uuid] = ai_uuid
            logger.info("Add %s to mapping (result=%s)", s_uuid, ai_uuid)

        with open(os.path.join(folder, self._mapping_name), "w") as f:
            yaml.safe_dump(mapping, f)

        # save the input features expected
        input_features = context.datamodule.valset.get_args().names_in
        with open(os.path.join(folder, self._features_name), "w") as f:
            yaml.safe_dump(
                {"input_features": input_features, "output_features": output_features},
                f,
            )

        logger.info("Exported model to %s", onnx_path)
        return context
