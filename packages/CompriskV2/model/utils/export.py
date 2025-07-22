import os
from datetime import datetime

from Dataflux.pytorch import Dataset
from Dataflux import get_logger
from CompriskV2.model.utils import OnnxArgs
from CompriskV2.model import LinkedParameterModel

logger = get_logger(__name__)


def export(model: LinkedParameterModel, save_dir: str, dataset: Dataset, version: int = 0) -> None:
    """This function exports a model to the ONNX format. Moreover, it adds custom metadata to the model
    based on the given dataset (should be the one used for training).

    Args:
        model (LinkedParameterModel): The model to save.
        save_dir (str): The folder to save the model to.
        dataset (Dataset): The dataset use for training, here the dataset arguments were used to add metadata to the model.
        version (int): The model's version.
    """
    path = os.path.join(save_dir, "model.onnx")

    model = model.to("cpu")
    x, y = model.embedding
    output = model.predict_step(x, y)[1:]

    if not isinstance(output, tuple):
        output = output

    logger.info("Start export with embedding %s, %s", x.shape, y.shape)
    model.to_onnx(
        file_path=path,
        input_sample=model.embedding,
        input_names=["x", "y"],
        output_names=["y_scaled"] + [model.labels[i + 1] for i, _ in enumerate(output)],
        export_params=True,
        dynamic_axes={
            "x": {0: "batch_size"},
            "y": {0: "batch_size"},
        }
        | {model.labels[i + 1]: {0: "batch_size"} for i, _ in enumerate(output)},
    )

    onnx_args = OnnxArgs.load_from_model(path)

    data_args = dataset.get_args()
    onnx_args.machine_uuid = str(dataset.UUID)
    onnx_args.version = version
    onnx_args.creation_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    onnx_args.input_names = data_args.names_in
    onnx_args.output_names = data_args.names_out
    onnx_args.type = data_args.behavior.value

    onnx_args.set_value_corridor(dataset)
    onnx_args.save_to_model(path)

    logger.info("Exported checkpoint to '%s'", path)
