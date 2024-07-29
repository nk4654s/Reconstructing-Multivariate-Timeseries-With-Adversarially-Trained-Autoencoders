"""Load a model using a CLI."""

from lightning.pytorch.loggers import TensorBoardLogger
import pandas as pd

# simple demo classes for your convenience
from Dataflux.pytorch import Metadata
from Dataflux.visualization import show_dataframe
from Dataflux.helper import get_logger

from Comprisk.model import BaseModel
from Comprisk.data import DataModule
from Comprisk.cli import CompriskCLI

logger = get_logger(__name__)


def cli_main():
    """
    The main entry point for the Comprisk CLI. This function initializes the
    CompriskCLI object and sets up the command-line interface.
    """
    logger.debug("Starting CompriskCLI")
    cli = CompriskCLI(BaseModel, DataModule, subclass_mode_model=True, run=False)

    # set up the logger
    logger.debug("Defining TensorBoard logger for %s", cli.model.__class__.__name__)
    tb_logger = TensorBoardLogger("lightning_logs", name=cli.model.__class__.__name__)
    cli.trainer.logger = tb_logger

    # run methods before fit
    logger.debug("CompriskCLI prepare model for inference.")
    model_class = cli.model.__class__

    # predict on test dataset
    logger.debug("Load the model from file")
    model = model_class.load_from_checkpoint("lightning_logs/AutoEncoder/version_2/epoch=1-step=11412.ckpt")
    model.eval()

    # 1. get dataset configuration: start, end, step size (size of input window)
    dataset = cli.datamodule.dataset
    meta: Metadata = dataset.get_meta()

    # 2. iterate over dataset, with shuffle deactivated
    predicted = []
    inputs = []
    for i in range(0, len(dataset), meta.num_input_sequence):
        sensors, weather, _ = dataset[i]
        inputs.append(sensors)

        # 3. pass sample into model, store prediction in list/numpy/dataframe
        target = model.predict_step({"sensors": sensors, "weather": weather})
        predicted.append(target)

    # 4. finalize dataframe, give columns names matching input dataframe column names
    df_predicted = {}
    df_inputs = {}
    for i, name in enumerate(meta.column_names):
        df_predicted[name] = [_[i] for _ in predicted]
        df_inputs[name] = [_[i] for _ in inputs]

    df_predicted = pd.DataFrame(df_predicted)
    df_inputs = pd.DataFrame(df_inputs)

    # 5. visualize
    sensors = [
        "SaugtemperaturZweiteStufeTETIT74341 (°C)",
        "DifferenzdruckTriebwerksölPDIT74553 (bar)",
        "KühlwassertemperaturnachKühlerTETIT74422 (°C)",
        "ÖltemperaturVorlaufTriebwerkTETIT74538 (°C)",
        "Motorlager (°C)",
        "SaugdruckErsteStufePIT74319 (bar)",
        "SchwingungsmessungTriebwerkVT74280 (mm/s)",
        "KompressordrehzahlSE74201 (rpm)",
        "EndtemperaturErsteStufeTETIT74334 (°C)",
        "EndtemperaturZweiteStufeTETIT74371 (°C)",
    ]
    show_dataframe(sensors, [df_predicted, df_inputs], grid_size=3, save="prediction.png")


if __name__ == "__main__":
    cli_main()
