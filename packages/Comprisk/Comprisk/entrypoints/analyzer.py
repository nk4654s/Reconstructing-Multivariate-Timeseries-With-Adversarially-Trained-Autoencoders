import os
import shutil
from uuid import UUID

import matplotlib.pyplot as plt

from Comprisk.anomaly import ScoreWrapper
from Comprisk.entrypoints import EntrypointBase, EntrypointContext
from Comprisk.helper import ModelCheckpoint, ModelIterator
from Dataflux.pytorch import Dataset
from Dataflux.visualization import show_dataframe
from Dataflux.helper import get_sensors
from Dataflux import logging_mode
from Dataflux.logger import get_logger



# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class AnalyzerEntrypoint(EntrypointBase):
    """
    This entrypoint takes a trained model and runs inference it on the validation set. Then the results are plotted in
    a graph located in the model's output folder. To customize the folder's name, use the `--prefix folder_name`
    argument.

    If a different dataset should be used, then overwrite the validation set using `--data.validate`.
    """
    def __init__(self) -> None:
        super().__init__()

        self._scores_path = None
        self._scores_summary_path = None
        self._full_frame_path = None

    def _get_dataset(self, context: EntrypointContext) -> Dataset:
        # load the dataset using the datamodule
        context.datamodule.set_path(None, context.config.data.validate)
        return context.datamodule.valset

    def _create_analysis_folder(self, context: EntrypointContext):
        # create prefix sub-folders
        prefix = "analysis" if not context.config.prefix else context.config.prefix
        base_path = os.path.join(context.config_folder, prefix)

        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        os.makedirs(base_path)

        self._scores_summary_path = os.path.join(base_path, "scores-summary.png")
        self._scores_path = os.path.join(base_path, "scores.png")
        self._full_frame_path = os.path.join(base_path, "full.png")

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        dataset = self._get_dataset(context)

        # load the model with the helper class, then apply the model on the dataset
        checkpoint = ModelCheckpoint(context.config_folder)
        if not checkpoint.from_disk:
            logger.error("No model loaded from disk")
            return context

        wrapper = ScoreWrapper(checkpoint.get_model())
        wrapper.import_from(context.config_folder)

        # visualize the loaded dataset
        self._create_analysis_folder(context)

        logger.info("Downloading uuid to sensor table for sensor renaming in visualizations.")
        sensors_name_dict = get_sensors(UUID(dataset.UUID))

        # analyzing health score performance
        iterator = ModelIterator(wrapper, dataset)
        targets, inputs = iterator.run(batch_size=128)
        targets = targets.rename(columns=sensors_name_dict, errors="ignore")
        show_dataframe(
            [targets],
            ["Healthscore"],
            y_limit=(0, 1),
            grid_size=5,
            save=self._scores_path,
        )

        plt.close("all")
        plt.plot(targets.min(axis=1), label="min", linestyle="dashed", color="orange", alpha=0.6)
        plt.plot(targets.mean(axis=1), label="mean", color="teal", alpha=0.7)
        plt.plot(targets.max(axis=1), label="max", linestyle="dashdot", color="red", alpha=0.6)
        plt.title("Healthscores")
        plt.legend()
        plt.savefig(self._scores_summary_path)

        # save the whole prediction of the dataset
        logger.info("Run prediction loop on dataset (samples=%s).", len(dataset))
        iterator = ModelIterator(wrapper.sub_model, dataset)
        targets, inputs = iterator.run(batch_size=128)

        inputs = inputs.rename(columns=sensors_name_dict, errors="ignore")
        targets = targets.rename(columns=sensors_name_dict, errors="ignore")

        logger.debug("Visualizing whole dataset (samples=%s)", len(dataset))
        show_dataframe(
            [inputs, targets],
            ["Inputs", "Targets"],
            grid_size=5,
            save=self._full_frame_path,
        )

        logger.info(
            "Finished analyzing model %s on machine %s.",
            checkpoint.get_model().__class__.__name__,
            dataset.UUID,
        )

        return context
