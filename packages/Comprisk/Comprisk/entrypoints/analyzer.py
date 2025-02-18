import os
import shutil
from uuid import UUID

from Comprisk.entrypoints import EntrypointBase, EntrypointContext
from Dataflux.pytorch import Dataset
from Dataflux.visualization import show_dataframe
from Dataflux.helper import get_sensors
from Dataflux import logging_mode
from Dataflux.logger import get_logger

from Comprisk.helper import ModelIterator


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

        self._scores_path = os.path.join(base_path, "scores.png")
        self._full_frame_path = os.path.join(base_path, "full.png")

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        dataset = self._get_dataset(context)

        # load the model with the helper class, then apply the model on the dataset
        loader = ModelLoader(context.config_folder, context.model)
        score = loader.get_score_transformer()
        logger.info("Loaded score transformer (mode=%s)", score.mode)
        logger.info("Run prediction loop on dataset (samples=%s).", len(dataset))
        target, predicted = loader.predict_loop(dataset)

        # visualize the loaded dataset
        self._create_analysis_folder(context)

        # transforms the loss between prediction and target to a score between 0 and 1.
        score.transform(predicted, target, visualize=self._scores_path, factors=0.1)
        logger.info("Downloading uuid to sensor table for sensor renaming in visualizations.")
        sensors_name_dict = get_sensors(UUID(dataset.UUID))
        predicted = predicted.rename(columns=sensors_name_dict, errors="ignore")
        target = target.rename(columns=sensors_name_dict, errors="ignore")

        # save the whole prediction of the dataset
        logger.debug("Visualizing whole dataset (samples=%s)", len(dataset))
        show_dataframe(
            [predicted, target],
            ["Prediction", "Target"],
            grid_size=5,
            save=self._full_frame_path,
        )

        logger.info(
            "Finished analyzing model %s on machine %s.",
            loader._model.__class__.__name__,
            dataset.UUID,
        )

        return context
