import os

from Dataflux import logging_mode
from Dataflux.logger import get_logger

from Comprisk.entrypoints import EntrypointBase, EntrypointContext
from Comprisk.helper import Report, ModelIterator

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

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        # load the dataset using the datamodule
        context.datamodule.set_path(None, context.config.data.validate)
        dataset = context.datamodule.valset

        onnx_path = os.path.join(context.config_folder, "model.onnx")
        iterator = ModelIterator(onnx_path, dataset)
        result = iterator.run(batch_size=128, num_workers=2)

        # create a report and generate the graphs
        name = "analysis" if not context.config.prefix else context.config.prefix
        report = Report(context.config_folder, name)
        report.generate_report_graphs(result)

        logger.info("Finished analyzing model on machine %s.", dataset.UUID)

        return context
