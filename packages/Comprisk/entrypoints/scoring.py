from Dataflux import DEBUG_MODE, logging_mode
from Dataflux.logger import get_logger
from Dataflux.pytorch import DatasetTypes

from Comprisk.anomaly import R2ScoreWrapper
from Comprisk.entrypoints.base import EntrypointBase, EntrypointContext
from Comprisk.model.utils import Checkpoint

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class TrainScoreTransformerEntrypoint(EntrypointBase):
    """
    This entrypoint loads a given model and runs the prediction loop on the validation set. Then, by calculating the
     loss between prediction and expected values, the Comprisk.anomaly.ScoreTransformer is fitted. Finally, the
     entrypoint stores the trained ScoreTransformer in the models output directory under `metadata`.

     Args:
         reduced_mode (bool): Whether to use the trainset in addition to the validation set to fit the
            ScoreTransformer. Defaults to False.
    """

    def __init__(self, reduced_mode: bool = False) -> None:
        super().__init__()

        self._reduced = reduced_mode

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        # adjust the dataset's stride to be as long as an output window for
        # a faster transformer fit
        dataset_args = context.datamodule.valset.get_args()
        stride = dataset_args.sequence_in
        if dataset_args.behavior != DatasetTypes.AUTOENCODER:
            stride = dataset_args.sequence_out

        context.datamodule.valset.set_stride(stride)
        context.datamodule.trainset.set_stride(stride)
        context.datamodule.setup(stage="fit")

        # load the best checkpoint, modifying an exported ONNX file does not work as the
        # score transformer is tightly integrated
        checkpoint = Checkpoint(path=context.trainer.log_dir)

        # wrap the model with a score transformer such that the fitted values are
        # included in the model graph. This is required for ONNX
        checkpoint.apply_score_wrapper(
            cls=R2ScoreWrapper,
            data=context.datamodule,
            reduced=self._reduced,
            verbose=DEBUG_MODE
        )

        #wrapper.export_to(context.config_folder)
        logger.info("Fitted ScoreTransformer successfully.")

        context.model = checkpoint.model
        return context
