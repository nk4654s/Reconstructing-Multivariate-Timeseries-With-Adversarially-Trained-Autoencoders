from Dataflux import logging_mode, DEBUG_MODE
from Dataflux.pytorch import DatasetTypes
from Dataflux.logger import get_logger

from Comprisk.entrypoints.base import EntrypointBase, EntrypointContext
from Comprisk.anomaly import ScoreWrapper
from Comprisk.helper import ModelLoader

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class TrainScoreTransformerEntrypoint(EntrypointBase):
    """
    This entrypoint loads a given model and runs the prediction loop on the validation set. Then, by calculating the
     loss between prediction and expected values, the Comprisk.anomaly.ScoreTransformer is fitted. Finally, the
     entrypoint stores the trained ScoreTransformer in the models output directory under `metadata`.
    """
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        dataset_args = context.datamodule.valset.get_args()
        stride = dataset_args.sequence_in
        if dataset_args.behavior != DatasetTypes.AUTOENCODER:
            stride = dataset_args.sequence_out

        logger.info(
            "Run prediction loop on valset (samples=%s) which can take some time.",
            len(context.datamodule.valset),
        )
        context.datamodule.valset.set_stride(stride)
        context.datamodule.setup(stage="fit")

        # load the (best) trained model back
        loader = ModelLoader(path=context.config_folder, model=context.model)
        if not loader.from_disk:
            logger.warning("Attempted to fit ScoreTransformer on an untrained model.")
            return context

        wrapper = ScoreWrapper(loader.model)
        wrapper.train_wrapper(context.datamodule.val_dataloader(), verbose=DEBUG_MODE)
        logger.info("Fitted ScoreTransformer successfully.")

        context.model = wrapper
        return context
