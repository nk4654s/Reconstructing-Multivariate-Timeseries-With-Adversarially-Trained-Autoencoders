import os.path

from Dataflux import logging_mode, DEBUG_MODE
from Dataflux.pytorch import DatasetTypes
from Dataflux.logger import get_logger

from Comprisk.entrypoints.base import EntrypointBase, EntrypointContext
from Comprisk.anomaly import ScoreWrapper
from Comprisk.helper import ModelCheckpoint, ModelIterator

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
        # adjust the dataset's stride to be as long as an output window for
        # a faster transformer fit
        dataset_args = context.datamodule.valset.get_args()
        stride = dataset_args.sequence_in
        if dataset_args.behavior != DatasetTypes.AUTOENCODER:
            stride = dataset_args.sequence_out

        n = len(context.datamodule.valset)
        logger.info("Fitting score transformer on n=%s samples.", n)
        context.datamodule.valset.set_stride(stride)
        context.datamodule.setup(stage="fit")

        # load the best checkpoint, modifying an exported ONNX file does not work as the
        # score transformer is tightly integrated
        checkpoint = ModelCheckpoint(path=context.config_folder, model=context.model)
        if not checkpoint.from_disk:
            logger.warning("Attempted to fit ScoreTransformer on an untrained model.")
            return context

        # wrap the model with a score transformer such that the fitted values are
        # included in the model graph. This is required for ONNX
        wrapper = ScoreWrapper(checkpoint.get_model())
        wrapper.train_wrapper(context.datamodule.val_dataloader(), verbose=DEBUG_MODE)
        wrapper.export_to(context.config_folder)

        # plot a bit
        iterator = ModelIterator(wrapper, context.datamodule.valset)
        score, inputs = iterator.run(batch_size=256)
        from matplotlib.pyplot import figure
        import matplotlib.pyplot as plt
        figure(figsize=(22, 10), dpi=60)

        plt.plot(score.min(axis=1), label="min", linestyle="dashed", color="orange", alpha=0.6)
        plt.plot(score.mean(axis=1), label="mean", color="teal", alpha=0.7)
        plt.plot(score.max(axis=1), label="max", linestyle="dashdot", color="red", alpha=0.6)
        plt.title("Healthscores: Validation")
        plt.legend()
        plt.savefig(os.path.join(context.config_folder, "valset.png"))

        iterator = ModelIterator(wrapper, context.datamodule.trainset)
        score, inputs = iterator.run(batch_size=256)

        plt.plot(score.min(axis=1), label="min", linestyle="dashed", color="orange", alpha=0.6)
        plt.plot(score.mean(axis=1), label="mean", color="teal", alpha=0.7)
        plt.plot(score.max(axis=1), label="max", linestyle="dashdot", color="red", alpha=0.6)
        plt.title("Healthscores: Training")
        plt.legend()
        plt.savefig(os.path.join(context.config_folder, "trainset.png"))

        logger.info("Fitted ScoreTransformer successfully.")

        context.model = wrapper
        return context
