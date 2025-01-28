import os
import pickle

from Dataflux import logging_mode
from Dataflux.pytorch import DatasetTypes
from Dataflux.logger import get_logger

from Comprisk.entrypoints.base import EntrypointBase, EntrypointContext
from Comprisk.anomaly import ScoreTransformer
from Comprisk.helper import ModelLoader

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class TrainScoreTransformerEntrypoint(EntrypointBase):
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

        # load the (best) trained model back
        loader = ModelLoader(path=context.config_folder, model=context.model)
        if not loader.from_disk:
            logger.warning("Attempted to fit ScoreTransformer on an untrained model.")
            return context

        predicted, target = loader.predict_loop(context.datamodule.valset)

        # fit and save the score transformer
        score = ScoreTransformer(**vars(context.config.score_transformer))
        score.fit(predicted, target)
        logger.info("Fitted ScoreTransformer (mode=%s) successfully.", score.mode)

        score_path = os.path.join(loader.path, "metadata", "score_transformer.pkl")
        with open(score_path, mode="wb") as f:
            pickle.dump(score, f)

        return context
