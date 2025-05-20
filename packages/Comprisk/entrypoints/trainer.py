from Dataflux import logging_mode
from Dataflux.logger import get_logger

from Comprisk.entrypoints.base import (
    EntrypointBase,
    EntrypointContext,
)

# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)


class TrainerEntrypoint(EntrypointBase):
    """
    This entrypoints builds on the PrepareEntrypoint and uses the datasets to train a model.

    Lightning stores the used configuration file automatically in the model's output folder. Additionally, the
    entrypoint generates the upload configuration next to it. This file contains information about the latest
    sensor input and pre-processing. When generating the scores in production, this file will be utilized.
    """

    def __init__(self) -> None:
        super().__init__()

        self._checkpoint_path = None
        self._metadata_path = None

    def __call__(
        self, context: EntrypointContext, *args, **kwargs
    ) -> EntrypointContext:
        # start the training
        logger.debug("Starting training (seed=%s)", context.config.seed_everything)
        context.trainer.fit(context.model, context.datamodule)

        logger.info(
            "Finished training model %s on machine %s.",
            context.model.__class__.__name__,
            context.datamodule.trainset.UUID,
        )

        context.config.data.shuffle = False
        return context
