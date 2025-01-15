from Dataflux import DEBUG_MODE, logging_mode
from Dataflux.logger import get_logger


# switch logging mode depending on DEBUG_MODE
logger = get_logger(__name__, path="./logfile", level=logging_mode)
logger.debug("Running in debug mode.")

__all__ = ["DEBUG_MODE", "logger"]

