"""This module modifies the python logger."""
import logging
import logging.config

def get_logger(name: str, level: int = logging.DEBUG):
    """This function returns a configured logger.

    Args:
        name (str): The name of the logger to create. Should be a package
        name e.g.: `'S3Prep.helper.logger'` or `__name__`.
        level (int): The log level to use. Defaults to `logging.DEBUG`.
    """
    # set up the logger
    name = name.split(".", maxsplit=1)[0]
    logger = logging.getLogger(name)
    logging.basicConfig(filename="s3prep.log", encoding="utf-8", level=logging.DEBUG)

    # disable all other loggers
    logger_list = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for child_logger in logger_list:
        if name not in child_logger.name:
            child_logger.setLevel(logging.CRITICAL)
        else:
            child_logger.setLevel(level)

    return logger
