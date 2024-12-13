"""This module modifies the python logger."""
import os
import logging
import logging.config


LOGGING_ENABLED = True
class LoggerDummy:
    """This dummy class represents a basic imitation of the python logger but
    implements methods without any function. THis becomes handy when disabling
    the Dataflux logging.
    """

    def debug(self, *args) -> None:
        """Disables the debug function"""
        return None

    def info(self, *args) -> None:
        """Disables the info function"""
        return None

    def warning(self, *args) -> None:
        """Disables the warning function"""
        return None

    def critical(self, *args) -> None:
        """Disables the critical function"""
        return None

    def error(self, *args) -> None:
        """Disables the error function"""
        return None


def get_logger(name: str, path: str = "logfile.log", level: int = logging.DEBUG):
    """This function returns a configured logger.

    Args:
        name (str): The name of the logger to create. Should be a package
            name e.g.: `'S3Prep.helper.logger'` or `__name__`.
        path (str): The path and filename of the logfile. Defaults to 'logfile'.
        level (int): The log level to use. Defaults to `logging.DEBUG`.
    """
    name = name.split(".", maxsplit=1)[0]
    if name == "Dataflux" and not LOGGING_ENABLED:
        return LoggerDummy()

    # create absolute path
    path = path.replace("./", "")
    path =  os.path.abspath(os.getcwd()) + "/" + path

    # create logdir
    basedir = os.path.dirname(path)
    if basedir != "" and not os.path.exists(basedir):
        os.makedirs(basedir)

    # setup the logger
    logging.basicConfig(filename=path, encoding="utf-8", level=logging.DEBUG)
    logger = logging.getLogger(name)

    # disable all other loggers
    logger_list = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for child_logger in logger_list:
        if name not in child_logger.name:
            child_logger.setLevel(logging.CRITICAL)
        else:
            child_logger.setLevel(level)

    return logger
