"""This module modifies the python logger."""
import logging
import logging.config
import os
import sys


def get_logger(name: str, path: str = "logfile.log", level: int = logging.DEBUG):
    """This function returns a configured logger.

    Args:
        name (str): The name of the logger to create. Should be a package
            name e.g.: `'Dataflux.logger'` or `__name__`.
        path (str): The path and filename of the logfile. Defaults to 'logfile'.
        level (int): The log level to use. Defaults to `logging.DEBUG`.
    """
    # create absolute path
    path = path.replace("./", "")
    path =  os.path.abspath(os.getcwd()) + "/" + path

    # create logdir
    basedir = os.path.dirname(path)
    if basedir != "" and not os.path.exists(basedir):
        os.makedirs(basedir)

    # setup the logger
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)-20.20s] [%(levelname)-5.5s]  %(message)s",
        defaults={
            "name": name.split(".", maxsplit=1)[0][:12]
        }
    )

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # disable all other loggers
    package_dir = os.environ.get("CAS_PROJECT_ROOT", "/app")
    package_path = os.path.join(package_dir, "packages/")
    if os.path.exists(package_path):
        enabled = os.listdir(package_path)
    else:
        enabled = [name]

    logger_list = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for child_logger in logger_list:
        for name in enabled:
            if name not in child_logger.name:
                child_logger.setLevel(logging.CRITICAL)
            else:
                child_logger.setLevel(level)
                break

    return logger
