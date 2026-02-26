"""
utils/logger.py

Central logging configuration for the FDA Clinical Pharmacology Pipeline.
All modules obtain their named logger from this module via get_module_logger().
No module writes to the root logger directly.
"""

import logging
import sys


LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_logging_configured = False


def configure_root_logging() -> None:
    """
    Configure the root logging handler exactly once.
    Called once at application startup (app.py).
    Writes to both stdout and app.log.
    """
    global _logging_configured
    if _logging_configured:
        return

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log"),
        ],
    )
    _logging_configured = True


def get_module_logger(module_name: str) -> logging.Logger:
    """
    Return a named logger for the given module.
    All pipeline modules must call this instead of logging.getLogger(__name__)
    directly to ensure consistent formatting is applied.

    Parameters
    ----------
    module_name : str
        Fully qualified module name, typically passed as __name__.

    Returns
    -------
    logging.Logger
        Named logger instance using the shared format.
    """
    if not _logging_configured:
        configure_root_logging()

    logger = logging.getLogger(module_name)
    return logger
