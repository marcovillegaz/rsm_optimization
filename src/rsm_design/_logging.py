"""Console logging for the lab-log style output of the examples and the CLI."""

import logging
import sys


def enable_console_logging(level: int = logging.INFO) -> None:
    """Send the package log messages to stdout as plain text.

    The library never prints; it logs. Scripts and the CLI call this function
    so a run reads like a lab log.

    params:
        level (int): logging level, e.g. logging.INFO or logging.DEBUG
    return:
        None
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("rsm_design")
    logger.handlers = [handler]  # Replace handlers to avoid duplicated lines
    logger.setLevel(level)
    logger.propagate = False
