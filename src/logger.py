import logging
import os
import sys
from datetime import datetime
from src import config

_logger = None


def setup_logging(log_file: str = None, level: int = logging.INFO):
    global _logger

    _logger = logging.getLogger("vinbigdata")
    _logger.setLevel(level)
    _logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file) or config.OUTPUT_DIR
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

        _logger.info(f"Logging to: {log_file}")


def get_logger():
    global _logger
    if _logger is None:
        setup_logging()
    return _logger
