"""Application logging. Frame-level events are intentionally not written here."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from utils.paths import LOG_PATH, ensure_dirs


def setup_logging() -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger("dpvs")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"dpvs.{name}")
