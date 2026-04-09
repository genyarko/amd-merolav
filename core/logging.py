"""Centralized logging configuration for the CUDA-to-ROCm migration tool."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
) -> None:
    """Configure the root 'rocm_migrate' logger.

    Args:
        level: Log level name — DEBUG, INFO, WARNING, ERROR.
        log_file: Optional path to a log file. If provided, logs are written
                  to both stderr and the file.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger("rocm_migrate")
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers on repeated calls
    root_logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Always log to stderr (Rich console uses stdout, so logs stay separate)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(numeric_level)
    stderr_handler.setFormatter(fmt)
    root_logger.addHandler(stderr_handler)

    # Optional rotating file handler
    if log_file:
        from logging.handlers import RotatingFileHandler

        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            str(path), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'rocm_migrate' namespace.

    Usage::

        from core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Starting analysis")
    """
    return logging.getLogger(f"rocm_migrate.{name}")
