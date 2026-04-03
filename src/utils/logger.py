"""Structured logging for the pipeline — uses Rich for pretty console output."""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

console = Console()

_CONFIGURED = False


def setup_logger(name: str = "stock_forecast", level: str = "INFO") -> logging.Logger:
    global _CONFIGURED

    logger = logging.getLogger(name)

    if not _CONFIGURED:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        # Also log to file
        file_handler = logging.FileHandler("pipeline.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(file_handler)

        _CONFIGURED = True

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Get a child logger for a specific module."""
    parent = setup_logger()
    return parent.getChild(module_name)
