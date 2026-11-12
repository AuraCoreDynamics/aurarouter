"""Logging configuration for the BitNet backend plugin."""

import logging

LOGGER_NAME = "AuraRouter.Backend.BitNet"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger scoped under AuraRouter.Backend.BitNet."""
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


logger = get_logger()
