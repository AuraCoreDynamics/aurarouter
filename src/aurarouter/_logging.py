import logging


def get_logger(name: str = "AuraRouter") -> logging.Logger:
    return logging.getLogger(name)
