import logging


def get_logger(name: str = "AuraRouter") -> logging.Logger:
    prefix = "AuraRouter"
    if not name or name == prefix:
        return logging.getLogger(prefix)
    if name.startswith(prefix + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{prefix}.{name}")
