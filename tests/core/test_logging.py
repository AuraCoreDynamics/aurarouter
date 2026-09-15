import logging
from aurarouter._logging import get_logger

def test_get_logger_name():
    """Verify get_logger returns a logger with the correct name."""
    logger = get_logger("TestLogger")
    assert logger.name == "TestLogger"

def test_no_double_handlers():
    """
    Verify that calling get_logger multiple times does not add duplicate handlers.
    This tests the fix for the B-1 anti-pattern.
    """
    # Ensure root logger is clean before test
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    root_logger.handlers = []

    try:
        # Configure logging once, as an entry point would
        logging.basicConfig(level=logging.INFO)
        
        # Now get a logger instance
        logger = get_logger("MyTestLogger")

        # Handlers should be configured by basicConfig, not get_logger
        # A simple check is that the number of handlers isn't growing
        # on subsequent calls to get_logger.
        num_handlers = len(logger.handlers)
        
        # Call it again
        logger_2 = get_logger("MyTestLogger2")

        # The number of handlers on the root logger should not have changed
        # and handlers on the specific loggers should also be stable
        # (usually 0, as they delegate to root).
        assert len(root_logger.handlers) == 1
        assert len(logger.handlers) == 0
        assert len(logger_2.handlers) == 0

    finally:
        # Restore original handlers to not affect other tests
        root_logger.handlers = original_handlers

