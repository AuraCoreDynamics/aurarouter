import threading
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class EventReporter:
    """Bounded thread pool for fire-and-forget background events.

    max_workers controls concurrency; max_queue_depth caps total in-flight
    + pending work. Events beyond the cap are dropped with a warning rather
    than allowed to grow the queue without bound.
    """

    def __init__(self, max_workers: int = 8, max_queue_depth: int = 256):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aurarouter-event")
        self._semaphore = threading.Semaphore(max_queue_depth)

    def submit(self, fn, *args, **kwargs) -> None:
        """Submit a callable. Drops and logs if queue is full. Never raises."""
        if not self._semaphore.acquire(blocking=False):
            logger.warning("EventReporter queue full (depth exhausted); dropping event %s", getattr(fn, '__name__', fn))
            return
        try:
            future = self._executor.submit(fn, *args, **kwargs)
        except RuntimeError:
            # executor already shut down
            self._semaphore.release()
            return
        future.add_done_callback(self._on_done)

    def _on_done(self, future):
        self._semaphore.release()
        exc = future.exception()
        if exc:
            logger.warning("EventReporter task failed: %s", exc)

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait)
