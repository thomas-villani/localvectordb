"""Retry helpers for transient SQLite "database is locked" errors.

Concurrent writers to a shared-cache in-memory database contend on SQLite's
table-level lock and fail with a non-retryable ``SQLITE_LOCKED`` ("database is
locked" / "database table is locked") rather than the retryable ``SQLITE_BUSY``
that file-backed databases raise (which ``busy_timeout`` handles). These helpers
retry a self-contained, idempotent write with exponential backoff plus jitter so
concurrent operations succeed instead of erroring out.

Only wrap operations that are safe to run more than once (e.g. a single
``INSERT OR REPLACE`` + ``commit``, or a whole transaction with no external side
effects). Do not wrap a block that performs non-transactional side effects
(such as adding vectors to a FAISS index) unless those are separately guarded.
"""

import asyncio
import random
import sqlite3
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

MAX_RETRIES = 10
BASE_DELAY = 0.02  # seconds
MAX_DELAY = 0.5  # seconds (cap per backoff)


def is_sqlite_locked_error(exc: BaseException) -> bool:
    """True for a SQLite "database is locked" / "table is locked" ``OperationalError``."""
    return isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()


def _backoff_delay(attempt: int) -> float:
    delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
    # jitter for retry backoff spacing, not a security/crypto use
    return float(delay + random.uniform(0, delay))  # nosec B311


def retry_on_locked(fn: Callable[[], T]) -> T:
    """Call ``fn`` (a zero-arg callable), retrying on transient SQLite lock errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below unless it's a retryable lock
            if is_sqlite_locked_error(exc) and attempt < MAX_RETRIES - 1:
                time.sleep(_backoff_delay(attempt))
            else:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


async def retry_on_locked_async(fn: Callable[[], Awaitable[T]]) -> T:
    """Await ``fn()`` (a zero-arg coroutine function), retrying on SQLite lock errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below unless it's a retryable lock
            if is_sqlite_locked_error(exc) and attempt < MAX_RETRIES - 1:
                await asyncio.sleep(_backoff_delay(attempt))
            else:
                raise
    raise AssertionError("unreachable")  # pragma: no cover
