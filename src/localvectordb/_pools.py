import asyncio
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

import aiosqlite
from aiosqlite import Connection

from localvectordb.exceptions import ConnectionPoolError


class PooledConnection:
    """Wrapper for a pooled connection that handles automatic return to pool"""

    def __init__(self, connection: sqlite3.Connection, pool: "ConnectionPool") -> None:
        self.connection = connection
        self.pool = pool
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying connection"""
        return getattr(self.connection, name)

    def __enter__(self) -> sqlite3.Connection:
        self._closed = False
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Return connection to pool on exit"""
        if not self._closed:
            self.pool.return_connection(self.connection)
            self._closed = True

    def close(self) -> None:
        """Manually return connection to pool"""
        if not self._closed:
            self.pool.return_connection(self.connection)
            self._closed = True


class ConnectionPool:
    """Thread-safe connection pool for SQLite with proper context manager support"""

    def __init__(self, db_path: Union[str, Path], max_connections: int = 10, pragmas: Optional[Dict[str, Any]] = None):
        self.db_path = Path(db_path)
        self.max_connections = max_connections
        self._pool: List[sqlite3.Connection] = []
        self._lock = threading.RLock()
        self._created_connections = 0
        self._pragmas = pragmas or {}

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with proper settings"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)

        # For in-memory shared cache databases, foreign keys can cause constraint issues
        # across connections due to transaction isolation
        if "mode=memory&cache=shared" in str(self.db_path):
            conn.execute("PRAGMA foreign_keys = OFF")
        else:
            conn.execute("PRAGMA foreign_keys = ON")

        conn.row_factory = sqlite3.Row

        # Apply tuning pragmas if provided
        if self._pragmas:
            try:
                from localvectordb.sqlite_tuning import apply_pragmas

                apply_pragmas(conn, self._pragmas)
            except Exception:
                pass  # Best-effort pragma application

        self._created_connections += 1
        return conn

    @property
    def closed(self) -> bool:
        return self._created_connections == 0

    def get_connection(self) -> PooledConnection:
        """Get a connection from the pool (wrapped for automatic return)"""
        with self._lock:
            if self._pool:
                # Reuse existing connection from pool
                conn = self._pool.pop()
                # Verify connection is still valid
                try:
                    if conn.in_transaction:
                        conn.rollback()
                    conn.execute("SELECT 1")
                    return PooledConnection(conn, self)
                except sqlite3.Error:
                    # Connection is invalid, create a new one
                    conn.close()
                    self._created_connections -= 1

            # Create new connection if pool is empty or connection was invalid
            if self._created_connections < self.max_connections:
                conn = self._create_connection()
                return PooledConnection(conn, self)
            else:
                raise ConnectionPoolError("No connections available!")

    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool"""
        with self._lock:
            if len(self._pool) < self.max_connections:
                # Check if connection is still valid before returning to pool
                try:
                    if conn.in_transaction:
                        conn.rollback()
                    conn.execute("SELECT 1")
                    self._pool.append(conn)
                except sqlite3.Error:
                    # Connection is invalid, close it
                    conn.close()
                    self._created_connections -= 1
            else:
                # Pool is full, close the connection
                conn.close()
                self._created_connections -= 1

    @contextmanager
    def get_connection_context(self) -> Generator[sqlite3.Connection, None, None]:
        """Alternative context manager interface"""
        pooled_conn = self.get_connection()
        try:
            yield pooled_conn.connection
        finally:
            pooled_conn.close()

    def close_all(self) -> None:
        """Close all connections in the pool"""
        with self._lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()
            self._created_connections = 0

    @property
    def stats(self) -> dict:
        """Get pool statistics for debugging"""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "max_connections": self.max_connections,
                "created_connections": self._created_connections,
                "available_connections": len(self._pool),
            }

    def __del__(self) -> None:
        """Cleanup on garbage collection"""
        try:
            self.close_all()
        except Exception:
            pass  # Ignore errors during cleanup


class AsyncConnectionPool:
    """
    Async connection pool for SQLite using aiosqlite

    This provides async database operations while maintaining the same interface
    patterns as the sync ConnectionPool. Uses condition variables for proper
    waiting when pool is exhausted instead of immediately failing.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        max_connections: int = 10,
        wait_timeout: float = 30.0,
        pragmas: Optional[Dict[str, Any]] = None,
    ):
        self.db_path = Path(db_path)
        self.max_connections = max_connections
        self.wait_timeout = wait_timeout
        self._pool: List[aiosqlite.Connection] = []
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._created_connections = 0
        self._pragmas = pragmas or {}

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new async SQLite connection with proper settings"""
        # Re-enable type detection with proper converters registered
        conn = await aiosqlite.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

        # For in-memory shared cache databases, foreign keys can cause constraint issues
        # across connections due to transaction isolation
        if "mode=memory&cache=shared" in str(self.db_path):
            await conn.execute("PRAGMA foreign_keys = OFF")
        else:
            await conn.execute("PRAGMA foreign_keys = ON")

        # Enable row factory for dict-like access
        conn.row_factory = aiosqlite.Row

        # Apply tuning pragmas if provided
        if self._pragmas:
            try:
                from localvectordb.sqlite_tuning import apply_pragmas_async

                await apply_pragmas_async(conn, self._pragmas)
            except Exception:
                pass  # Best-effort pragma application

        self._created_connections += 1
        return conn

    @property
    def closed(self) -> bool:
        return self._created_connections == 0

    async def get_connection(self) -> aiosqlite.Connection:
        """
        Get a connection from the pool, waiting if necessary.

        Returns
        -------
        aiosqlite.Connection
            A database connection

        Raises
        ------
        asyncio.TimeoutError
            If no connection becomes available within wait_timeout
        """
        async with self._condition:
            # Wait until a connection is available
            await asyncio.wait_for(self._wait_for_connection(), timeout=self.wait_timeout)

            if self._pool:
                # Reuse existing connection from pool
                conn = self._pool.pop()
                # Verify connection is still valid
                try:
                    await conn.execute("SELECT 1")
                    return conn
                except Exception:
                    # Connection is invalid, close it and create new one
                    await conn.close()
                    self._created_connections -= 1

            # Create new connection if pool is empty or connection was invalid
            if self._created_connections < self.max_connections:
                return await self._create_connection()
            else:
                # This should not happen after waiting, but handle gracefully
                raise RuntimeError("Async connection pool exhausted after waiting")

    async def _wait_for_connection(self) -> None:
        """Wait until a connection is available (either pooled or can be created)."""
        while not self._pool and self._created_connections >= self.max_connections:
            await self._condition.wait()

    async def return_connection(self, conn: aiosqlite.Connection) -> None:
        """Return a connection to the pool and notify waiting tasks"""
        async with self._condition:
            if len(self._pool) < self.max_connections:
                # Check if connection is still valid before returning to pool
                try:
                    await conn.execute("SELECT 1")
                    self._pool.append(conn)
                    # Notify one waiting task that a connection is available
                    self._condition.notify()
                except Exception:
                    # Connection is invalid, close it
                    await conn.close()
                    self._created_connections -= 1
                    # Still notify in case we can create a new connection
                    self._condition.notify()
            else:
                # Pool is full, close the connection
                await conn.close()
                self._created_connections -= 1
                # Notify in case we can create a new connection now
                self._condition.notify()

    @asynccontextmanager
    async def get_connection_context(self) -> AsyncGenerator[Connection, None]:
        """Async context manager for getting/returning connections"""
        conn = await self.get_connection()
        try:
            yield conn
        finally:
            await self.return_connection(conn)

    async def close_all(self) -> None:
        """Close all connections in the pool and notify all waiting tasks"""
        async with self._condition:
            for conn in self._pool:
                await conn.close()
            self._pool.clear()
            self._created_connections = 0
            # Notify all waiting tasks that pool is closed
            self._condition.notify_all()

    @property
    def stats(self) -> dict:
        """Get pool statistics for debugging"""
        return {
            "pool_size": len(self._pool),
            "max_connections": self.max_connections,
            "created_connections": self._created_connections,
            "available_connections": len(self._pool),
        }


class ReadWriteLock:
    """
    Optimized ReadWriteLock that supports re-entrant write locks from the same thread

    This implementation fixes the thundering herd problem by using separate condition
    variables for readers and writers, reducing unnecessary wake-ups and context switches
    in high-contention scenarios.

    Features:
    - Re-entrant write locks (same thread can acquire multiple write locks)
    - Writer preference (thread holding write lock can also acquire read locks)
    - Separate condition variables to minimize spurious wake-ups
    - Efficient notification targeting only threads that can proceed
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()  # Main coordination lock
        self._readers = 0
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_count = 0  # Track nested write locks from same thread

        # Separate condition variables for better wake-up targeting
        self._read_ready = threading.Condition(self._lock)  # Readers wait here
        self._write_ready = threading.Condition(self._lock)  # Writers wait here

    @contextmanager
    def read_lock(self) -> Generator[None, None, None]:
        """Acquire read lock (multiple readers allowed, unless writer active)"""
        current_thread = threading.current_thread()

        with self._lock:
            # If current thread already holds write lock, allow read (writer preference)
            if self._writer_thread == current_thread:
                yield
                return

            # Wait for any writer to finish
            while self._writer_thread is not None:
                self._read_ready.wait()

            self._readers += 1

        try:
            yield
        finally:
            with self._lock:
                self._readers -= 1
                # If last reader finished, notify waiting writers
                if self._readers == 0:
                    self._write_ready.notify()

    @contextmanager
    def write_lock(self) -> Generator[None, None, None]:
        """Acquire write lock (exclusive, but re-entrant for same thread)"""
        current_thread = threading.current_thread()

        with self._lock:
            # If same thread already holds write lock, just increment counter (re-entrant)
            if self._writer_thread == current_thread:
                self._writer_count += 1
                try:
                    yield
                finally:
                    self._writer_count -= 1
                return

            # Wait for readers to finish AND for any other writer to finish
            while self._readers > 0 or self._writer_thread is not None:
                self._write_ready.wait()

            # Acquire write lock
            self._writer_thread = current_thread
            self._writer_count = 1

        try:
            yield
        finally:
            with self._lock:
                self._writer_count -= 1
                if self._writer_count == 0:
                    # Release write lock
                    self._writer_thread = None

                    self._write_ready.notify()
                    self._read_ready.notify_all()
