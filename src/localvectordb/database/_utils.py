"""
Shared database utilities for sync/async operations.

This module provides common abstractions used across the database layer to eliminate
code duplication between synchronous and asynchronous database operations.
"""

from __future__ import annotations


def glob_escape(text: str) -> str:
    """Escape SQLite GLOB metacharacters (``*``, ``?``, ``[``) in a literal string.

    GLOB has no ESCAPE clause, so each metacharacter is neutralized by wrapping it
    in a single-character class (``*`` -> ``[*]``). Used to turn a literal id
    prefix into a safe, case-sensitive ``prefix*`` GLOB pattern.
    """
    return "".join(f"[{ch}]" if ch in "*?[" else ch for ch in text)


class DatabaseExecutor:
    """
    Base abstraction for database operations that differ between sync and async.

    This class provides a common interface for database operations, allowing the same
    business logic to be used for both synchronous and asynchronous database connections.
    """

    def execute(self, conn, sql: str, params=None):
        """Execute a SQL statement."""
        raise NotImplementedError

    def fetchall(self, cursor):
        """Fetch all results from a cursor."""
        raise NotImplementedError

    def fetchone(self, cursor):
        """Fetch one result from a cursor."""
        raise NotImplementedError


class SyncDatabaseExecutor(DatabaseExecutor):
    """
    Synchronous database executor for standard sqlite3 connections.

    Provides synchronous implementations of database operations for use with
    sqlite3.Connection objects.
    """

    def execute(self, conn, sql: str, params=None):
        """Execute a SQL statement synchronously."""
        if params is None:
            return conn.execute(sql)
        return conn.execute(sql, params)

    def fetchall(self, cursor):
        """Fetch all results synchronously."""
        return cursor.fetchall()

    def fetchone(self, cursor):
        """Fetch one result synchronously."""
        return cursor.fetchone()


class AsyncDatabaseExecutor(DatabaseExecutor):
    """
    Asynchronous database executor for aiosqlite connections.

    Provides asynchronous implementations of database operations for use with
    aiosqlite.Connection objects.
    """

    async def execute(self, conn, sql: str, params=None):
        """Execute a SQL statement asynchronously."""
        if params is None:
            return await conn.execute(sql)
        return await conn.execute(sql, params)

    async def fetchall(self, cursor):
        """Fetch all results asynchronously."""
        return await cursor.fetchall()

    async def fetchone(self, cursor):
        """Fetch one result asynchronously."""
        return await cursor.fetchone()
