# src/localvectordb_server/routers/_deps.py
"""Common FastAPI dependencies for router modules."""

from typing import Any

from fastapi import Request

from localvectordb_server._dbmanager import DatabaseManager
from localvectordb_server.config import Config


def get_config(request: Request) -> Config:
    """Get the application Config object."""
    config: Config = request.app.state.config
    return config


def get_db_manager(request: Request) -> DatabaseManager:
    """Get the DatabaseManager instance."""
    db_manager: DatabaseManager = request.app.state.db_manager
    return db_manager


def get_db(db_name: str, request: Request) -> Any:
    """Resolve a LocalVectorDB instance by name from the db_manager.

    Exceptions (e.g. ``APIError(DATABASE_NOT_FOUND)`` raised by the manager) are
    allowed to propagate so the app's registered exception handlers produce the
    standard ``{"error": {...}}`` envelope, consistent with every other route.
    """
    db_manager = request.app.state.db_manager
    return db_manager.get_db(db_name)
