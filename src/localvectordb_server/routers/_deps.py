# src/localvectordb_server/routers/_deps.py
"""Common FastAPI dependencies for router modules."""

from typing import Any

from fastapi import HTTPException, Request

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
    """Resolve a LocalVectorDB instance by name from the db_manager."""
    db_manager = request.app.state.db_manager
    try:
        return db_manager.get_db(db_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Database '{db_name}' not found: {e}") from e
