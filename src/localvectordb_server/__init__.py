"""
localvectordb_server/__init__.py

FastAPI-based server for interacting with `localvectordb.LocalVectorDB` via HTTP
with structured logging, error handling, and performance monitoring.

The heavy server dependencies (fastapi, uvicorn, ...) belong to the optional
``server`` extra, so ``create_app`` is exposed lazily (PEP 562): importing this
package must stay possible on a base install, and the ImportError users hit
should tell them which extra to install.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from localvectordb_server.app import create_app

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        try:
            from localvectordb_server.app import create_app
        except ImportError as exc:
            raise ImportError(
                "The localvectordb server requires the 'server' extra "
                f"(missing dependency: {getattr(exc, 'name', None) or exc}). "
                'Install it with:  pip install "localvectordb[server]"'
            ) from exc
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
