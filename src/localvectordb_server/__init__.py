# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
localvectordb_server/__init__.py

FastAPI-based server for interacting with `localvectordb.LocalVectorDB` via HTTP
with structured logging, error handling, and performance monitoring.
"""

from localvectordb_server.app import create_app

__all__ = ["create_app"]
