#  Copyright (c) 2023-2025 Tom Villani, Ph.D. All rights reserved.
"""
localvectordb_server/__init__.py

FastAPI-based server for interacting with `localvectordb.LocalVectorDB` via HTTP
with structured logging, error handling, and performance monitoring.
"""

from localvectordb_server.app import create_app

__all__ = ["create_app"]
