#  Copyright (c) 2023-2025 Tom Villani, Ph.D. All rights reserved.
#

"""
localvectordb_server/_auth.py
Authentication utilities for the LocalVectorDB server.
"""
from functools import wraps

from flask import request, current_app
from werkzeug.exceptions import Unauthorized


def require_api_key(f):
    """
    Decorator to require Bearer token authentication for routes.

    Only checks for Bearer token if REQUIRE_API_KEY is True in config.
    Token must be provided in the Authorization header as 'Bearer <token>'
    and must match one of the keys in the AUTHORIZED_API_KEYS config list.

    Parameters
    ----------
    f : callable
        The route function to protect

    Returns
    -------
    callable
        Wrapped function that checks for Bearer token before executing

    Raises
    ------
    Unauthorized
        If authentication token is required but missing or invalid
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_app.config.get("REQUIRE_API_KEY", False):
            return f(*args, **kwargs)

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise Unauthorized("Authorization header required")

        try:
            auth_type, token = auth_header.split(None, 1)
        except ValueError:
            raise Unauthorized("Invalid Authorization header format")

        if auth_type.lower() != "bearer":
            raise Unauthorized("Bearer token required")

        valid_keys = current_app.config.get("AUTHORIZED_API_KEYS", [])
        if not valid_keys:
            # If API keys are required but none configured, deny all access
            raise Unauthorized("API authentication misconfigured")

        if token not in valid_keys:
            raise Unauthorized("Invalid Bearer token")

        return f(*args, **kwargs)

    return decorated

