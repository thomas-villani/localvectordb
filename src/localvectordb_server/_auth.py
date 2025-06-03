# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/_auth.py
"""
Authentication utilities for the LocalVectorDB server.

Updated to support both legacy config-based API keys and the new SQLite-based
key management system for improved security and functionality.

Features:
    - Backward compatibility with config-based keys
    - Integration with SQLite-based KeyManager
    - Proper audit logging of authentication attempts
    - Support for key expiration and rotation

Security Improvements:
    - Keys are hashed with bcrypt in the database
    - Automatic expiration checking
    - Usage tracking and audit trails
    - Secure key generation
"""

import logging
import os
from functools import wraps
from typing import Optional

from flask import request, current_app, g
from werkzeug.exceptions import Unauthorized

logger = logging.getLogger(__name__)


def _get_key_manager() -> Optional['KeyManager']:
    """
    Get KeyManager instance, cached in Flask's g object

    Returns
    -------
    KeyManager or None
        KeyManager instance if available, None if initialization fails
    """
    if hasattr(g, 'key_manager'):
        return g.key_manager

    try:
        from localvectordb_server.keymanager import get_key_manager

        # Try to get config path from app
        # config_path = current_app.config.get("API_KEY_DB_PATH")
        key_db_path = (current_app.config_obj.key_database_path
                       or os.path.join(current_app.config_obj.database.root_dir, "api_keys.db"))
        key_manager = get_key_manager(key_db_path)

        # Cache in g for this request
        g.key_manager = key_manager
        return key_manager

    except Exception as e:
        logger.warning(f"Could not initialize KeyManager: {e}")
        g.key_manager = None
        return None



def _validate_database_key(token: str) -> bool:
    """
    Validate token against database-stored API keys

    Parameters
    ----------
    token : str
        The API token to validate

    Returns
    -------
    bool
        True if token is valid in database, False otherwise
    """
    key_manager = _get_key_manager()
    if not key_manager:
        return False

    try:
        is_valid = key_manager.validate_key(token,
                                            update_last_used=True,
                                            prune_expired=current_app.config_obj.api_key_prune_expired)

        if is_valid:
            logger.debug("Token validated against database keys")

        return is_valid

    except Exception as e:
        logger.error(f"Error validating database key: {e}")
        return False


def validate_api_key(token: str) -> bool:
    """
    Validate an API key against both config and database sources

    Parameters
    ----------
    token : str
        The API token to validate

    Returns
    -------
    bool
        True if token is valid from any source, False otherwise
    """
    if not token:
        return False

    # Try database keys first (preferred method)
    if _validate_database_key(token):
        return True

    # Fall back to config keys for backward compatibility
    # if _validate_config_key(token):
    #     logger.warning(
    #         "Authentication using legacy config-based key. "
    #         "Consider migrating to database-managed keys for better security."
    #     )
    #     return True

    logger.debug("Token validation failed for all sources")
    return False


def require_api_key(f):
    """
    Decorator to require Bearer token authentication for routes.

    Supports both legacy config-based keys and new database-managed keys.
    Only checks for Bearer token if REQUIRE_API_KEY is True in config.
    Token must be provided in the Authorization header as 'Bearer <token>'.

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

    Examples
    --------

    Protecting a route::

        @app.route('/api/v1/protected')
        @require_api_key
        def protected_endpoint():
            return {"message": "Access granted"}

    Using with Bearer token::

        curl -H "Authorization: Bearer lvdb_abc123..." http://localhost:5000/api/v1/protected
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        # Check if authentication is required
        if not current_app.config.get("REQUIRE_API_KEY", False):
            # logger.debug("API key authentication disabled")
            return f(*args, **kwargs)

        # log_usage = current_app.config.get("API_KEY_AUDIT_LOGGING", True)
        log_usage = current_app.config_obj.key_audit_logging

        # auth_header_key = current_app.config.get("API_KEY_HEADER", "Authorization")
        auth_header_key = current_app.config_obj.api_key_header or "authorization"

        # Extract Authorization header
        auth_header = request.headers.get(auth_header_key)
        if not auth_header:
            logger.warning("Missing Authorization header")
            raise Unauthorized("Authorization header required")

        # Parse Bearer token
        try:
            auth_type, token = auth_header.split(None, 1)
        except ValueError:
            logger.warning("Invalid Authorization header format")
            raise Unauthorized("Invalid Authorization header format")

        if auth_type.lower() != "bearer":
            logger.warning(f"Invalid auth type: {auth_type}")
            raise Unauthorized("Bearer token required")

        # Validate the token
        if not validate_api_key(token):
            logger.warning("Invalid API key attempt")
            if log_usage:
                audit_key_usage(token, request.endpoint, False)
            raise Unauthorized("Invalid Bearer token")

        logger.debug("API key authentication successful")
        if log_usage:
            audit_key_usage(token, request.endpoint, True)
        g.api_key_hash = hash(token)
        return f(*args, **kwargs)

    return decorated

def audit_key_usage(token: str, endpoint: str, success: bool):
    """
    Audit API key usage for security monitoring

    Parameters
    ----------
    token : str
        The API token that was used (will be partially masked in logs)
    endpoint : str
        The endpoint that was accessed
    success : bool
        Whether the authentication was successful
    """
    # Mask the token for logging (show first 8 chars + last 4)
    if len(token) > 12:
        masked_token = token[:8] + "..." + token[-4:]
    else:
        masked_token = token[:4] + "..."

    status = "SUCCESS" if success else "FAILED"
    client_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', 'Unknown')

    logger.info(
        f"API_AUTH {status}: token={masked_token} endpoint={endpoint} "
        f"ip={client_ip} user_agent={user_agent[:100]}"
    )