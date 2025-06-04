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
Enhanced authentication utilities for the LocalVectorDB server with
structured security logging, comprehensive audit trails, and improved error handling.

Features:
    - Structured security event logging
    - Enhanced audit trails with request context
    - Rate limiting integration
    - Security metrics collection
    - Improved error handling and user feedback
"""

import logging
import os
import time
from functools import wraps

from flask import request, current_app, g
from werkzeug.exceptions import Unauthorized

from localvectordb_server._logcfg import SecurityLogger

logger = logging.getLogger(__name__)
security_logger = SecurityLogger()


def _validate_database_key(token: str) -> bool:
    """
    Validate token against database-stored API keys with enhanced logging

    Parameters
    ----------
    token : str
        The API token to validate

    Returns
    -------
    bool
        True if token is valid in database, False otherwise
    """
    key_manager = current_app.key_manager
    if not key_manager:
        security_logger.log_auth_attempt(
            success=False,
            reason="KeyManager not available",
            token_prefix=_mask_token(token)
        )
        return False

    try:
        # Check if we should auto-prune expired keys
        auto_prune = (hasattr(current_app, 'config_obj') and
                      getattr(current_app.config_obj.server, 'auto_prune_expired_keys', False))

        is_valid = key_manager.validate_key(
            token,
            update_last_used=True,
            prune_expired=auto_prune
        )

        if is_valid:
            security_logger.log_auth_attempt(
                success=True,
                reason="Database key validated",
                token_prefix=_mask_token(token),
                validation_method="database"
            )
            logger.debug("Token validated against database keys")
        else:
            security_logger.log_auth_attempt(
                success=False,
                reason="Invalid or expired database key",
                token_prefix=_mask_token(token),
                validation_method="database"
            )

        return is_valid

    except Exception as e:
        logger.error(f"Error validating database key: {e}")
        security_logger.log_auth_attempt(
            success=False,
            reason="Database key validation error",
            token_prefix=_mask_token(token),
            validation_method="database",
            error=str(e)
        )
        return False


def _mask_token(token: str) -> str:
    """
    Mask token for logging (show first 8 chars + last 4)

    Parameters
    ----------
    token : str
        Token to mask

    Returns
    -------
    str
        Masked token for safe logging
    """
    if not token:
        return "empty"

    if len(token) > 12:
        return token[:8] + "..." + token[-4:]
    else:
        return token[:4] + "..."


def validate_api_key(token: str) -> bool:
    """
    Validate an API key against database sources with comprehensive logging

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
        security_logger.log_auth_attempt(
            success=False,
            reason="No token provided",
            token_prefix="empty"
        )
        return False

    # Validate token format
    if not token.startswith('lvdb_'):
        security_logger.log_auth_attempt(
            success=False,
            reason="Invalid token format",
            token_prefix=_mask_token(token)
        )
        return False

    # Try database keys
    if _validate_database_key(token):
        return True
    # Log final failure
    security_logger.log_auth_attempt(
        success=False,
        reason="Token validation failed for all sources",
        token_prefix=_mask_token(token)
    )
    logger.debug("Token validation failed for all sources")
    return False


def require_api_key(f):
    """
    Enhanced decorator to require Bearer token authentication for routes.

    Features:
    - Comprehensive security logging
    - Rate limiting integration
    - Enhanced error messages
    - Security metrics collection
    - Request context tracking

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
        auth_required = current_app.config.get("REQUIRE_API_KEY", False)
        if not auth_required:
            return f(*args, **kwargs)

        auth_header_key = getattr(current_app.config_obj.server, 'api_key_header', 'Authorization') if hasattr(
            current_app, 'config_obj') else 'Authorization'

        # Extract Authorization header
        auth_header = request.headers.get(auth_header_key)
        if not auth_header:
            security_logger.log_auth_attempt(
                success=False,
                reason="Missing Authorization header",
                endpoint=request.endpoint
            )
            raise Unauthorized("Authorization header required")

        # Parse Bearer token
        try:
            auth_type, token = auth_header.split(None, 1)
        except ValueError:
            security_logger.log_auth_attempt(
                success=False,
                reason="Invalid Authorization header format",
                endpoint=request.endpoint,
                auth_header_preview=auth_header[:20] + "..." if len(auth_header) > 20 else auth_header
            )
            raise Unauthorized("Invalid Authorization header format. Expected: Bearer <token>")

        if auth_type.lower() != "bearer":
            security_logger.log_auth_attempt(
                success=False,
                reason=f"Invalid auth type: {auth_type}",
                endpoint=request.endpoint,
                auth_type=auth_type
            )
            raise Unauthorized("Bearer token required")

        # Validate the token
        start_time = time.time()
        is_valid = validate_api_key(token)
        validation_time = time.time() - start_time

        if not is_valid:
            security_logger.log_auth_attempt(
                success=False,
                reason="Invalid Bearer token",
                endpoint=request.endpoint,
                token_prefix=_mask_token(token),
                validation_time_ms=validation_time * 1000
            )
            raise Unauthorized("Invalid Bearer token")

        # Success logging
        security_logger.log_auth_attempt(
            success=True,
            reason="Authentication successful",
            endpoint=request.endpoint,
            token_prefix=_mask_token(token),
            validation_time_ms=validation_time * 1000
        )

        logger.debug("API key authentication successful")

        # Store token hash for request tracking
        g.api_key_hash = hash(token)
        g.authenticated = True

        print("auth success")
        return f(*args, **kwargs)

    return decorated
