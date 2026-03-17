# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# src/localvectordb_server/_auth.py
"""
Authentication utilities for the LocalVectorDB server using FastAPI dependency injection.
"""

import logging
import time
from hashlib import sha256
from typing import Optional

from fastapi import HTTPException, Request

from localvectordb_server._logcfg import SecurityLogger, api_key_hash_var
from localvectordb_server.keymanager import PermissionLevel

logger = logging.getLogger(__name__)
security_logger = SecurityLogger()


def _validate_database_key(token: str, request: Request) -> tuple[bool, PermissionLevel]:
    """Validate token against database-stored API keys and return permission level."""
    key_manager = getattr(request.app.state, "key_manager", None)
    if not key_manager:
        security_logger.log_auth_attempt(
            success=False, reason="KeyManager not available", token_prefix=_mask_token(token)
        )
        return False, PermissionLevel.READ_WRITE

    try:
        config = getattr(request.app.state, "config", None)
        auto_prune = config and getattr(config.server.security, "auto_prune_expired_keys", False)

        is_valid, permission_level, key_id = key_manager.validate_key_with_permissions(
            token, update_last_used=True, prune_expired=auto_prune
        )

        if is_valid:
            security_logger.log_auth_attempt(
                success=True,
                reason="Database key validated",
                token_prefix=_mask_token(token),
                validation_method="database",
                permission_level=permission_level.value,
                key_id=key_id,
            )
        else:
            security_logger.log_auth_attempt(
                success=False,
                reason="Invalid or expired database key",
                token_prefix=_mask_token(token),
                validation_method="database",
            )

        return is_valid, permission_level or PermissionLevel.READ_WRITE

    except Exception as e:
        logger.error(f"Error validating database key: {e}")
        security_logger.log_auth_attempt(
            success=False,
            reason="Database key validation error",
            token_prefix=_mask_token(token),
            validation_method="database",
            error=str(e),
        )
        return False, PermissionLevel.READ_WRITE


def _mask_token(token: str) -> str:
    """Mask token for logging (show first 8 chars + last 4)."""
    if not token:
        return "empty"
    if len(token) > 12:
        return token[:8] + "..." + token[-4:]
    return token[:4] + "..."


def validate_api_key(token: str, request: Request) -> bool:
    """Validate an API key against database sources."""
    if not token:
        security_logger.log_auth_attempt(success=False, reason="No token provided", token_prefix="empty")  # nosec B106
        return False

    if not token.startswith("lvdb_"):
        security_logger.log_auth_attempt(success=False, reason="Invalid token format", token_prefix=_mask_token(token))
        return False

    is_valid, _ = _validate_database_key(token, request)
    if is_valid:
        return True

    security_logger.log_auth_attempt(
        success=False, reason="Token validation failed for all sources", token_prefix=_mask_token(token)
    )
    return False


def validate_api_key_with_permissions(token: str, request: Request) -> tuple[bool, PermissionLevel]:
    """Validate an API key and return permission level."""
    if not token:
        security_logger.log_auth_attempt(success=False, reason="No token provided", token_prefix="empty")  # nosec B106
        return False, PermissionLevel.READ_WRITE

    if not token.startswith("lvdb_"):
        security_logger.log_auth_attempt(success=False, reason="Invalid token format", token_prefix=_mask_token(token))
        return False, PermissionLevel.READ_WRITE

    is_valid, permission_level = _validate_database_key(token, request)
    if is_valid:
        return True, permission_level

    security_logger.log_auth_attempt(
        success=False, reason="Token validation failed for all sources", token_prefix=_mask_token(token)
    )
    return False, PermissionLevel.READ_WRITE


def _extract_and_validate_token(request: Request, required_permission: PermissionLevel) -> Optional[PermissionLevel]:
    """Extract Bearer token from request and validate permissions.

    Returns the permission level if valid, raises HTTPException otherwise.
    Returns None if auth is not required.
    """
    config = getattr(request.app.state, "config", None)
    auth_required = config and config.server.security.require_api_key
    if not auth_required:
        return None

    api_key_header = "Authorization"
    if config:
        api_key_header = getattr(config.server.security, "api_key_header", "Authorization")

    auth_header = request.headers.get(api_key_header)
    if not auth_header:
        security_logger.log_auth_attempt(success=False, reason="Missing Authorization header")
        raise HTTPException(status_code=401, detail="Authorization header required")

    try:
        auth_type, token = auth_header.split(None, 1)
    except ValueError:
        security_logger.log_auth_attempt(success=False, reason="Invalid Authorization header format")
        raise HTTPException(
            status_code=401, detail="Invalid Authorization header format. Expected: Bearer <token>"
        ) from None

    if auth_type.lower() != "bearer":
        security_logger.log_auth_attempt(success=False, reason=f"Invalid auth type: {auth_type}")
        raise HTTPException(status_code=401, detail="Bearer token required")

    start_time = time.time()
    is_valid, permission_level = validate_api_key_with_permissions(token, request)
    validation_time = time.time() - start_time

    if not is_valid:
        security_logger.log_auth_attempt(
            success=False,
            reason="Invalid Bearer token",
            token_prefix=_mask_token(token),
            validation_time_ms=validation_time * 1000,
        )
        raise HTTPException(status_code=401, detail="Invalid Bearer token")

    if required_permission == PermissionLevel.READ_WRITE and permission_level == PermissionLevel.READ_ONLY:
        security_logger.log_auth_attempt(
            success=False,
            reason=f"Insufficient permissions: requires {required_permission.value}, has {permission_level.value}",
            token_prefix=_mask_token(token),
        )
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions. This endpoint requires {required_permission.value} access.",
        )

    security_logger.log_auth_attempt(
        success=True,
        reason=f"Authentication successful with {permission_level.value} permission",
        token_prefix=_mask_token(token),
        validation_time_ms=validation_time * 1000,
        permission_level=permission_level.value,
    )

    api_key_hash_var.set(sha256(token.encode("utf-8")).hexdigest())
    return permission_level


async def require_read_permission(request: Request) -> None:
    """FastAPI dependency: require read permission (READ_ONLY or READ_WRITE)."""
    _extract_and_validate_token(request, PermissionLevel.READ_ONLY)


async def require_write_permission(request: Request) -> None:
    """FastAPI dependency: require write permission (READ_WRITE only)."""
    _extract_and_validate_token(request, PermissionLevel.READ_WRITE)
