# src/localvectordb_server/_logcfg.py
"""
Enhanced logging configuration for LocalVectorDB Server with structured logging,
performance monitoring, and security event tracking.

Framework-agnostic: uses contextvars instead of Flask g/request.
"""

import asyncio
import contextvars
import json
import logging
import logging.config
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Context variables (replace Flask g.request_id, g.start_time, etc.)
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
api_key_hash_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("api_key_hash", default=None)
request_start_time_var: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "request_start_time", default=None
)

# Control characters (incl. CR/LF) let user-controlled values forge or split log
# entries. Strip them before interpolating any request-derived value into a log line.
_LOG_UNSAFE_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_log_value(value: Any, max_length: int = 256) -> str:
    """Neutralize a user-controlled value for safe logging (prevents log injection).

    Removes control characters (newlines, carriage returns, etc.) so a crafted
    value cannot inject or split log lines, and truncates overly long values.
    """
    text = _LOG_UNSAFE_CHARS.sub("", str(value))
    if len(text) > max_length:
        text = text[:max_length] + "…"
    return text


class SafePlainFormatter(logging.Formatter):
    """Plain-text formatter that prevents log-injection via embedded newlines.

    Values interpolated through ``%(message)s`` are sanitized at the call site,
    but ``exc_info`` hands the raw exception (and its traceback) straight to the
    formatter, bypassing that sanitizer. A crafted exception message containing a
    newline could otherwise emit a physical line that mimics a genuine record
    (``TIMESTAMP [LEVEL] logger: ...``). We indent every continuation line so it
    cannot start at column 0 like a real record, keeping multi-line tracebacks
    readable while neutralizing forged entries. The structured (JSON) formatter is
    already immune because json.dumps escapes newlines.
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return formatted.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\n  | ")


def configure_logging(config, log_file: Optional[str] = None, debug: bool = False) -> None:
    """Configure enhanced logging for the application.

    Parameters
    ----------
    config : Config
        Application configuration object.
    log_file : Optional[str]
        Path to log file.
    debug : bool
        Whether debug mode is enabled.
    """
    level_name = config.server.log_level if hasattr(config, "server") else "INFO"
    level = getattr(logging, level_name, logging.INFO)
    if debug:
        level = logging.DEBUG

    use_structured = not debug
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    log_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"()": SafePlainFormatter, "format": log_format},
            "structured": {"()": StructuredFormatter},
            "security": {
                "()": SafePlainFormatter,
                "format": "%(asctime)s [SECURITY] [%(levelname)s] %(name)s: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "structured" if use_structured else "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {"handlers": ["console"], "level": level},
            "localvectordb": {"handlers": ["console"], "level": level, "propagate": False},
            "localvectordb_server": {"handlers": ["console"], "level": level, "propagate": False},
            "localvectordb_server._auth": {"handlers": ["console"], "level": level, "propagate": False},
            "localvectordb.database": {"handlers": ["console"], "level": level, "propagate": False},
            "localvectordb.security": {"handlers": ["console"], "level": logging.INFO, "propagate": False},
            "localvectordb.errors": {"handlers": ["console"], "level": logging.ERROR, "propagate": False},
            "localvectordb.http": {"handlers": ["console"], "level": logging.INFO, "propagate": False},
            "localvectordb.request": {"handlers": ["console"], "level": level, "propagate": False},
            "httpx": {"handlers": ["console"], "level": logging.WARNING, "propagate": False},
            "httpcore": {"handlers": ["console"], "level": logging.WARNING, "propagate": False},
            "asyncio": {"handlers": ["console"], "level": logging.WARNING, "propagate": False},
            "uvicorn": {"handlers": ["console"], "level": level, "propagate": False},
        },
    }

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        log_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "structured",
            "filename": log_file,
            "maxBytes": 10485760,
            "backupCount": 5,
        }

        security_log_file = log_file.replace(".log", "_security.log")
        log_config["handlers"]["security_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "security",
            "filename": security_log_file,
            "maxBytes": 10485760,
            "backupCount": 10,
        }

        error_log_file = log_file.replace(".log", "_errors.log")
        log_config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "structured",
            "filename": error_log_file,
            "maxBytes": 10485760,
            "backupCount": 10,
        }

        for logger_config in log_config["loggers"].values():
            logger_config["handlers"].append("file")

        log_config["loggers"]["localvectordb.security"]["handlers"].append("security_file")
        log_config["loggers"]["localvectordb.errors"]["handlers"].append("error_file")

    logging.config.dictConfig(log_config)

    _logger = logging.getLogger("localvectordb_server")
    _logger.info(f"Logging configured - Level: {logging.getLevelName(level)}, Structured: {use_structured}")
    if log_file:
        _logger.info(f"File logging enabled: {log_file}")


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging with consistent fields."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        ansi_escape = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        message = ansi_escape.sub("", message)

        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        # Add context from contextvars
        req_id = request_id_var.get()
        if req_id:
            log_entry["request_id"] = req_id

        key_hash = api_key_hash_var.get()
        if key_hash:
            log_entry["api_key_hash"] = key_hash

        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        if record.exc_info and record.exc_info[0] is not None:
            exc_info_tuple = record.exc_info
            log_entry["exception"] = {
                "type": exc_info_tuple[0].__name__,
                "message": str(exc_info_tuple[1]),
                "traceback": self.formatException(exc_info_tuple),
            }

        return json.dumps(log_entry, ensure_ascii=False)


class DatabaseLogger:
    """Specialized logger for database operations with performance tracking."""

    _loggers: Dict[str, "DatabaseLogger"] = {}

    def __init__(self, logger_name: str = "localvectordb.database"):
        self.logger = logging.getLogger(logger_name)

    @classmethod
    def get_logger(cls, db_name: Optional[str] = None) -> "DatabaseLogger":
        key = db_name or "__default__"
        if key not in cls._loggers:
            cls._loggers[key] = cls()
        return cls._loggers[key]

    def log_query(self, operation: str, **kwargs):
        self.logger.info(
            f"Database operation: {operation}",
            extra={"extra_fields": {"operation_type": "database", "operation": operation, **kwargs}},
        )

    def log_performance(self, operation: str, duration: float, **kwargs):
        self.logger.info(
            f"Performance: {operation} completed in {duration:.3f}s",
            extra={
                "extra_fields": {
                    "operation_type": "performance",
                    "operation": operation,
                    "duration_seconds": duration,
                    **kwargs,
                }
            },
        )

    def log_error(self, operation: str, error: Exception, **kwargs):
        self.logger.error(
            f"Database error in {operation}: {str(error)}",
            exc_info=True,
            extra={
                "extra_fields": {
                    "operation_type": "database_error",
                    "operation": operation,
                    "error_type": type(error).__name__,
                    **kwargs,
                }
            },
        )


class SecurityLogger:
    """Specialized logger for security events."""

    def __init__(self, logger_name: str = "localvectordb.security"):
        self.logger = logging.getLogger(logger_name)

    def log_auth_attempt(self, success: bool, reason: Optional[str] = None, **kwargs):
        level = logging.INFO if success else logging.WARNING
        message = f"Authentication {'successful' if success else 'failed'}"
        if reason:
            message += f": {reason}"
        self.logger.log(
            level,
            message,
            extra={"extra_fields": {"event_type": "authentication", "success": success, "reason": reason, **kwargs}},
        )

    def log_rate_limit(self, exceeded: bool, **kwargs):
        self.logger.warning(
            f"Rate limit {'exceeded' if exceeded else 'approaching'}",
            extra={"extra_fields": {"event_type": "rate_limit", "exceeded": exceeded, **kwargs}},
        )


def log_performance(operation: str, logger: Optional[logging.Logger] = None):
    """Decorator to log function performance.

    Works for both sync and async functions. For coroutine functions the timing
    spans the awaited execution (not just coroutine creation) and success/failure
    reflects the actual outcome.
    """

    def decorator(func):
        func_logger = logger or logging.getLogger(func.__module__)

        def _log_success(duration: float) -> None:
            func_logger.info(
                f"Operation {operation} completed successfully",
                extra={
                    "extra_fields": {
                        "operation_type": "performance",
                        "operation": operation,
                        "duration_seconds": duration,
                        "success": True,
                    }
                },
            )

        def _log_failure(duration: float, exc: Exception) -> None:
            func_logger.error(
                f"Operation {operation} failed: {str(exc)}",
                exc_info=True,
                extra={
                    "extra_fields": {
                        "operation_type": "performance",
                        "operation": operation,
                        "duration_seconds": duration,
                        "success": False,
                        "error_type": type(exc).__name__,
                    }
                },
            )

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    _log_success(time.time() - start_time)
                    return result
                except Exception as e:
                    _log_failure(time.time() - start_time, e)
                    raise

            return async_wrapper

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                _log_success(time.time() - start_time)
                return result
            except Exception as e:
                _log_failure(time.time() - start_time, e)
                raise

        return wrapper

    return decorator


@contextmanager
def request_context(operation: str):
    """Context manager for tracking operations with request context."""
    req_id = request_id_var.get() or str(uuid.uuid4())
    request_id_var.set(req_id)

    _logger = logging.getLogger("localvectordb.request")
    start_time = time.time()

    _logger.info(
        f"Starting operation: {operation}",
        extra={"extra_fields": {"operation_type": "request_start", "operation": operation, "request_id": req_id}},
    )

    try:
        yield req_id
        duration = time.time() - start_time
        _logger.info(
            f"Completed operation: {operation}",
            extra={
                "extra_fields": {
                    "operation_type": "request_end",
                    "operation": operation,
                    "request_id": req_id,
                    "duration_seconds": duration,
                    "success": True,
                }
            },
        )
    except Exception as e:
        duration = time.time() - start_time
        _logger.error(
            f"Failed operation: {operation}: {str(e)}",
            exc_info=True,
            extra={
                "extra_fields": {
                    "operation_type": "request_end",
                    "operation": operation,
                    "request_id": req_id,
                    "duration_seconds": duration,
                    "success": False,
                    "error_type": type(e).__name__,
                }
            },
        )
        raise


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Starlette middleware for HTTP request/response logging (replaces Flask before/after_request)."""

    async def dispatch(self, request: Request, call_next):
        req_id = str(uuid.uuid4())
        request_id_var.set(req_id)
        start_time = time.time()
        request_start_time_var.set(start_time)

        http_logger = logging.getLogger("localvectordb.http")
        http_logger.info(
            f"HTTP Request: {request.method} {request.url.path}",
            extra={
                "extra_fields": {
                    "event_type": "http_request_start",
                    "method": request.method,
                    "path": request.url.path,
                    "query_string": str(request.url.query) if request.url.query else "",
                    "request_id": req_id,
                }
            },
        )

        response = await call_next(request)
        duration = time.time() - start_time

        http_logger.info(
            f"HTTP Response: {response.status_code} in {duration:.3f}s",
            extra={
                "extra_fields": {
                    "event_type": "http_request_end",
                    "status_code": response.status_code,
                    "duration_seconds": duration,
                    "request_id": req_id,
                }
            },
        )

        return response


# Keep backward-compatible function name for code that imports it
def setup_request_logging(app):
    """No-op for backward compatibility. Use RequestLoggingMiddleware instead."""
    pass
