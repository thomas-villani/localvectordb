# src/localvectordb_server/_logcfg.py (Updated)
"""
Enhanced logging configuration for LocalVectorDB Server with structured logging,
performance monitoring, and security event tracking.
"""

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

import flask
from flask import g, has_app_context, has_request_context, request


def configure_logging(app: flask.Flask, log_file: Optional[str] = None) -> None:
    """Configure enhanced logging for the application

    Parameters
    ----------
    app : Flask
        Flask app
    log_file : Optional[str], default=None
        Path to log file. If None, only console logging is configured.
    """

    level = app.config.get("LOG_LEVEL", logging.INFO)
    if app.debug:
        level = logging.DEBUG

    # Determine if we should use structured logging
    use_structured = app.config.get("LOG_STRUCTURED", not app.debug)
    log_format = app.config.get("LOG_FORMAT", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": log_format},
            "structured": {"()": StructuredFormatter},
            "security": {"format": "%(asctime)s [SECURITY] [%(levelname)s] %(name)s: %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "structured" if use_structured else "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console"],
                "level": level,
            },
            "localvectordb": {"handlers": ["console"], "level": level, "propagate": False},
            "localvectordb_server": {"handlers": ["console"], "level": level, "propagate": False},
            "localvectordb_server._auth": {
                "handlers": ["console"],
                "level": app.config.get("AUTH_LOG_LEVEL", level),
                "propagate": False,
            },
            "localvectordb.database": {"handlers": ["console"], "level": level, "propagate": False},
            "localvectordb.security": {
                "handlers": ["console"],
                "level": app.config.get("SECURITY_LOG_LEVEL", logging.INFO),
                "propagate": False,
            },
            "localvectordb.errors": {"handlers": ["console"], "level": logging.ERROR, "propagate": False},
            "localvectordb.http": {"handlers": ["console"], "level": logging.INFO, "propagate": False},
            "localvectordb.request": {"handlers": ["console"], "level": level, "propagate": False},
            "flask-limiter": {"handlers": ["console"], "level": logging.INFO, "propagate": False},
            "httpx": {"handlers": ["console"], "level": logging.WARNING, "propagate": False},
            "httpcore": {"handlers": ["console"], "level": logging.WARNING, "propagate": False},
            "asyncio": {"handlers": ["console"], "level": logging.WARNING, "propagate": False},
        },
    }

    # Add file logging if specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "structured",
            "filename": log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        }

        # Separate file for security events
        security_log_file = log_file.replace(".log", "_security.log")
        config["handlers"]["security_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "security",
            "filename": security_log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
        }

        # Separate file for errors
        error_log_file = log_file.replace(".log", "_errors.log")
        config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "structured",
            "filename": error_log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
        }

        # Add file handlers to all loggers
        for logger_config in config["loggers"].values():
            logger_config["handlers"].append("file")

        # Security logger gets its own file
        config["loggers"]["localvectordb.security"]["handlers"].append("security_file")

        # Error logger gets its own file
        config["loggers"]["localvectordb.errors"]["handlers"].append("error_file")

        config["loggers"]["flask-limiter"]["handlers"].append("file")

    # Add performance logging if enabled
    if app.config.get("LOG_PERFORMANCE", False):
        config["loggers"]["localvectordb.performance"] = {
            "handlers": ["console"],
            "level": logging.INFO,
            "propagate": False,
        }

        if log_file:
            perf_log_file = log_file.replace(".log", "_performance.log")
            config["handlers"]["performance_file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "structured",
                "filename": perf_log_file,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            }
            config["loggers"]["localvectordb.performance"]["handlers"].append("performance_file")

    logging.config.dictConfig(config)

    # Log configuration info
    logger = logging.getLogger("localvectordb_server")
    logger.info(f"Logging configured - Level: {logging.getLevelName(level)}, Structured: {use_structured}")
    if log_file:
        logger.info(f"File logging enabled: {log_file}")


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging with consistent fields
    """

    def format(self, record: logging.LogRecord) -> str:

        message = record.getMessage()
        ansi_escape = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        message = ansi_escape.sub("", message)

        # Base log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        # Add request context if available
        if has_app_context():
            if hasattr(g, "request_id"):
                log_entry["request_id"] = g.request_id

            if hasattr(g, "api_key_hash"):
                log_entry["api_key_hash"] = g.api_key_hash

        # Add Flask request context
        if has_request_context():
            try:
                log_entry.update(
                    {
                        "method": request.method,
                        "path": request.path,
                        "remote_addr": request.remote_addr or "",
                        "user_agent": request.headers.get("User-Agent", "")[:100],
                    }
                )
            except Exception:
                # Handle any potential errors when accessing request
                pass

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            exc_info_tuple = record.exc_info
            log_entry["exception"] = {
                "type": exc_info_tuple[0].__name__,
                "message": str(exc_info_tuple[1]),
                "traceback": self.formatException(exc_info_tuple),
            }

        return json.dumps(log_entry, ensure_ascii=False)


class DatabaseLogger:
    """
    Specialized logger for database operations with performance tracking
    """

    def __init__(self, logger_name: str = "localvectordb.database"):
        self.logger = logging.getLogger(logger_name)

    def log_query(self, operation: str, **kwargs):
        """Log database query with context"""
        self.logger.info(
            f"Database operation: {operation}",
            extra={"extra_fields": {"operation_type": "database", "operation": operation, **kwargs}},
        )

    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
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
        """Log database errors with context"""
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
    """
    Specialized logger for security events
    """

    def __init__(self, logger_name: str = "localvectordb.security"):
        self.logger = logging.getLogger(logger_name)

    def log_auth_attempt(self, success: bool, reason: Optional[str] = None, **kwargs):
        """Log authentication attempts"""
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
        """Log rate limiting events"""
        self.logger.warning(
            f"Rate limit {'exceeded' if exceeded else 'approaching'}",
            extra={"extra_fields": {"event_type": "rate_limit", "exceeded": exceeded, **kwargs}},
        )


# Performance monitoring decorator
def log_performance(operation: str, logger: Optional[logging.Logger] = None):
    """
    Decorator to log function performance
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_logger = logger or logging.getLogger(func.__module__)

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

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
                return result

            except Exception as e:
                duration = time.time() - start_time
                func_logger.error(
                    f"Operation {operation} failed: {str(e)}",
                    exc_info=True,
                    extra={
                        "extra_fields": {
                            "operation_type": "performance",
                            "operation": operation,
                            "duration_seconds": duration,
                            "success": False,
                            "error_type": type(e).__name__,
                        }
                    },
                )
                raise

        return wrapper

    return decorator


# Request context manager
@contextmanager
def request_context(operation: str):
    """
    Context manager for tracking operations with request context
    """
    request_id = str(uuid.uuid4())
    g.request_id = request_id

    logger = logging.getLogger("localvectordb.request")
    start_time = time.time()

    logger.info(
        f"Starting operation: {operation}",
        extra={"extra_fields": {"operation_type": "request_start", "operation": operation, "request_id": request_id}},
    )

    try:
        yield request_id
        duration = time.time() - start_time

        logger.info(
            f"Completed operation: {operation}",
            extra={
                "extra_fields": {
                    "operation_type": "request_end",
                    "operation": operation,
                    "request_id": request_id,
                    "duration_seconds": duration,
                    "success": True,
                }
            },
        )

    except Exception as e:
        duration = time.time() - start_time

        logger.error(
            f"Failed operation: {operation}: {str(e)}",
            exc_info=True,
            extra={
                "extra_fields": {
                    "operation_type": "request_end",
                    "operation": operation,
                    "request_id": request_id,
                    "duration_seconds": duration,
                    "success": False,
                    "error_type": type(e).__name__,
                }
            },
        )
        raise


# Middleware for request tracking
def setup_request_logging(app):
    """
    Setup request-level logging middleware
    """

    @app.before_request
    def before_request():
        g.request_id = str(uuid.uuid4())
        g.start_time = time.time()

        logger = logging.getLogger("localvectordb.http")
        logger.info(
            f"HTTP Request: {request.method} {request.path}",
            extra={
                "extra_fields": {
                    "event_type": "http_request_start",
                    "method": request.method,
                    "path": request.path,
                    "query_string": request.query_string.decode("utf-8"),
                    "content_length": request.content_length or 0,
                    "request_id": g.request_id,
                }
            },
        )

    @app.after_request
    def after_request(response):
        duration = time.time() - g.start_time

        logger = logging.getLogger("localvectordb.http")
        logger.info(
            f"HTTP Response: {response.status_code} in {duration:.3f}s",
            extra={
                "extra_fields": {
                    "event_type": "http_request_end",
                    "status_code": response.status_code,
                    "duration_seconds": duration,
                    "content_length": response.content_length or 0,
                    "request_id": g.request_id,
                }
            },
        )

        return response
