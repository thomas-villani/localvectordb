"""
localvectordb_server/app.py

FastAPI application factory for LocalVectorDB server with structured logging,
error handling, CORS, rate limiting, and security headers.
"""

import ipaddress
import logging
import os
from contextlib import asynccontextmanager
from typing import Union

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from localvectordb.exceptions import BaseLocalVectorDBException, ConfigurationError
from localvectordb_server._dbmanager import DatabaseManager
from localvectordb_server._error_handlers import (
    APIError,
    ValidationError,
    standardize_error_response,
)
from localvectordb_server._logcfg import RequestLoggingMiddleware, configure_logging
from localvectordb_server.config import Config, load_config
from localvectordb_server.keymanager import KeyManager
from localvectordb_server.utils.hostmatch import validate_host_against_patterns

try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    _SLOWAPI_AVAILABLE = True
except ImportError:
    _SLOWAPI_AVAILABLE = False


logger = logging.getLogger("localvectordb_server")

# Maps HTTP status codes raised via HTTPException (e.g. by auth) to stable error
# codes so 4xx/5xx responses share the standard {"error": {...}} envelope.
_HTTP_STATUS_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMIT_EXCEEDED",
}


def _error_response(status_code: int, message: str, error_code: str) -> JSONResponse:
    """Build a JSONResponse using the standard ``{"error": {...}}`` envelope.

    Used by middleware (which runs outside the exception-handler stack) so that
    host/proxy rejections match the envelope every other failure uses.
    """
    api_error = APIError(message=message, error_code=error_code, status_code=status_code, recoverable=status_code < 500)
    return JSONResponse(status_code=status_code, content=api_error.to_dict())


def _rate_limit_envelope_handler(request: "Request", exc: Exception) -> JSONResponse:
    """Rate-limit (429) handler that emits the standard ``{"error": {...}}`` envelope.

    Replaces slowapi's stock handler, whose ``{"error": "<string>"}`` body breaks
    the object envelope every other error response uses. Rate-limit headers
    (Retry-After, X-RateLimit-*) are still injected via the limiter.
    """
    api_error = APIError(
        message=f"Rate limit exceeded: {exc.detail}",  # type: ignore[attr-defined]
        error_code="RATE_LIMIT_EXCEEDED",
        status_code=429,
        recoverable=True,
    )
    response = JSONResponse(status_code=429, content=api_error.to_dict())
    limiter = getattr(request.app.state, "limiter", None)
    view_rate_limit = getattr(request.state, "view_rate_limit", None)
    if limiter is not None and view_rate_limit is not None:
        # slowapi attaches the current window's headers (Retry-After, etc.).
        response = limiter._inject_headers(response, view_rate_limit)
    return response


def register_exception_handlers(app: FastAPI, *, debug: bool = False) -> None:
    """Register the standard exception handlers on ``app``.

    Shared by ``create_app`` and the integration test fixtures so the error
    envelope stays identical everywhere. Every handler emits the standard
    ``{"error": {...}}`` envelope, including auth/HTTP errors (which otherwise
    use Starlette's ``{"detail": ...}``) and Pydantic 422 validation errors.
    """

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(ValidationError)
    async def handle_validation_error(request: Request, exc: ValidationError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(BaseLocalVectorDBException)
    async def handle_domain_error(request: Request, exc: BaseLocalVectorDBException):
        # Map known library exceptions (e.g. DatabaseNotFoundError -> 404) to the
        # standard envelope. Registering an explicit handler keeps these as proper
        # client errors rather than falling through to the 500 catch-all.
        error_response, status_code = standardize_error_response(exc, debug=debug)
        return JSONResponse(status_code=status_code, content=error_response)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException):
        # Auth (and any HTTPException) goes through the standard envelope instead
        # of Starlette's default {"detail": ...}, so clients parse one error shape.
        error_code = _HTTP_STATUS_ERROR_CODES.get(exc.status_code, "HTTP_ERROR")
        api_error = APIError(
            message=str(exc.detail),
            error_code=error_code,
            status_code=exc.status_code,
            recoverable=exc.status_code < 500,
        )
        return JSONResponse(
            status_code=exc.status_code, content=api_error.to_dict(), headers=getattr(exc, "headers", None)
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(request: Request, exc: RequestValidationError):
        # Pydantic body/query validation uses the same 400 VALIDATION_ERROR shape as
        # the hand-rolled ValidationError, so all validation failures look identical
        # to clients (the SDK maps HTTP 400 -> ValueError).
        api_error = ValidationError("Request validation failed", details={"errors": jsonable_encoder(exc.errors())})
        return JSONResponse(status_code=api_error.status_code, content=api_error.to_dict())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        error_response, status_code = standardize_error_response(exc, debug=debug)
        return JSONResponse(status_code=status_code, content=error_response)


def _is_trusted_proxy(remote_addr: str, trusted_proxies: list) -> bool:
    if not trusted_proxies:
        return False
    try:
        remote_ip = ipaddress.ip_address(remote_addr)
        for proxy in trusted_proxies:
            try:
                proxy_network = ipaddress.ip_network(proxy, strict=False)
                if remote_ip in proxy_network:
                    return True
            except ValueError:
                continue
        return False
    except ValueError:
        return False


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds security headers to all responses (replaces flask-talisman)."""

    def __init__(self, app, config):
        super().__init__(app)
        self.security = config.server.security

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        sec = self.security

        if sec.content_type_nosniff:
            response.headers["X-Content-Type-Options"] = "nosniff"
        if sec.x_frame_options:
            response.headers["X-Frame-Options"] = sec.x_frame_options
        if sec.x_xss_protection:
            response.headers["X-XSS-Protection"] = str(sec.x_xss_protection)
        if sec.referrer_policy:
            response.headers["Referrer-Policy"] = sec.referrer_policy
        if sec.content_security_policy:
            csp = sec.content_security_policy
            if isinstance(csp, dict):
                csp_str = "; ".join(f"{k} {v}" for k, v in csp.items())
            else:
                csp_str = str(csp)
            response.headers["Content-Security-Policy"] = csp_str
        if sec.strict_transport_security and sec.force_https:
            max_age = getattr(sec, "strict_transport_security_max_age", 31536000)
            response.headers["Strict-Transport-Security"] = f"max-age={max_age}; includeSubDomains"

        return response


class HostValidationMiddleware(BaseHTTPMiddleware):
    """Validates Host header against trusted hosts (replaces @app.before_request)."""

    def __init__(self, app, config):
        super().__init__(app)
        self.config = config
        self.trusted_patterns = config.server.security.trusted_hosts

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "")

        if self.config.server.proxy_enabled:
            remote_addr = request.client.host if request.client else None
            if remote_addr and not _is_trusted_proxy(remote_addr, self.config.server.trusted_proxies):
                logger.warning(f"Rejected request from untrusted proxy: {remote_addr}")
                return _error_response(403, "Request from untrusted proxy", "UNTRUSTED_PROXY")

            forwarded_host = request.headers.get("x-forwarded-host")
            if forwarded_host and not validate_host_against_patterns(forwarded_host, self.trusted_patterns):
                logger.warning(f"Rejected request with untrusted X-Forwarded-Host: {forwarded_host}")
                return _error_response(400, f"Invalid X-Forwarded-Host header: {forwarded_host}", "INVALID_HOST")

            if not validate_host_against_patterns(host, self.trusted_patterns):
                logger.warning(f"Rejected request with untrusted effective Host header: {host}")
                return _error_response(400, f"Invalid Host header: {host}", "INVALID_HOST")
        else:
            if not validate_host_against_patterns(host, self.trusted_patterns):
                logger.warning(f"Rejected request with untrusted Host header: {host}")
                return _error_response(400, f"Invalid Host header: {host}", "INVALID_HOST")

        return await call_next(request)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects request bodies larger than ``server.max_request_size``.

    ``max_request_size`` was configured and validated but never wired into the
    stack, so ``upload.py``'s ``await file.read()`` could buffer an arbitrarily
    large upload into memory (a memory-exhaustion DoS). This restores the
    Flask ``MAX_CONTENT_LENGTH`` parity the config already advertises by
    rejecting on the declared ``Content-Length`` with a 413.
    """

    def __init__(self, app, config):
        super().__init__(app)
        self.max_request_size = config.server.max_request_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                return _error_response(400, "Invalid Content-Length header", "INVALID_CONTENT_LENGTH")
            if declared > self.max_request_size:
                return _error_response(
                    413,
                    f"Request body too large: {declared} bytes exceeds the " f"{self.max_request_size}-byte limit.",
                    "REQUEST_TOO_LARGE",
                )
        return await call_next(request)


# Loopback hosts that keep the server unreachable from other machines. Binding
# to anything else exposes it on the network, where "no auth" means anyone who
# can reach the port has full read/write access.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"})


def _warn_if_insecure_bind(config, logger) -> None:
    """Emit loud startup warnings when the server is exposed without protection.

    The dev defaults (no API key, CORS ``*``, no Host validation) are convenient
    on ``127.0.0.1`` but dangerous on a routable interface. We do not change the
    defaults (that would break existing docker/compose deployments), but a
    non-loopback bind with authentication disabled gets a prominent warning so an
    operator cannot expose an open read/write endpoint without being told.
    """
    host = (getattr(config.server, "host", "") or "").strip()
    security = config.server.security
    if host in _LOOPBACK_HOSTS:
        return
    if security.require_api_key:
        return  # Exposed, but authenticated -- the operator opted in knowingly.

    logger.warning("=" * 72)
    # nosec B104 - "0.0.0.0" here is a display fallback in a warning message, not a bind address.
    logger.warning("SECURITY: server is bound to %r with API authentication DISABLED.", host or "0.0.0.0")  # nosec B104
    logger.warning("Any host that can reach this port has full read/write access to every database.")
    if security.cors_allowed_origins == "*":
        logger.warning("CORS is also open to all origins ('*').")
    if not security.trusted_hosts:
        logger.warning("Host header validation is off (security.trusted_hosts is unset).")
    logger.warning("Before exposing this server, set security.require_api_key = true and trusted_hosts.")
    logger.warning("See the 'Production deployment' section of the docs for the hardening checklist.")
    logger.warning("=" * 72)


def create_app(
    configuration: Union[str, Config, None] = None,
    database_directory=None,
    debug=False,
    log_level=None,
    **kwargs,
) -> FastAPI:
    """FastAPI application factory with structured logging and error handling."""

    _config = load_config(configuration)

    if database_directory:
        _config.database.root_dir = database_directory
    if debug:
        _config.server.log_level = "DEBUG"
    if log_level and log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        _config.server.log_level = log_level

    for key, value in kwargs.items():
        if key == "host" and value:
            _config.server.host = value
        elif key == "port" and value:
            _config.server.port = value

    # Configure logging
    log_file = None
    if not debug and _config.server.environment != "development":
        log_dir = os.path.join(_config.database.root_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "localvectordb.log")

    configure_logging(_config, log_file, debug=debug)

    # Build lifespan
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        logger.info("Starting LocalVectorDB Server initialization")

        if not os.path.exists(_config.database.root_dir):
            os.makedirs(_config.database.root_dir)
            logger.info(f"Created database directory: {_config.database.root_dir}")

        logger.info(f"Database Root Path: {os.path.abspath(_config.database.root_dir)}")

        try:
            app.state.db_manager = DatabaseManager(_config)
            logger.info("Database manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}", exc_info=True)
            raise ConfigurationError(f"Database manager initialization failed: {e}") from e

        key_db_path = _config.server.security.key_database_path or os.path.join(
            _config.database.root_dir, "api_keys.db"
        )
        app.state.key_manager = KeyManager(key_db_path)

        logger.info("Server Application successfully initialized!")
        logger.info(f"  - Debug mode: {debug}")
        logger.info(f"  - Log level: {_config.server.log_level}")
        logger.info(f"  - Database path: {_config.database.root_dir}")
        logger.info(f"  - API authentication: {'enabled' if _config.server.security.require_api_key else 'disabled'}")
        logger.info(f"  - CORS: {'enabled' if _config.server.security.cors_enabled else 'disabled'}")
        logger.info(f"  - Rate limiting: {'enabled' if _config.server.enable_rate_limiting else 'disabled'}")

        _warn_if_insecure_bind(_config, logger)

        yield

        # Shutdown
        logger.info("Shutting down LocalVectorDB Server")
        if hasattr(app.state, "db_manager"):
            app.state.db_manager.close_all()
        logger.info("Shutdown complete")

    app = FastAPI(
        title="LocalVectorDB",
        version="0.1.0",
        lifespan=lifespan,
        debug=debug,
    )
    app.state.config = _config

    # --- Middleware stack (order matters: last added = outermost) ---

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # Request body size limit (enforces server.max_request_size). Added after
    # RequestLoggingMiddleware so it sits outside it and rejects oversized
    # bodies before any router buffers them.
    app.add_middleware(RequestSizeLimitMiddleware, config=_config)

    # Host validation
    if _config.server.security.trusted_hosts:
        app.add_middleware(HostValidationMiddleware, config=_config)
        logger.info(f"Host header validation enabled for: {_config.server.security.trusted_hosts}")

    # Security headers
    if _config.server.security.security_headers_enabled:
        app.add_middleware(SecurityHeadersMiddleware, config=_config)

    # CORS
    if _config.server.security.cors_enabled:
        origins = _config.server.security.cors_allowed_origins
        if origins == "*":
            logger.warning("CORS enabled for all origins. This may be insecure in production.")
            allow_origins = ["*"]
        elif isinstance(origins, list) and origins:
            allow_origins = origins
        elif isinstance(origins, str) and origins:
            allow_origins = [origins]
        else:
            raise ConfigurationError("CORS enabled but `cors_allowed_origins` is invalid or empty.")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_methods=_config.server.security.cors_allowed_methods,
            allow_headers=_config.server.security.cors_allowed_headers,
            max_age=_config.server.security.cors_max_age,
        )

    # Rate limiting
    if _config.server.enable_rate_limiting:
        if not _SLOWAPI_AVAILABLE:
            raise ConfigurationError("Rate limiting enabled but slowapi not installed")

        # storage_uri must be passed through: without it slowapi silently falls back
        # to a per-process in-memory store, so under N workers the effective limit is
        # N x the configured one. Point it at Redis to share counters across workers.
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[_config.server.rate_limit],
            storage_uri=_config.server.rate_limit_storage_uri,
        )
        app.state.limiter = limiter
        # Use our own handler rather than slowapi's stock one: the stock handler
        # returns {"error": "<string>"}, which breaks the {"error": {...}} object
        # envelope every other response uses (and crashes clients that read
        # error["code"]). We still inject slowapi's rate-limit headers.
        app.add_exception_handler(RateLimitExceeded, _rate_limit_envelope_handler)
        logger.info(
            f"Rate limiting enabled: {_config.server.rate_limit} (storage: {_config.server.rate_limit_storage_uri})"
        )

    # --- Exception handlers (shared with tests via register_exception_handlers) ---
    register_exception_handlers(app, debug=debug)

    # --- Register routers ---
    from localvectordb_server.routers import register_routers

    register_routers(app)
    logger.info("API routes registered")

    return app
