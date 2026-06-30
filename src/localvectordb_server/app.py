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
    from slowapi import Limiter, _rate_limit_exceeded_handler
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

        limiter = Limiter(key_func=get_remote_address, default_limits=[_config.server.rate_limit])
        app.state.limiter = limiter
        # slowapi's handler is typed for its own RateLimitExceeded rather than the
        # broad Exception signature Starlette expects; the runtime contract is correct.
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
        logger.info(f"Rate limiting enabled: {_config.server.rate_limit}")

    # --- Exception handlers (shared with tests via register_exception_handlers) ---
    register_exception_handlers(app, debug=debug)

    # --- Register routers ---
    from localvectordb_server.routers import register_routers

    register_routers(app)
    logger.info("API routes registered")

    return app
