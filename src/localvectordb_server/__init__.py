#  Copyright (c) 2023-2025 Tom Villani, Ph.D. All rights reserved.
"""
localvectordb_server/__init__.py

Enhanced server for interacting with `localvectordb.LocalVectorDB` via http
with structured logging, error handling, and performance monitoring.
"""
import os
from typing import Union
import logging

from flask import Flask, request, abort
from werkzeug.middleware.proxy_fix import ProxyFix

from localvectordb.exceptions import ConfigurationError
from localvectordb_server._cache import cache
from localvectordb_server._dbmanager import DatabaseManager
from localvectordb_server._error_handlers import register_error_handlers
from localvectordb_server._logcfg import configure_logging, setup_request_logging
from localvectordb_server.config import Config, load_config
from localvectordb_server.keymanager import KeyManager

try:
    from flask_talisman import Talisman
    _FLASK_TALISMAN_AVAILABLE = True
except ImportError:
    Talisman = None
    _FLASK_TALISMAN_AVAILABLE = False

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    _FLASK_LIMITER_AVAILABLE = True
except ImportError:
    _FLASK_LIMITER_AVAILABLE = False
    Limiter = None
    get_remote_address = lambda: None

try:
    from flask_cors import CORS
    _FLASK_CORS_AVAILABLE = True
except ImportError:
    CORS = None
    _FLASK_CORS_AVAILABLE = False


def create_app(
        configuration: Union[str, Config, None] = None,
        database_directory=None,
        debug=False,
        log_level=None,
        **kwargs
        ):
    """Enhanced application factory function with improved logging and error handling"""


    app = Flask(__name__, instance_relative_config=False)

    # Start with default config
    _config = load_config(configuration)

    # Apply explicit CLI arguments
    if database_directory:
        _config.database.root_dir = database_directory

    if debug:
        app.debug = True
        _config.server.log_level = "DEBUG"

    if log_level and log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        _config.server.log_level = log_level

    # Handle other kwargs
    for key, value in kwargs.items():
        if key == 'host' and value:
            _config.server.host = value
        elif key == 'port' and value:
            _config.server.port = value

    # Apply config to Flask app
    flask_config = _config.to_flask_config()

    app.config.update(flask_config)
    app.config_obj = _config

    # Configure enhanced logging first
    log_file = None
    if not debug and _config.server.environment != 'development':
        # Set up log file in production
        log_dir = os.path.join(_config.database.root_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'localvectordb.log')

    configure_logging(app, log_file)

    # Set up request-level logging middleware
    setup_request_logging(app)

    # Setup Host header validation if trusted_hosts is configured
    if _config.server.trusted_hosts:
        @app.before_request
        def validate_host_header():
            """Validate the Host header against trusted hosts"""
            
            # Skip validation if running behind a trusted proxy
            if _config.server.proxy_enabled:
                return
            
            # Get the host from the request
            host = request.host
            
            # Check if the host is in the trusted hosts list
            if host not in _config.server.trusted_hosts:
                app.logger.warning(f"Rejected request with untrusted Host header: {host}")
                abort(400, description=f"Invalid Host header: {host}")

    # Register enhanced error handlers
    register_error_handlers(app)

    logger = logging.getLogger('localvectordb_server')
    
    if _config.server.trusted_hosts:
        logger.info(f"Host header validation enabled for: {_config.server.trusted_hosts}")
    logger.info("Starting LocalVectorDB Server initialization")

    # Make sure DB directory exists
    if not os.path.exists(_config.database.root_dir):
        os.makedirs(_config.database.root_dir)
        logger.info(f"Created database directory: {_config.database.root_dir}")

    logger.info(f"Database Root Path: {os.path.abspath(_config.database.root_dir)}")

    # Configure CORS if enabled
    if _config.server.cors_enabled:
        if not _FLASK_CORS_AVAILABLE:
            raise RuntimeError("CORS requires installation of `flask_cors`. Run `pip install flask_cors`")

        origins = _config.server.cors_allowed_origins
        if origins == "*":
            logger.warning("CORS enabled for all origins. This may be insecure in production.")
            CORS(
                app,
                resources={
                    r"/api/*": {
                        "origins": "*",
                        "methods": _config.server.cors_allowed_methods,
                        "allow_headers": _config.server.cors_allowed_headers,
                        "max_age": _config.server.cors_max_age
                    }
                }
            )
        elif isinstance(origins, list) and origins:
            logger.info(f"CORS enabled for origins: {origins}")
            CORS(
                app,
                resources={
                    r"/api/*": {
                        "origins": origins,
                        "methods": _config.server.cors_allowed_methods,
                        "allow_headers": _config.server.cors_allowed_headers,
                        "max_age": _config.server.cors_max_age
                    }
                }
            )
        elif isinstance(origins, str) and origins:
            logger.info(f"CORS enabled for origin: {origins}")
            CORS(
                app,
                resources={
                    r"/api/*": {
                        "origins": [origins],
                        "methods": _config.server.cors_allowed_methods,
                        "allow_headers": _config.server.cors_allowed_headers,
                        "max_age": _config.server.cors_max_age
                    }
                }
            )
        else:
            raise ConfigurationError("CORS enabled but `cors_allowed_origins` is invalid or empty.")

    if _config.server.proxy_enabled:
        logger.info("Enabling ProxyFix")

        if _config.server.proxy_settings:
            logger.debug(f"Proxy settings: {_config.server.proxy_settings}")
            app.wsgi_app = ProxyFix(app.wsgi_app, **_config.server.proxy_settings)
        else:  # Default config, single proxy, forward
            app.wsgi_app = ProxyFix(
                app.wsgi_app,
                x_for=1,  # Number of proxies setting X-Forwarded-For, one for single proxy
            )

    if _config.server.security_headers_enabled:
        security_config = {
            'force_https': _config.server.force_https,
            'strict_transport_security': _config.server.strict_transport_security,
            'strict_transport_security_max_age': _config.server.strict_transport_security_max_age,
            'content_security_policy': _config.server.content_security_policy,
            'content_type_nosniff': _config.server.content_type_nosniff,
            'x_frame_options': _config.server.x_frame_options,
            'x_xss_protection': _config.server.x_xss_protection,
            'referrer_policy': _config.server.referrer_policy
        }
        if not _FLASK_TALISMAN_AVAILABLE:
            raise RuntimeError("server.securty_headers_enabled = true, but `flask-talisman` not installed. "
                               "Install using `pip install flask-talisman>=1.1.0`")
        Talisman(app, **security_config)


    if _config.server.enable_rate_limiting:
        if not _FLASK_LIMITER_AVAILABLE:
            logger.error("Rate limiting requires flask-limiter. Install with: pip install flask-limiter")
            raise ConfigurationError("Rate limiting enabled but flask-limiter not installed")

        limiter = Limiter(
            key_func=get_remote_address,
            app=app,
            default_limits=[_config.server.rate_limit],
            storage_uri=_config.server.rate_limit_storage_uri
        )
        app.limiter = limiter
        logger.info(f"Rate limiting enabled: {_config.server.rate_limit}")

    # Initialize caching
    if _config.server.cache_enabled:
        logger.info(f"Caching enabled with {_config.server.cache_type}")
    else:
        logger.info("Caching disabled")
        app.config["CACHE_TYPE"] = "NullCache"

    cache.init_app(app)  # configured from flask's config, including whether disabled (NullCache)

    # Initialize database manager with error handling
    try:
        app.db_manager = DatabaseManager(app)
        logger.info("Database manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database manager: {e}", exc_info=True)
        raise ConfigurationError(f"Database manager initialization failed: {e}")

    app.key_manager = KeyManager(_config.server.key_database_path or os.path.join(_config.database.root_dir, "api_keys.db"))

    # Register blueprints
    from localvectordb_server.routes import api
    app.register_blueprint(api)
    logger.info("API routes registered")

    # Register inspector blueprint if enabled
    inspector_enabled = getattr(_config.server, 'inspector_enabled', False)
    if inspector_enabled:
        try:
            from localvectordb_server.inspector import inspector_bp
            app.register_blueprint(inspector_bp, url_prefix='/inspector')
            if app.config.get("SECRET_KEY") is None:
                app.config["SECRET_KEY"] = os.urandom(32)
            logger.info("Inspector UI registered at /inspector")

            if not app.config_obj.server.require_api_key:
                logger.warning("Inspector enabled without api-key protection.")
                logger.warning("**The inspector is available and allows full database access to anyone with "
                               "the url where the app is exposed.**")

        except ImportError as e:
            logger.warning(f"Inspector UI not available: {e}")
            inspector_bp = None
        except Exception as e:
            logger.error(f"Failed to register inspector UI: {e}")
    else:
        logger.info("Inspector UI disabled in configuration")

    # Store config for access elsewhere

    # Log final configuration
    logger.info("Server Application successfully initialized!")
    logger.info("Configuration summary:")
    logger.info(f"  - Debug mode: {app.debug}")
    logger.info(f"  - Log level: {_config.server.log_level}")
    logger.info(f"  - Database path: {_config.database.root_dir}")
    logger.info(f"  - API authentication: {'enabled' if _config.server.require_api_key else 'disabled'}")
    logger.info(f"  - CORS: {'enabled' if _config.server.cors_enabled else 'disabled'}")
    logger.info(f"  - Rate limiting: {'enabled' if _config.server.enable_rate_limiting else 'disabled'}")

    return app


if __name__ == "__main__":
    _app = create_app(debug=True)
    _app.run(debug=True)
