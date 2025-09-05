#  Copyright (c) 2023-2025 Tom Villani, Ph.D. All rights reserved.
"""
localvectordb_server/__init__.py

Enhanced server for interacting with `localvectordb.LocalVectorDB` via http
with structured logging, error handling, and performance monitoring.
"""
import logging
import os
from typing import Union

from localvectordb.exceptions import ConfigurationError
from localvectordb_server._dbmanager import DatabaseManager
from localvectordb_server._error_handlers import register_error_handlers
from localvectordb_server._logcfg import configure_logging, setup_request_logging
from localvectordb_server.config import Config, load_config


def create_app(
        configuration: Union[str, Config, None] = None,
        database_directory=None,
        debug=False,
        log_level=None,
        **kwargs
        ):
    """Enhanced application factory function with improved logging and error handling"""
    from flask import Flask

    app = Flask(__name__, instance_relative_config=False)

    # Start with default config
    config = load_config(configuration)

    # Apply explicit CLI arguments
    if database_directory:
        config.database.root_dir = database_directory

    if debug:
        app.debug = True
        config.server.log_level = "DEBUG"

    if log_level and log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        config.server.log_level = log_level

    # Handle other kwargs
    for key, value in kwargs.items():
        if key == 'host' and value:
            config.server.host = value
        elif key == 'port' and value:
            config.server.port = value

    # Apply config to Flask app
    flask_config = config.to_flask_config()

    app.config.update(flask_config)
    app.config_obj = config

    # Configure enhanced logging first
    log_file = None
    if not debug and config.server.environment != 'development':
        # Set up log file in production
        log_dir = os.path.join(config.database.root_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'localvectordb.log')

    configure_logging(app, log_file)

    # Set up request-level logging middleware
    setup_request_logging(app)

    # Register enhanced error handlers
    register_error_handlers(app)

    logger = logging.getLogger('localvectordb_server')
    logger.info("Starting LocalVectorDB Server initialization")

    # Make sure DB directory exists
    if not os.path.exists(config.database.root_dir):
        os.makedirs(config.database.root_dir)
        logger.info(f"Created database directory: {config.database.root_dir}")

    logger.info(f"Database Root Path: {os.path.abspath(config.database.root_dir)}")

    # Configure CORS if enabled
    if config.server.cors_enabled:
        try:
            from flask_cors import CORS
        except ImportError:
            raise RuntimeError("CORS requires installation of `flask_cors`. Run `pip install flask_cors`")

        origins = config.server.cors_allowed_origins
        if origins == "*":
            logger.warning("CORS enabled for all origins. This may be insecure in production.")
            CORS(
                app,
                resources={
                    r"/api/*": {
                        "origins": "*",
                        "methods": config.server.cors_allowed_methods,
                        "allow_headers": config.server.cors_allowed_headers,
                        "max_age": config.server.cors_max_age
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
                        "methods": config.server.cors_allowed_methods,
                        "allow_headers": config.server.cors_allowed_headers,
                        "max_age": config.server.cors_max_age
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
                        "methods": config.server.cors_allowed_methods,
                        "allow_headers": config.server.cors_allowed_headers,
                        "max_age": config.server.cors_max_age
                    }
                }
            )
        else:
            raise ConfigurationError("CORS enabled but `cors_allowed_origins` is invalid or empty.")

    if config.server.proxy_enabled:
        logger.info("Enabling ProxyFix")

        from werkzeug.middleware.proxy_fix import ProxyFix

        if config.server.proxy_settings:
            logger.debug(f"Proxy settings: {config.server.proxy_settings}")
            app.wsgi_app = ProxyFix(app.wsgi_app, **config.server.proxy_settings)
        else:  # Default config, single proxy, forward
            app.wsgi_app = ProxyFix(
                app.wsgi_app,
                x_for=1,  # Number of proxies setting X-Forwarded-For, one for single proxy
            )

    if config.server.enable_rate_limiting:
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address

            limiter = Limiter(
                key_func=get_remote_address,
                app=app,
                default_limits=[config.server.rate_limit],
                storage_uri=config.server.rate_limit_storage_uri
            )
            app.limiter = limiter
            logger.info(f"Rate limiting enabled: {config.server.rate_limit}")

        except ImportError:
            logger.error("Rate limiting requires flask-limiter. Install with: pip install flask-limiter")
            raise ConfigurationError("Rate limiting enabled but flask-limiter not installed")

    # Initialize caching
    from localvectordb_server._cache import cache

    if config.server.cache_enabled:
        logger.info(f"Caching enabled with {config.server.cache_type}")
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

    from localvectordb_server.keymanager import KeyManager
    app.key_manager = KeyManager(config.server.key_database_path or os.path.join(config.database.root_dir, "api_keys.db"))

    # Register blueprints
    from localvectordb_server.routes import api
    app.register_blueprint(api)
    logger.info("API routes registered")

    # Store config for access elsewhere
    app.lvdb_config = config

    # Log final configuration summary
    logger.info("Server Application successfully initialized!")
    logger.info("Configuration summary:")
    logger.info(f"  - Debug mode: {app.debug}")
    logger.info(f"  - Log level: {config.server.log_level}")
    logger.info(f"  - Database path: {config.database.root_dir}")
    logger.info(f"  - API authentication: {'enabled' if config.server.require_api_key else 'disabled'}")
    logger.info(f"  - CORS: {'enabled' if config.server.cors_enabled else 'disabled'}")
    logger.info(f"  - Rate limiting: {'enabled' if config.server.enable_rate_limiting else 'disabled'}")

    return app


if __name__ == "__main__":
    _app = create_app(debug=True)
    _app.run(debug=True)
