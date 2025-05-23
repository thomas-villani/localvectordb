#  Copyright (c) 2023-2025 Tom Villani, Ph.D. All rights reserved.
"""
localvectordb_server/__init__.py

Server for interacting with `localvectordb.LocalVectorDB` via http
"""
import os
from typing import Union

from localvectordb.exceptions import ConfigurationError
from localvectordb_server._dbmanager import DatabaseManager
from localvectordb_server._logcfg import configure_logging
from localvectordb_server.config import Config, load_config


def create_app(configuration: Union[str, Config, None]=None,
               database_directory=None,
               debug=False,
               log_level=None,
               **kwargs):
    from flask import Flask
    """Application factory function"""
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
    app.config.update(config.to_flask_config())

    # Configure logging
    configure_logging(app)

    # Make sure DB directory exists
    if not os.path.exists(config.database.root_dir):
        os.makedirs(config.database.root_dir)

    app.logger.info(f"Database Root Path: {os.path.abspath(config.database.root_dir)}")

    # Configure CORS if enabled
    if config.server.cors_enabled:
        try:
            from flask_cors import CORS
        except ImportError:
            raise RuntimeError("CORS requires installation of `flask_cors`. Run `pip install flask_cors`")
        origins = config.server.cors_allowed_origins
        if origins == "*":
            app.logger.warning("CORS enabled for all origins. This may be insecure in production.")
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
            app.logger.info(f"CORS enabled for origins: {origins}")
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
            app.logger.info(f"CORS enabled for origin: {origins}")
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

    # Initialize database manager
    app.db_manager = DatabaseManager(app)

    # Register blueprints
    from localvectordb_server.routes import api
    app.register_blueprint(api)

    # Store config for access elsewhere
    app.lvdb_config = config

    app.logger.info("Server Application successfully initialized!")
    return app



if __name__ == "__main__":
    _app = create_app(debug=True)
    _app.run(debug=True)
