#  Copyright (c) 2023-2025 Tom Villani, Ph.D. All rights reserved.
#

"""
localvectordb_server/_logcfg.py
Logging configuration for LocalVectorDB Server
"""

import logging.config
from typing import Optional


def configure_logging(app: "Flask", log_file: Optional[str] = None):
    """Configure logging for the application

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

    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': app.config.get("LOG_FORMAT", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console'],
                'level': level,
            },
            'localvectordb': {
                'handlers': ['console'],
                'level': level,
                'propagate': False
            },
            'localvectordb_server': {
                'handlers': ['console'],
                'level': level,
                'propagate': False
            },
            'localvectordb_server._auth': {
                'handlers': ['console'],
                'level': app.config.get("AUTH_LOG_LEVEL", level),
                'propagate': False
            }
        }
    }

    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
        # Add file handler to all loggers
        for logger in config['loggers'].values():
            logger['handlers'].append('file')

    logging.config.dictConfig(config)
