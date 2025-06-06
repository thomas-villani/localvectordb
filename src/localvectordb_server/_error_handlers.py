# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb_server/_error_handlers.py
"""
Enhanced error handling framework for LocalVectorDB Server with
standardized error responses, validation, and recovery strategies.
"""

import logging
import traceback
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional, Tuple

from flask import jsonify, g, current_app
from werkzeug.exceptions import HTTPException

from localvectordb.exceptions import (
    BaseLocalVectorDBException, DatabaseNotFoundError,
    DuplicateDocumentIDError, EmbeddingError, ConfigurationError,
    ConnectionPoolError, OllamaNotFoundError
)


class APIError(Exception):
    """
    Standard API error with structured information
    """

    def __init__(
            self,
            message: str,
            error_code: str,
            status_code: int = 500,
            details: Optional[Dict[str, Any]] = None,
            recoverable: bool = False
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = datetime.utcnow().isoformat() + 'Z'
        self.request_id = getattr(g, 'request_id', None)
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response"""
        return {
            'error': {
                'message': self.message,
                'code': self.error_code,
                'timestamp': self.timestamp,
                'request_id': self.request_id,
                'details': self.details,
                'recoverable': self.recoverable
            }
        }


class ValidationError(APIError):
    """Error for input validation failures"""

    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        details = kwargs.pop('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['invalid_value'] = str(value)

        super().__init__(
            message=message,
            error_code='VALIDATION_ERROR',
            status_code=400,
            details=details,
            recoverable=True,
            **kwargs
        )


def standardize_error_response(
        error: Exception,
        default_message: str = "An unexpected error occurred",
        default_code: str = "INTERNAL_ERROR"
) -> Tuple[Dict[str, Any], int]:
    """
    Convert any exception to a standardized error response
    """
    logger = logging.getLogger('localvectordb.errors')
    print(str(repr(error)))
    # Handle our custom API errors
    if isinstance(error, APIError):
        logger.error(
            f"API Error: {error.message}",
            extra={
                'extra_fields': {
                    'error_code': error.error_code,
                    'status_code': error.status_code,
                    'recoverable': error.recoverable,
                    'details': error.details
                }
            }
        )
        return error.to_dict(), error.status_code

    # Handle HTTP exceptions from Werkzeug
    if isinstance(error, HTTPException):
        api_error = APIError(
            message=error.description or str(error),
            error_code=f"HTTP_{error.code}",
            status_code=error.code,
            recoverable=error.code < 500
        )
        return api_error.to_dict(), error.code

    # Handle LocalVectorDB exceptions
    if isinstance(error, DatabaseNotFoundError):
        api_error = APIError(
            message=str(error),
            error_code='DATABASE_NOT_FOUND',
            status_code=404,
            recoverable=True
        )
    elif isinstance(error, DuplicateDocumentIDError):
        api_error = APIError(
            message=str(error),
            error_code='DUPLICATE_DOCUMENT_ID',
            status_code=409,
            recoverable=True
        )
    elif isinstance(error, EmbeddingError):
        api_error = APIError(
            message=f"Embedding service error: {str(error)}",
            error_code='EMBEDDING_ERROR',
            status_code=503,
            recoverable=True,
            details={'service': 'embedding_provider'}
        )
    elif isinstance(error, OllamaNotFoundError):
        api_error = APIError(
            message=str(error),
            error_code='OLLAMA_NOT_AVAILABLE',
            status_code=503,
            recoverable=True,
            details={'service': 'ollama'}
        )
    elif isinstance(error, ConnectionPoolError):
        api_error = APIError(
            message="Database connection error",
            error_code='DATABASE_CONNECTION_ERROR',
            status_code=503,
            recoverable=True,
            details={'component': 'connection_pool'}
        )
    elif isinstance(error, ConfigurationError):
        api_error = APIError(
            message=f"Configuration error: {str(error)}",
            error_code='CONFIGURATION_ERROR',
            status_code=500,
            recoverable=False
        )
    elif isinstance(error, BaseLocalVectorDBException):
        api_error = APIError(
            message=str(error),
            error_code='DATABASE_ERROR',
            status_code=500,
            recoverable=False
        )
    else:
        # Unknown error - log full traceback and return generic error
        logger.error(
            f"Unexpected error: {str(error)}",
            exc_info=True,
            extra={
                'extra_fields': {
                    'error_type': type(error).__name__,
                    'error_message': str(error)
                }
            }
        )

        # In debug mode, include more details
        details = {}
        if current_app.debug:
            details = {
                'error_type': type(error).__name__,
                'traceback': traceback.format_exc()
            }

        api_error = APIError(
            message=default_message,
            error_code=default_code,
            status_code=500,
            recoverable=False,
            details=details
        )

    logger.error(
        f"Error Response: {api_error.message}",
        extra={
            'extra_fields': {
                'error_code': api_error.error_code,
                'status_code': api_error.status_code,
                'recoverable': api_error.recoverable
            }
        }
    )

    return api_error.to_dict(), api_error.status_code


def handle_errors(f):
    """
    Decorator to standardize error handling for route functions
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_response, status_code = standardize_error_response(e)
            return jsonify(error_response), status_code

    return wrapper


# Input validation utilities
def validate_required_fields(data: Dict[str, Any], required_fields: list) -> None:
    """Validate that required fields are present in request data"""
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")

    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)

    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            details={'missing_fields': missing_fields}
        )


def validate_field_type(data: Dict[str, Any], field: str, expected_type: type, required: bool = False) -> None:
    """Validate field type in request data"""
    if field not in data:
        if required:
            raise ValidationError(f"Required field '{field}' is missing", field=field)
        return

    value = data[field]
    if value is not None and not isinstance(value, expected_type):
        raise ValidationError(
            f"Field '{field}' must be of type {expected_type.__name__}",
            field=field,
            value=value
        )


def validate_pagination_params(page: Optional[int] = None, limit: Optional[int] = None) -> Tuple[int, int]:
    """Validate and normalize pagination parameters"""
    try:
        page = int(page) if page is not None else 1
        limit = int(limit) if limit is not None else 100
    except (ValueError, TypeError):
        raise ValidationError("Pagination parameters must be integers")

    if page < 1:
        raise ValidationError("Page number must be >= 1", field="page", value=page)

    if limit < 1 or limit > 1000:
        raise ValidationError(
            "Limit must be between 1 and 1000",
            field="limit",
            value=limit
        )

    return page, limit


def validate_search_params(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate search request parameters"""
    # Required fields
    validate_required_fields(data, ['query'])

    # Type validation
    validate_field_type(data, 'query', str, required=True)
    validate_field_type(data, 'search_type', str)
    validate_field_type(data, 'return_type', str)
    validate_field_type(data, 'k', int)
    validate_field_type(data, 'score_threshold', (int, float))
    validate_field_type(data, 'filters', dict)
    validate_field_type(data, 'vector_weight', (int, float))

    # Value validation
    if 'search_type' in data and data['search_type'] not in ['vector', 'keyword', 'hybrid']:
        raise ValidationError(
            "search_type must be one of: vector, keyword, hybrid",
            field='search_type',
            value=data['search_type']
        )

    if 'return_type' in data and data['return_type'] not in ['documents', 'chunks']:
        raise ValidationError(
            "return_type must be one of: documents, chunks",
            field='return_type',
            value=data['return_type']
        )

    if 'k' in data and (data['k'] < 1 or data['k'] > 1000):
        raise ValidationError(
            "k must be between 1 and 1000",
            field='k',
            value=data['k']
        )

    if 'score_threshold' in data and (data['score_threshold'] < 0 or data['score_threshold'] > 1):
        raise ValidationError(
            "score_threshold must be between 0 and 1",
            field='score_threshold',
            value=data['score_threshold']
        )

    if 'vector_weight' in data and (data['vector_weight'] < 0 or data['vector_weight'] > 1):
        raise ValidationError(
            "vector_weight must be between 0 and 1",
            field='vector_weight',
            value=data['vector_weight']
        )

    return data


def validate_database_creation_params(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate database creation parameters"""
    validate_required_fields(data, ['name'])

    # Validate name
    name = data['name']
    if not isinstance(name, str) or not name.strip():
        raise ValidationError("Database name must be a non-empty string", field='name', value=name)

    # Validate name doesn't contain invalid characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    if any(char in name for char in invalid_chars):
        raise ValidationError(
            f"Database name contains invalid characters: {invalid_chars}",
            field='name',
            value=name
        )

    # Validate optional fields
    if 'metadata_schema' in data:
        validate_field_type(data, 'metadata_schema', dict)

    if 'database' in data:
        validate_field_type(data, 'database', dict)
        db_config = data['database']

        if 'chunk_size' in db_config:
            validate_field_type(db_config, 'chunk_size', int)
            if db_config['chunk_size'] < 1 or db_config['chunk_size'] > 10000:
                raise ValidationError(
                    "chunk_size must be between 1 and 10000",
                    field='database.chunk_size',
                    value=db_config['chunk_size']
                )

    if 'embedding' in data:
        validate_field_type(data, 'embedding', dict)
        emb_config = data['embedding']

        if 'provider' in emb_config and emb_config['provider'] not in ['ollama', 'openai']:
            raise ValidationError(
                "embedding provider must be 'ollama' or 'openai'",
                field='embedding.provider',
                value=emb_config['provider']
            )

    return data



# Global error handlers for Flask app
def register_error_handlers(app):
    """Register global error handlers for the Flask app"""

    @app.errorhandler(APIError)
    def handle_api_error(error):
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(BaseLocalVectorDBException)
    def handle_vectordb_error(error):
        error_response, status_code = standardize_error_response(error)
        return jsonify(error_response), status_code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        error_response, status_code = standardize_error_response(error)
        return jsonify(error_response), status_code