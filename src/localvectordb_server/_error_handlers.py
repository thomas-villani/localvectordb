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
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Dict, Optional, Tuple

from flask import current_app, g, jsonify
from werkzeug.exceptions import HTTPException

from localvectordb.exceptions import (
    BaseLocalVectorDBException,
    ConfigurationError,
    ConnectionPoolError,
    DatabaseNotFoundError,
    DuplicateDocumentIDError,
    EmbeddingError,
    OllamaNotFoundError,
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
        self.timestamp = datetime.now(UTC).isoformat() + 'Z'
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
    # print(str(repr(error)))
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
    """Validate search parameters with enhanced validation"""

    # Validate required fields
    validate_required_fields(data, ['query'])

    query = data["query"]
    if not isinstance(query, str) or not query.strip():
        raise ValidationError("Query must be a non-empty string", field="query")

    # Validate search_type
    search_type = data.get("search_type", "vector")
    if search_type not in ["vector", "keyword", "hybrid"]:
        raise ValidationError(
            "search_type must be one of: vector, keyword, hybrid",
            field="search_type",
            value=search_type
        )

    # Validate return_type (UPDATED)
    return_type = data.get("return_type", "documents")
    if return_type not in ["documents", "chunks", "context", "enriched"]:
        raise ValidationError(
            "return_type must be one of: documents, chunks, context, enriched",
            field="return_type",
            value=return_type
        )

    # Validate k
    k = data.get("k", 10)
    validate_field_type(data, "k", int)
    if k < 1 or k > 1000:
        raise ValidationError("k must be between 1 and 1000", field="k", value=k)

    # Validate score_threshold
    score_threshold = data.get("score_threshold", 0.0)
    validate_field_type(data, "score_threshold", (int, float))
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise ValidationError(
            "score_threshold must be between 0.0 and 1.0",
            field="score_threshold",
            value=score_threshold
        )

    # Validate vector_weight
    vector_weight = data.get("vector_weight", 0.7)
    validate_field_type(data, "vector_weight", (int, float))
    if vector_weight < 0.0 or vector_weight > 1.0:
        raise ValidationError(
            "vector_weight must be between 0.0 and 1.0",
            field="vector_weight",
            value=vector_weight
        )

    # Validate context_window
    context_window = data.get("context_window", 2)
    validate_field_type(data, "context_window", int)
    if context_window < 0 or context_window > 20:
        raise ValidationError(
            "context_window must be between 0 and 20",
            field="context_window",
            value=context_window
        )

    # Validate semantic_dedup_threshold
    semantic_dedup_threshold = data.get("semantic_dedup_threshold")
    if semantic_dedup_threshold is not None:
        validate_field_type(data, "semantic_dedup_threshold", (int, float))
        if semantic_dedup_threshold < 0.0 or semantic_dedup_threshold > 1.0:
            raise ValidationError(
                "semantic_dedup_threshold must be between 0.0 and 1.0",
                field="semantic_dedup_threshold",
                value=semantic_dedup_threshold
            )

    # Validate document_scoring_method
    document_scoring_method = data.get("document_scoring_method", "frequency_boost")
    valid_methods = ["best", "average", "worst", "weighted_average", "frequency_boost",
                     "harmonic_mean", "diminishing_returns", "statistical", "robust_mean",
                     "percentile", "geometric_mean"]

    if document_scoring_method not in valid_methods:
        raise ValidationError(
            f"document_scoring_method must be one of: {', '.join(valid_methods)}",
            field="document_scoring_method",
            value=document_scoring_method
        )

    document_scoring_options = data.get("document_scoring_options")
    if document_scoring_options and not isinstance(document_scoring_options, dict):
        raise ValidationError("document_scoring_options must be a dictionary object",
                              field="document_scoring_options",
                              value=document_scoring_options)

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
