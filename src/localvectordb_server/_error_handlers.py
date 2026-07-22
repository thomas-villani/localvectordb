"""
Error handling framework for LocalVectorDB Server (FastAPI).
"""

import logging
from datetime import UTC, datetime
from typing import Any, Dict, Optional, Tuple

from localvectordb.exceptions import (
    BaseLocalVectorDBException,
    ConfigurationError,
    ConnectionPoolError,
    DatabaseNotFoundError,
    DocumentNotFoundError,
    DuplicateDocumentIDError,
    EmbeddingError,
    MetadataFilterError,
    OllamaNotFoundError,
    PatchConflictError,
    PatchError,
)
from localvectordb_server._logcfg import request_id_var

# Per-database routes are all namespaced under /api/v1/databases/{db_name}/...,
# so a database name never shares a path segment with a static top-level endpoint
# (health, search, embeddings, upload, ...). No names need to be reserved; the
# only constraint is that the name be filesystem/path-safe (checked below).
RESERVED_DATABASE_NAMES: frozenset[str] = frozenset()


class APIError(Exception):
    """Standard API error with structured information."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.recoverable = recoverable
        # isoformat() on an aware UTC datetime already yields a '+00:00' offset;
        # appending 'Z' produced an invalid '...+00:00Z'. Keep the ISO-8601 offset.
        self.timestamp = datetime.now(UTC).isoformat()
        self.request_id = request_id_var.get()
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "code": self.error_code,
                "timestamp": self.timestamp,
                "request_id": self.request_id,
                "details": self.details,
                "recoverable": self.recoverable,
            }
        }


class ValidationError(APIError):
    """Error for input validation failures."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None, **kwargs: Any):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)

        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details,
            recoverable=True,
            **kwargs,
        )


def standardize_error_response(
    error: Exception,
    default_message: str = "An unexpected error occurred",
    default_code: str = "INTERNAL_ERROR",
    debug: bool = False,
) -> Tuple[Dict[str, Any], int]:
    """Convert any exception to a standardized error response."""
    error_logger = logging.getLogger("localvectordb.errors")

    if isinstance(error, APIError):
        error_logger.error(
            f"API Error: {error.message}",
            extra={
                "extra_fields": {
                    "error_code": error.error_code,
                    "status_code": error.status_code,
                    "recoverable": error.recoverable,
                    "details": error.details,
                }
            },
        )
        return error.to_dict(), error.status_code

    if isinstance(error, DatabaseNotFoundError):
        api_error = APIError(message=str(error), error_code="DATABASE_NOT_FOUND", status_code=404, recoverable=True)
    elif isinstance(error, DocumentNotFoundError):
        # A missing document is a client-addressable 404, not a server fault.
        # Without this branch it falls through to BaseLocalVectorDBException -> 500
        # (e.g. comparison endpoints and multi-get of an unknown id).
        api_error = APIError(message=str(error), error_code="DOCUMENT_NOT_FOUND", status_code=404, recoverable=True)
    elif isinstance(error, DuplicateDocumentIDError):
        api_error = APIError(message=str(error), error_code="DUPLICATE_DOCUMENT_ID", status_code=409, recoverable=True)
    elif isinstance(error, PatchConflictError):
        # Stale expect_hash precondition on a patch -- the document changed under us.
        api_error = APIError(message=str(error), error_code="HASH_CONFLICT", status_code=409, recoverable=True)
    elif isinstance(error, PatchError):
        # Unmatched/ambiguous/overlapping patch op -- a client error, not a fault.
        api_error = APIError(message=str(error), error_code="PATCH_FAILED", status_code=422, recoverable=True)
    elif isinstance(error, MetadataFilterError):
        # Bad filter/order_by specs (unknown fields, unsupported operators, ...)
        # are client errors, not server faults.
        api_error = APIError(message=str(error), error_code="INVALID_FILTER", status_code=400, recoverable=True)
    elif isinstance(error, EmbeddingError):
        api_error = APIError(
            message=f"Embedding service error: {str(error)}",
            error_code="EMBEDDING_ERROR",
            status_code=503,
            recoverable=True,
            details={"service": "embedding_provider"},
        )
    elif isinstance(error, OllamaNotFoundError):
        api_error = APIError(
            message=str(error),
            error_code="OLLAMA_NOT_AVAILABLE",
            status_code=503,
            recoverable=True,
            details={"service": "ollama"},
        )
    elif isinstance(error, ConnectionPoolError):
        api_error = APIError(
            message="Database connection error",
            error_code="DATABASE_CONNECTION_ERROR",
            status_code=503,
            recoverable=True,
            details={"component": "connection_pool"},
        )
    elif isinstance(error, ConfigurationError):
        api_error = APIError(
            message=f"Configuration error: {str(error)}",
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            recoverable=False,
        )
    elif isinstance(error, BaseLocalVectorDBException):
        api_error = APIError(message=str(error), error_code="DATABASE_ERROR", status_code=500, recoverable=False)
    else:
        error_logger.error(
            f"Unexpected error: {str(error)}",
            exc_info=True,
            extra={"extra_fields": {"error_type": type(error).__name__, "error_message": str(error)}},
        )

        # The full traceback is logged above (exc_info=True) but deliberately kept
        # out of the HTTP response: exposing stack traces to clients leaks internal
        # structure/paths. In debug mode we surface only the exception class name.
        details = {"error_type": type(error).__name__} if debug else {}

        api_error = APIError(
            message=default_message, error_code=default_code, status_code=500, recoverable=False, details=details
        )

    error_logger.error(
        f"Error Response: {api_error.message}",
        extra={
            "extra_fields": {
                "error_code": api_error.error_code,
                "status_code": api_error.status_code,
                "recoverable": api_error.recoverable,
            }
        },
    )

    return api_error.to_dict(), api_error.status_code


# Input validation utilities (kept for router code that uses them)
def validate_required_fields(data: Dict[str, Any], required_fields: list) -> None:
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")
    missing_fields = [f for f in required_fields if f not in data or data[f] is None]
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}", details={"missing_fields": missing_fields}
        )


def validate_field_type(
    data: Dict[str, Any], field: str, expected_type: "type | tuple[type, ...]", required: bool = False
) -> None:
    if field not in data:
        if required:
            raise ValidationError(f"Required field '{field}' is missing", field=field)
        return
    value = data[field]
    if value is not None and not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_str = " or ".join(t.__name__ for t in expected_type)
        else:
            type_str = expected_type.__name__ if hasattr(expected_type, "__name__") else str(expected_type)
        raise ValidationError(f"Field '{field}' must be of type {type_str}", field=field, value=value)


def validate_pagination_params(page=None, limit=None) -> Tuple[int, int]:
    try:
        page = int(page) if page is not None else 1
        limit = int(limit) if limit is not None else 100
    except (ValueError, TypeError) as e:
        raise ValidationError("Pagination parameters must be integers") from e
    if page < 1:
        raise ValidationError("Page number must be >= 1", field="page", value=page)
    if limit < 1 or limit > 1000:
        raise ValidationError("Limit must be between 1 and 1000", field="limit", value=limit)
    return page, limit


def validate_search_params(data: Dict[str, Any]) -> Dict[str, Any]:
    validate_required_fields(data, ["query"])
    query = data["query"]
    if not isinstance(query, str) or not query.strip():
        raise ValidationError("Query must be a non-empty string", field="query")

    search_type = data.get("search_type", "hybrid")
    if search_type not in ["vector", "keyword", "hybrid"]:
        raise ValidationError(
            "search_type must be one of: vector, keyword, hybrid", field="search_type", value=search_type
        )

    return_type = data.get("return_type", "documents")
    if return_type not in ["documents", "chunks", "context", "enriched"]:
        raise ValidationError(
            "return_type must be one of: documents, chunks, context, enriched", field="return_type", value=return_type
        )

    k = data.get("k", 10)
    validate_field_type(data, "k", int)
    if k < 1 or k > 1000:
        raise ValidationError("k must be between 1 and 1000", field="k", value=k)

    score_threshold = data.get("score_threshold", 0.0)
    validate_field_type(data, "score_threshold", (int, float))
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise ValidationError(
            "score_threshold must be between 0.0 and 1.0", field="score_threshold", value=score_threshold
        )

    vector_weight = data.get("vector_weight", 0.5)
    validate_field_type(data, "vector_weight", (int, float))
    if vector_weight < 0.0 or vector_weight > 1.0:
        raise ValidationError("vector_weight must be between 0.0 and 1.0", field="vector_weight", value=vector_weight)

    context_window = data.get("context_window", 2)
    validate_field_type(data, "context_window", int)
    if context_window < 0 or context_window > 20:
        raise ValidationError("context_window must be between 0 and 20", field="context_window", value=context_window)

    semantic_dedup_threshold = data.get("semantic_dedup_threshold")
    if semantic_dedup_threshold is not None:
        validate_field_type(data, "semantic_dedup_threshold", (int, float))
        if semantic_dedup_threshold < 0.0 or semantic_dedup_threshold > 1.0:
            raise ValidationError(
                "semantic_dedup_threshold must be between 0.0 and 1.0",
                field="semantic_dedup_threshold",
                value=semantic_dedup_threshold,
            )

    document_scoring_method = data.get("document_scoring_method", "frequency_boost")
    valid_methods = [
        "best",
        "average",
        "frequency_boost",
    ]
    if document_scoring_method not in valid_methods:
        raise ValidationError(
            f"document_scoring_method must be one of: {', '.join(valid_methods)}",
            field="document_scoring_method",
            value=document_scoring_method,
        )

    document_scoring_options = data.get("document_scoring_options")
    if document_scoring_options and not isinstance(document_scoring_options, dict):
        raise ValidationError(
            "document_scoring_options must be a dictionary object",
            field="document_scoring_options",
            value=document_scoring_options,
        )

    semantic_filters = data.get("semantic_filters")
    if semantic_filters is not None:
        if not isinstance(semantic_filters, list):
            raise ValidationError("semantic_filters must be a list", field="semantic_filters", value=semantic_filters)
        for idx, filter_item in enumerate(semantic_filters):
            if not isinstance(filter_item, dict):
                raise ValidationError(f"semantic_filters[{idx}] must be a dictionary", field=f"semantic_filters[{idx}]")
            if "field" not in filter_item:
                raise ValidationError(
                    f"semantic_filters[{idx}] missing required field 'field'", field=f"semantic_filters[{idx}]"
                )
            if "concept" not in filter_item:
                raise ValidationError(
                    f"semantic_filters[{idx}] missing required field 'concept'", field=f"semantic_filters[{idx}]"
                )
            threshold = filter_item.get("threshold", 0.7)
            if not isinstance(threshold, (int, float)):
                raise ValidationError(
                    f"semantic_filters[{idx}].threshold must be a number",
                    field=f"semantic_filters[{idx}].threshold",
                    value=threshold,
                )
            if threshold < 0.0 or threshold > 1.0:
                raise ValidationError(
                    f"semantic_filters[{idx}].threshold must be between 0.0 and 1.0",
                    field=f"semantic_filters[{idx}].threshold",
                    value=threshold,
                )
            metric = filter_item.get("metric", "cosine")
            if metric not in ["cosine", "euclidean", "dot"]:
                raise ValidationError(
                    f"semantic_filters[{idx}].metric must be one of: cosine, euclidean, dot",
                    field=f"semantic_filters[{idx}].metric",
                    value=metric,
                )

    return data


def validate_database_creation_params(data: Dict[str, Any]) -> Dict[str, Any]:
    validate_required_fields(data, ["name"])
    name = data["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValidationError("Database name must be a non-empty string", field="name", value=name)
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    if any(char in name for char in invalid_chars):
        raise ValidationError(f"Database name contains invalid characters: {invalid_chars}", field="name", value=name)
    if "metadata_schema" in data:
        validate_field_type(data, "metadata_schema", dict)
    if "database" in data:
        validate_field_type(data, "database", dict)
        db_config = data["database"]
        if "chunk_size" in db_config:
            validate_field_type(db_config, "chunk_size", int)
            if db_config["chunk_size"] < 1 or db_config["chunk_size"] > 10000:
                raise ValidationError(
                    "chunk_size must be between 1 and 10000", field="database.chunk_size", value=db_config["chunk_size"]
                )
    if "embedding" in data:
        validate_field_type(data, "embedding", dict)
        emb_config = data["embedding"]
        if "provider" in emb_config:
            from localvectordb.embeddings import EmbeddingRegistry

            available = EmbeddingRegistry.list()
            provider = emb_config["provider"]
            if not isinstance(provider, str) or provider.lower() not in available:
                raise ValidationError(
                    f"embedding provider must be one of: {', '.join(sorted(available))}",
                    field="embedding.provider",
                    value=provider,
                )
    return data
