from typing import List, Union


class BaseLocalVectorDBException(Exception):
    pass


class DatabaseError(BaseLocalVectorDBException):
    pass


class DatabaseNotFoundError(DatabaseError, KeyError):
    """Raised if the Database cannot be found"""

    pass


class MetadataFilterError(DatabaseError, ValueError):
    """Raised when there's an error in metadata filter specification or processing"""

    pass


class DuplicateDocumentIDError(DatabaseError, ValueError):
    """Raised when inserting document(s) and the id(s) already exist"""

    pass


class DocumentNotFoundError(DatabaseError, KeyError):
    """Raised when one or more requested documents cannot be found"""

    def __init__(self, message: str, missing_ids: Union[str, List[str], None] = None):
        super().__init__(message)
        self.missing_ids = missing_ids if isinstance(missing_ids, list) else [missing_ids] if missing_ids else []


class EmbeddingError(BaseLocalVectorDBException, RuntimeError):
    pass


class OllamaNotFoundError(EmbeddingError):
    """Raised when Ollama is not installed or not running."""

    pass


class ConfigurationError(BaseLocalVectorDBException, RuntimeError):
    pass


class ValidationError(BaseLocalVectorDBException, ValueError):
    """Raised when there's a validation error in input data"""

    pass


class ConnectionPoolError(BaseLocalVectorDBException):
    pass


class RerankerError(BaseLocalVectorDBException, RuntimeError):
    """Raised when there's an error in reranking operations."""

    pass


class CursorError(BaseLocalVectorDBException):
    """Base class for cursor-related errors."""

    pass


class CursorExpiredError(CursorError):
    """Raised when a cursor has expired or been closed."""

    pass


class CursorExhaustedError(CursorError):
    """Raised when attempting to fetch from an exhausted cursor."""

    pass


# Reranking must score the fully materialized result set, which is incompatible
# with the lazy, batch-at-a-time hydration of cursor/streaming queries. Rather
# than silently ignore a configured reranker, the cursor/stream paths reject it.
# Lives here (a leaf module) so the consumers in database._search and
# query_builder can import it without a cursor <-> consumer import cycle.
_RERANK_STREAMING_UNSUPPORTED = (
    "Reranking is not supported with cursor/streaming queries because a reranker "
    "must score the fully materialized result set, which is incompatible with lazy "
    "cursor hydration. Use query()/query_async() (or QueryBuilder.execute()) for "
    "reranked results."
)
