from typing import Dict, List, Optional, Union


class BaseLocalVectorDBException(Exception):
    """Base class for all LocalVectorDB exceptions."""


class DatabaseError(BaseLocalVectorDBException):
    """Raised for general database operation failures."""


class DatabaseNotFoundError(DatabaseError, KeyError):
    """Raised if the Database cannot be found"""

    pass


class MetadataFilterError(DatabaseError, ValueError):
    """Raised when there's an error in metadata filter specification or processing"""

    pass


class DuplicateDocumentIDError(DatabaseError, ValueError):
    """Raised when inserting document(s) and the id(s) already exist"""

    pass


class IngestError(DatabaseError):
    """
    Raised by ``upsert``/``insert`` (and their async twins) when one or more
    documents fail to embed or write and ``errors="raise"`` (the default).

    Each document is committed in its own transaction, so the documents that
    succeeded *are* durably persisted -- their IDs are in ``succeeded``. The
    documents that failed are in ``failures`` (``doc_id -> error message``).
    Pass ``errors="ignore"`` to suppress this and get best-effort ingestion that
    returns only the IDs that landed.
    """

    def __init__(
        self,
        message: str,
        failures: Optional[Dict[str, str]] = None,
        succeeded: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.failures: Dict[str, str] = failures or {}
        self.succeeded: List[str] = succeeded or []


class IndexIntegrityError(DatabaseError):
    """
    Raised when the SQLite rows and the FAISS index disagree in a way that
    corrupts query results -- notably duplicate ``faiss_id`` values, which cause
    one vector to be attributed to two different documents.

    Recover with ``lvdb db <name> repair``.
    """

    pass


class UnsupportedIndexOperationError(DatabaseError):
    """
    Raised when an operation is not supported by the configured FAISS index type.

    Most commonly: ``IndexHNSWFlat`` cannot remove vectors, so documents cannot be
    deleted or replaced in a database backed by it.
    """

    pass


class DocumentNotFoundError(DatabaseError, KeyError):
    """Raised when one or more requested documents cannot be found"""

    def __init__(self, message: str, missing_ids: Union[str, List[str], None] = None):
        super().__init__(message)
        self.missing_ids = missing_ids if isinstance(missing_ids, list) else [missing_ids] if missing_ids else []


class PatchConflictError(DatabaseError):
    """
    Raised when a document patch's ``expect_hash`` precondition does not match the
    stored ``content_hash`` -- i.e. the document changed since the caller read it.

    This is a distinct outcome from "document not found" and "no-op": the document
    exists and the ops are valid, but applying them would clobber a concurrent
    write. Surfaces as HTTP 409 on the server.
    """

    def __init__(self, message: str, expected: Union[str, None] = None, actual: Union[str, None] = None):
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class PatchError(DatabaseError, ValueError):
    """
    Raised when a document patch's ops cannot be applied: an unmatched or
    ambiguous ``find``, an out-of-range or overlapping splice, or a malformed op.

    The whole patch fails atomically -- no partial write. Surfaces as HTTP 422 on
    the server.
    """

    pass


class EmbeddingError(BaseLocalVectorDBException, RuntimeError):
    """Raised when an embedding provider fails to generate embeddings."""


class OllamaNotFoundError(EmbeddingError):
    """Raised when Ollama is not installed or not running."""

    pass


class ConfigurationError(BaseLocalVectorDBException, RuntimeError):
    """Raised when configuration is invalid or inconsistent."""


class ValidationError(BaseLocalVectorDBException, ValueError):
    """Raised when there's a validation error in input data"""

    pass


class ConnectionPoolError(BaseLocalVectorDBException):
    """Raised when a database connection cannot be acquired from the pool."""


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
