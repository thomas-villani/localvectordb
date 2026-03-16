# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/client.py

"""Remote interface for LocalVectorDB over HTTP.

This module provides a client interface to interact with a LocalVectorDB server.
It implements the same document-focused interface as the new LocalVectorDB class but
connects to a remote server via HTTP.

Main Components:

- RemoteVectorDB: Client for connecting to a LocalVectorDB server
- Document: Document object for remote use
- QueryResult: Search result object
- MetadataField: Metadata field definition

Examples
--------

Basic usage::

    from localvectordb.client import RemoteVectorDB
    from localvectordb.core import MetadataField, MetadataFieldType

    # Connect to an existing database
    db = RemoteVectorDB(
        name="my_database",
        base_url="http://localhost:5000",
        api_key="your_api_key"
    )

    # Upsert documents
    db.upsert(["Document 1", "Document 2"])

    # Search for similar documents
    results = db.query("search query", k=5)

Creating a new database with metadata schema::

    from localvectordb.core import MetadataField, MetadataFieldType

    db = RemoteVectorDB(
        name="new_database",
        base_url="http://localhost:5000",
        api_key="your_api_key",
        create_if_not_exists=True,
        metadata_schema={
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'publish_date': MetadataField(type=MetadataFieldType.DATE, indexed=True),
            'tags': MetadataField(type=MetadataFieldType.JSON)
        },
        embedding_model="nomic-embed-text",
        embedding_provider="ollama",
        chunk_size=512,
        chunking_method="sentences",
        chunk_overlap=1
    )

Document operations::

    # Upsert documents with metadata
    docs = ["Python programming guide", "Machine learning tutorial"]
    metadata = [
        {"author": "Jane Doe", "publish_date": "2024-01-01", "tags": ["python", "programming"]},
        {"author": "John Smith", "publish_date": "2024-02-01", "tags": ["ml", "ai"]}
    ]
    doc_ids = db.upsert(docs, metadata=metadata)

    # Get documents
    doc = db.get(doc_ids[0])
    docs = db.get(doc_ids)

    # Update a document
    db.update(doc_ids[0], content="Updated content", metadata={"author": "Jane Smith"})

    # Delete documents
    db.delete(doc_ids)

Unified search interface::

    # Vector search
    results = db.query("python programming", search_type="vector", k=5)

    # Keyword search
    results = db.query("python programming", search_type="keyword", k=5)

    # Hybrid search
    results = db.query("python programming", search_type="hybrid", k=5, vector_weight=0.7)

    # Search with filters
    results = db.query(
        "programming guide",
        search_type="vector",
        filters={"author": "Jane Doe", "publish_date": {">=": "2024-01-01"}}
    )

MongoDB-like filtering::

    # Filter documents by metadata
    docs = db.filter(where={"author": "Jane Doe"})

    # SQL filtering with ordering and pagination
    docs = db.filter(
        where={"publish_date": {"$gte": "2024-01-01"}},
        order_by="publish_date DESC",
        limit=10
    )

.. Note::

    This client requires a running LocalVectorDB v1.0 server. The interface is designed
    to be a drop-in replacement for the new LocalVectorDB, allowing code to work with
    either local or remote databases with minimal changes.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
import numpy as np

from localvectordb.core import (
    Chunk,
    Document,
    DocumentScoringMethod,
    MetadataField,
    MetadataFieldType,
    QueryResult,
)
from localvectordb.database import BaseVectorDB, TuningMixin
from localvectordb.embeddings import EmbeddingProvider, HTTPEmbeddingProvider
from localvectordb.exceptions import (
    BaseLocalVectorDBException,
    DatabaseError,
    DatabaseNotFoundError,
    DocumentNotFoundError,
    DuplicateDocumentIDError,
    EmbeddingError,
)
from localvectordb.sqlite_tuning import SqliteProfile

logger = logging.getLogger(__name__)


class RemoteQueryBuilder:
    """
    Remote query builder that serializes state for server-side execution.

    This class provides the same fluent API as the local QueryBuilder but
    instead of executing queries locally, it sends the complete query state
    to the server for processing. This eliminates the need for client-side
    embedding operations.

    Parameters
    ----------
    db : RemoteVectorDB
        The RemoteVectorDB instance to execute queries against

    Examples
    --------
    Basic query with filters::

        results = (db.query_builder()
            .search("machine learning")
            .filter("year", gte_=2020)
            .semantic_filter("methodology", "neural networks", threshold=0.8)
            .order_by("relevance", "desc")
            .limit(10)
            .execute())
    """

    def __init__(self, db: "RemoteVectorDB"):
        """Initialize the RemoteQueryBuilder."""
        self._db = db
        self._search_clauses = []
        self._exact_filters = []
        self._semantic_filters = []
        self._search_type = "hybrid"
        self._vector_weight = 0.7
        self._return_type = "documents"
        self._order_by = []
        self._limit = 10
        self._offset = 0
        self._group_by = []
        self._aggregations = []

    def clone(self) -> "RemoteQueryBuilder":
        """Create a copy of the current builder state."""
        new_builder = RemoteQueryBuilder(self._db)
        new_builder._search_clauses = self._search_clauses.copy()
        new_builder._exact_filters = self._exact_filters.copy()
        new_builder._semantic_filters = self._semantic_filters.copy()
        new_builder._search_type = self._search_type
        new_builder._vector_weight = self._vector_weight
        new_builder._return_type = self._return_type
        new_builder._order_by = self._order_by.copy()
        new_builder._limit = self._limit
        new_builder._offset = self._offset
        new_builder._group_by = self._group_by.copy()
        new_builder._aggregations = self._aggregations.copy()
        return new_builder

    def search(
        self,
        text: str,
        columns: Optional[List[str]] = None,
        search_type: Literal["vector", "keyword", "hybrid"] = "hybrid",
    ) -> "RemoteQueryBuilder":
        """Add a search clause to the query."""
        builder = self.clone()
        builder._search_clauses.append({"text": text, "columns": columns, "search_type": search_type})
        return builder

    def filter(self, field: str, **conditions) -> "RemoteQueryBuilder":
        """Add an exact filter to the query."""
        builder = self.clone()
        builder._exact_filters.append({"field": field, "conditions": conditions})
        return builder

    def semantic_filter(
        self, field: str, concept: str, threshold: float = 0.7, metric: Literal["cosine", "euclidean", "dot"] = "cosine"
    ) -> "RemoteQueryBuilder":
        """
        Add a semantic filter to the query.

        The semantic filtering is performed server-side using the database's
        configured embedding provider.

        Parameters
        ----------
        field : str
            The field to filter on ('content' for main text or metadata field name)
        concept : str
            The concept to match against
        threshold : float
            Minimum similarity score (0.0 to 1.0)
        metric : Literal["cosine", "euclidean", "dot"]
            Similarity metric to use

        Returns
        -------
        RemoteQueryBuilder
            A new builder with the semantic filter added
        """
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        builder = self.clone()
        builder._semantic_filters.append({"field": field, "concept": concept, "threshold": threshold, "metric": metric})
        return builder

    def order_by(self, field: str, direction: Literal["asc", "desc"] = "asc") -> "RemoteQueryBuilder":
        """Add ordering to the query."""
        builder = self.clone()
        builder._order_by.append({"field": field, "direction": direction})
        return builder

    def limit(self, n: int) -> "RemoteQueryBuilder":
        """Set the maximum number of results."""
        builder = self.clone()
        builder._limit = n
        return builder

    def offset(self, n: int) -> "RemoteQueryBuilder":
        """Set the result offset for pagination."""
        builder = self.clone()
        builder._offset = n
        return builder

    def group_by(self, *fields: str) -> "RemoteQueryBuilder":
        """Group results by fields."""
        builder = self.clone()
        builder._group_by = list(fields)
        return builder

    def aggregate(
        self, field: str, function: Literal["count", "sum", "avg", "min", "max"], alias: Optional[str] = None
    ) -> "RemoteQueryBuilder":
        """Add an aggregation to the query."""
        builder = self.clone()
        builder._aggregations.append({"field": field, "function": function, "alias": alias})
        return builder

    def with_search_type(self, search_type: Literal["vector", "keyword", "hybrid"]) -> "RemoteQueryBuilder":
        """Set the default search type."""
        builder = self.clone()
        builder._search_type = search_type
        return builder

    def with_vector_weight(self, weight: float) -> "RemoteQueryBuilder":
        """Set the vector weight for hybrid search."""
        builder = self.clone()
        builder._vector_weight = weight
        return builder

    def with_return_type(self, return_type: Literal["documents", "chunks", "context"]) -> "RemoteQueryBuilder":
        """Set the return type."""
        builder = self.clone()
        builder._return_type = return_type
        return builder

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the builder state to a dictionary."""
        return {
            "search_clauses": self._search_clauses,
            "exact_filters": self._exact_filters,
            "semantic_filters": self._semantic_filters,
            "search_type": self._search_type,
            "vector_weight": self._vector_weight,
            "return_type": self._return_type,
            "order_by": self._order_by,
            "limit": self._limit,
            "offset": self._offset,
            "group_by": self._group_by,
            "aggregations": self._aggregations,
        }

    def execute(self) -> List[QueryResult]:
        """
        Execute the query on the server.

        Returns
        -------
        List[QueryResult]
            The query results
        """
        return self._db._execute_query_builder(self.to_dict())

    async def execute_async(self) -> List[QueryResult]:
        """
        Execute the query on the server asynchronously.

        Returns
        -------
        List[QueryResult]
            The query results
        """
        return await self._db._execute_query_builder_async(self.to_dict())


class _RemoteEmbeddingProvider(HTTPEmbeddingProvider):
    """Embedding provider that proxies requests to a LocalVectorDB server.

    This provider mimics the interface of local embedding providers but makes
    HTTP requests to the server's embedding endpoint. This provides API parity
    with LocalVectorDB while keeping all embedding operations server-side.

    Parameters
    ----------
    model : str
        The model to use for embedding
    provider : str
        The embedding provider for the model
    dimension : int
        The dimension of the embedding provider (should be known beforehand)
    base_url : str
        The base url for the localvectordb_server
    api_key : str, optional
        The API key for the localvectordb_server, if auth is enabled.
    timeout : int, default = 90
        Timeout in seconds for the http request
    max_retries : int, default = 3
        How many times to retry on a failed request.
    retry_delay : float, default = 1.0
        How long to delay after a failed request (the backoff is exponential)
    max_concurrent_requests : int, default = 5
        How many requests to make concurrently to the LVDB server.
    authorization_header : str, default = "Authorization"
        For custom auth headers.
    """

    def __init__(
        self,
        model: str,
        *,
        provider: str,
        dimension: int,
        base_url: str,
        api_key: Optional[str] = None,
        timeout=90,
        max_retries=3,
        retry_delay=1.0,
        max_concurrent_requests=5,
        authorization_header="Authorization",
    ):
        super().__init__(
            model,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            base_url=base_url,
            max_concurrent_requests=max_concurrent_requests,
        )

        self._provider = provider
        self._model_name = model
        self._base_url = base_url
        self.api_key = api_key
        self.__authorization_header = authorization_header
        self._dimension = dimension

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def max_batch_size(self) -> int:
        return 128

    def validate_model(self) -> bool:
        """Check if the model exists.

        Should always exist because this shouldn't be instantiated until after the RemoteVectorDB is
        created, which involves confirming the embedding model on the server side."""
        return True

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension

    async def _embed_single_batch(self, texts, client: Optional[httpx.AsyncClient] = None, **kwargs):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.__authorization_header] = f"Bearer {self.api_key}"

        url = f"{self._base_url}/api/v1/embeddings"
        payload = {"provider": self._provider, "model": self._model_name, "texts": texts}

        if client is None:
            # Use context manager to ensure proper AsyncClient cleanup
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    raise RuntimeError(f"Server error: {data['error']['message']}")

                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings
        else:
            # Use provided client
            response = await client.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise RuntimeError(f"Server error: {data['error']['message']}")

            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings


class RemoteVectorDB(TuningMixin, BaseVectorDB):
    """Client for interacting with a LocalVectorDB server.

    This client provides the same interface as LocalVectorDB but connects to a remote server via HTTP.

    Parameters
    ----------
    name : str
        Name of the database
    base_url : str
        URL of the LocalVectorDB server (e.g., "http://localhost:5000")
    api_key : str, optional
        API key for authentication. If not provided, checks `LVDB_API_KEY` environment variable.
        Specify a custom environment variable by passing "$CUSTOM_ENV_VARIABLE" for this parameter.
    create_if_not_exists : bool, default=True
        Whether to create the database if it doesn't exist
    metadata_schema : Dict[str, MetadataField], optional
        Schema definition for metadata fields
    embedding_provider : str, optional
        Provider for embeddings, by default "ollama"
    embedding_model : str, optional
        Model to use for embeddings, by default "nomic-embed-text"
    embedding_config : Dict[str, Any], optional
        Configuration for embedding provider
    chunking_method : str, optional
        Method for chunking documents, by default "sentences"
    chunk_size : int, optional
        Maximum tokens per chunk, by default 500
    chunk_overlap : int, optional
        Number of tokens to overlap between chunks, by default 1
    enable_gpu : bool, optional
        Whether to use GPU for FAISS, by default False
    enable_fts : bool, optional
        Whether to enable full-text search, by default True
    request_timeout : int, optional
        Timeout for HTTP requests
    authorization_header : str, default = "Authorization"
        The server can be configured to accept alternate headers, and the client can too.
    max_retries : int
        Max number of retries for a request
    retry_delay : int
        The delay after a failed request before trying again
    max_concurrent_requests : int
        How many maximum requests for embeddings
    connection_pool_limits : httpx.Limits
        Parameters for the httpx client connection pool

    """

    def __init__(
        self,
        name: str,
        base_url: str = "http://127.0.0.1:5000",
        api_key: str = None,
        *,
        create_if_not_exists: bool = True,
        metadata_schema: Optional[Dict[str, MetadataField]] = None,
        embedding_provider: str = "ollama",
        embedding_model: str = "nomic-embed-text",
        embedding_config: Optional[Dict[str, Any]] = None,
        chunking_method: str = "sentences",
        chunk_size: int = 500,
        chunk_overlap: int = 1,
        enable_gpu: bool = False,
        enable_fts: bool = True,
        sqlite_profile: SqliteProfile = "balanced",
        sqlite_pragma_overrides: Optional[Dict[str, Any]] = None,
        request_timeout: int = None,
        authorization_header: str = "Authorization",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent_requests: int = 5,
        connection_pool_limits: Optional[httpx.Limits] = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")

        api_key_env_var = "LVDB_API_KEY"
        # Allow the user to specify an environment variable by prefixing $
        if api_key and api_key.startswith("$") and api_key[1:].isupper():
            api_key_env_var = api_key[1:]
            api_key = None

        self.api_key = api_key or os.getenv(api_key_env_var)
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connection_pool_limits = connection_pool_limits or httpx.Limits(
            max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0
        )

        # Configuration
        self._metadata_schema = metadata_schema or {}
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._embedding_config = embedding_config or {}
        self._chunking_method = chunking_method
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._enable_gpu = enable_gpu
        self._enable_fts = enable_fts
        self._sqlite_profile = sqlite_profile
        self._sqlite_pragma_overrides = sqlite_pragma_overrides or {}
        self._authorization_header = authorization_header

        self._last_ping_timestamp = 0
        self._last_ping_status = False

        # State variables to be loaded from server
        self._embedding_dimension = 0

        # HTTP clients for connection pooling (initialize before making requests)
        self._sync_client = None
        self._client = None

        # Check if database exists and create if needed
        if create_if_not_exists:
            self._ensure_database_exists()
        else:
            # Load existing database info
            self._load_database_info()

        self._remote_embedding_provider = _RemoteEmbeddingProvider(
            model=self._embedding_model,
            provider=self._embedding_provider,
            dimension=self._embedding_dimension,
            base_url=self.base_url,
            api_key=self.api_key,
            max_concurrent_requests=max_concurrent_requests,
            timeout=self.request_timeout,
            authorization_header=self._authorization_header,
        )

    def _ensure_sync_client(self) -> httpx.Client:
        """Ensure sync HTTP client is available for connection pooling"""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                timeout=httpx.Timeout(self.request_timeout) if self.request_timeout else None,
                limits=self._connection_pool_limits,
                headers=self._get_headers(),
            )
        return self._sync_client

    def _make_request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response | None:
        """Make HTTP request with exponential backoff retry"""

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                client = self._ensure_sync_client()
                response = client.request(method, url, **kwargs)

                # Don't retry on 4xx errors (client errors)
                if 400 <= response.status_code < 500:
                    return response

                # Success or 5xx error that we might retry
                if response.status_code < 500:
                    return response

                # 5xx error - might retry
                if attempt == self.max_retries:
                    return response  # Last attempt, return even if error

                # Wait before retry with exponential backoff
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Request failed with {response.status_code}, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
                time.sleep(delay)

            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e

                if attempt == self.max_retries:
                    raise ConnectionError(f"Failed to connect after {self.max_retries + 1} attempts: {e}") from e

                # Wait before retry
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Request failed with {type(e).__name__}: {e}, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
                time.sleep(delay)
        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        return None

    def _get_headers(self) -> dict:
        """Get headers for API requests including authentication if provided"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self._authorization_header] = f"Bearer {self.api_key}"
        return headers

    def _build_url(self, endpoint: str) -> str:
        """Build a full URL for the given endpoint"""
        return f"{self.base_url}{endpoint}"

    @staticmethod
    def _normalize_error_response(response: httpx.Response) -> tuple[str, str]:
        """Normalize error response to consistent format"""
        try:
            error_data = response.json()
            if isinstance(error_data, dict) and "error" in error_data:
                error_dict = error_data["error"]
                error_type = error_dict.get("code", "unknown").lower()
                error_msg = error_dict.get("message", "")
            else:
                # Handle cases where error response is not in expected format
                error_type = "malformed_response"
                error_msg = str(error_data) if error_data else f"HTTP {response.status_code}"
        except ValueError as e:
            logger.debug(f"Failed to parse error response as JSON: {e}")
            # Fallback to response text when JSON parsing fails
            error_msg = response.text or f"HTTP Error: {response.status_code}"
            error_type = "parse_error"

        # Ensure we have some error message
        if not error_msg:
            error_msg = f"HTTP {response.status_code}: {response.reason_phrase or 'Unknown error'}"

        return error_type, error_msg

    def _handle_response(self, response: httpx.Response) -> dict:
        """Handle API response and raise appropriate exceptions"""
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as e:
                logger.warning(f"Failed to parse successful response as JSON: {e}")
                raise BaseLocalVectorDBException(f"Invalid JSON response from server: {response.text[:200]}") from e

        # Try to parse error response as JSON with fallback to text
        error_type, error_msg = self._normalize_error_response(response)
        logger.debug(f"Client error: {error_type} - {error_msg}")

        # Map error type to appropriate exception
        error_map = {
            "database_not_found": DatabaseNotFoundError,
            "duplicate_document_id": DuplicateDocumentIDError,
            "embedding_error": EmbeddingError,
            "document_not_found": DocumentNotFoundError,
        }

        # Raise the appropriate exception if we recognize the type
        if error_type in error_map:
            raise error_map[error_type](error_msg)

        # Generic error mapping based on HTTP status
        if response.status_code == 404:
            raise DatabaseNotFoundError(error_msg)
        elif response.status_code == 400:
            raise ValueError(error_msg)
        elif response.status_code == 401:
            raise PermissionError("Authentication failed. Check your API key.")
        elif response.status_code == 409:
            raise DuplicateDocumentIDError(error_msg)
        else:
            raise BaseLocalVectorDBException(f"API Error: {error_msg}")

    def _ensure_database_exists(self) -> None:
        """Check if database exists and create if it doesn't"""
        try:
            # Try to get database info
            self._load_database_info()
        except DatabaseNotFoundError:
            # Database doesn't exist, create it
            self._create_database()

    def _load_database_info(self) -> None:
        """Load database information from server"""
        url = self._build_url(f"/api/v1/{self.name}/info")
        response = self._make_request_with_retry("GET", url)
        db_info = self._handle_response(response)

        # Update configuration from server
        config = db_info.get("config", {})
        self._embedding_provider = config.get("embedding_provider", self._embedding_provider)
        self._embedding_model = config.get("embedding_model", self._embedding_model)
        self._embedding_dimension = config.get("embedding_dimension", 0)
        self._chunking_method = config.get("chunking_method", self._chunking_method)
        self._chunk_size = config.get("chunk_size", self._chunk_size)
        self._chunk_overlap = config.get("chunk_overlap", self._chunk_overlap)
        self._enable_fts = config.get("fts_enabled", self._enable_fts)

        self._last_ping_timestamp = time.time()
        self._last_ping_status = True

        # Load metadata schema
        schema_data = config.get("metadata_schema", {})
        self._metadata_schema = {}
        for field_name, field_config in schema_data.items():
            self._metadata_schema[field_name] = MetadataField(
                type=MetadataFieldType(field_config["type"]),
                indexed=field_config.get("indexed", False),
                required=field_config.get("required", False),
                default_value=field_config.get("default_value"),
            )

    def _create_database(self) -> None:
        """Create a new database with the current configuration"""
        url = self._build_url("/api/v1/databases")

        # Serialize metadata schema
        metadata_schema_data = {}
        if self.metadata_schema:
            for field_name, field_def in self.metadata_schema.items():
                metadata_schema_data[field_name] = {
                    "type": field_def.type.value,
                    "indexed": field_def.indexed,
                    "required": field_def.required,
                    "default_value": field_def.default_value,
                }

        payload = {
            "name": self.name,
            "metadata_schema": metadata_schema_data,
            "embedding_provider": self._embedding_provider,
            "embedding_model": self._embedding_model,
            "embedding_config": self._embedding_config,
            "chunking_method": self._chunking_method,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "enable_gpu": self._enable_gpu,
            "enable_fts": self._enable_fts,
            "sqlite_profile": self._sqlite_profile,
            "sqlite_pragma_overrides": self._sqlite_pragma_overrides,
        }

        response = self._make_request_with_retry("POST", url, json=payload)

        created_db_info = self._handle_response(response)
        config = created_db_info.get("config", {})
        self._embedding_dimension = config.get("embedding_dimension", 0)

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Return the remote embedding provider instance.

        This provider proxies embedding requests to the server, providing
        API parity with LocalVectorDB while keeping operations server-side.
        """
        return self._remote_embedding_provider

    @property
    def metadata_schema(self) -> Dict[str, MetadataField]:
        return self._metadata_schema.copy()

    @property
    def embedding_model(self) -> str:
        """Return the embedding model name"""
        return self._embedding_model

    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of the embeddings"""
        return self._embedding_dimension

    @property
    def chunk_size(self) -> int:
        """Return the maximum tokens per chunk"""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Return the chunk overlap between chunks"""
        return self._chunk_overlap

    @property
    def chunking_method(self) -> str:
        """Return the chunking method"""
        return self._chunking_method

    @property
    def fts_enabled(self) -> bool:
        """Return whether full-text search is enabled"""
        return self._enable_fts

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        url = self._build_url(f"/api/v1/{self.name}/info")
        response = self._make_request_with_retry("GET", url)
        db_info = self._handle_response(response)
        return db_info.get("stats", {})

    def upsert(
        self,
        documents: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Insert or update documents in the database

        Parameters
        ----------
        documents : Union[str, List[str]]
            Document text(s) to add
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents
        ids : Optional[Union[str, List[str]]]
            Document IDs (auto-generated if not provided)
        batch_size : int
            Batch size for processing, by default 100
        similarity_threshold : float, optional
            Skip adding any chunks that are more similar than this value. Good for "pre-deduplication"

        Returns
        -------
        List[str]
            List of document IDs that were upserted
        """
        # Handle single document case
        if isinstance(documents, str):
            documents = [documents]
            if isinstance(metadata, dict):
                metadata = [metadata]
            if isinstance(ids, str):
                ids = [ids]

        # Prepare request payload
        payload = {"documents": documents, "batch_size": batch_size}

        if metadata is not None:
            payload["metadata"] = metadata

        if ids is not None:
            payload["ids"] = ids

        if similarity_threshold is not None:
            payload["similarity_threshold"] = similarity_threshold

        url = self._build_url(f"/api/v1/{self.name}/documents")
        response = self._make_request_with_retry("POST", url, json=payload)
        result = self._handle_response(response)

        return result.get("ids", [])

    def insert(
        self,
        documents: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
        errors: Literal["ignore", "raise"] = "raise",
    ) -> List[str]:
        """
        Insert new documents into the database

        Parameters
        ----------
        documents : Union[str, List[str]]
            Document text(s) to add
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents
        ids : Optional[Union[str, List[str]]]
            Document IDs (auto-generated if not provided)
        batch_size : int
            Batch size for processing, by default 100
        errors : Literal["ignore", "raise"]
            How to handle document ID conflicts, by default "raise"
        similarity_threshold : Optional[float]
            Skip chunks that are too similar to existing chunks

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted
        """
        # Handle single document case
        if isinstance(documents, str):
            documents = [documents]
            if isinstance(metadata, dict):
                metadata = [metadata]
            if isinstance(ids, str):
                ids = [ids]

        # Prepare request payload
        payload = {"documents": documents, "batch_size": batch_size, "errors": errors}

        if metadata is not None:
            payload["metadata"] = metadata

        if ids is not None:
            payload["ids"] = ids

        if similarity_threshold is not None:
            payload["similarity_threshold"] = similarity_threshold

        url = self._build_url(f"/api/v1/{self.name}/documents/insert")
        response = self._make_request_with_retry("POST", url, json=payload)
        result = self._handle_response(response)

        return result.get("ids", [])

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents matching filters.

        Parameters
        ----------
        filters : Dict[str, Any], optional
            Metadata filters to apply

        Returns
        -------
        int
            Number of matching documents
        """
        url = self._build_url(f"/api/v1/{self.name}/documents/count")

        # Prepare payload
        payload = {}
        if filters is not None:
            payload["filters"] = filters

        response = self._make_request_with_retry("POST", url, json=payload)
        result = self._handle_response(response)

        return result.get("count", 0)

    def upsert_from_file(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
        extractor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Insert or update documents from files using file extraction.

        Parameters
        ----------
        file_paths : Union[str, Path, List[Union[str, Path]]]
            Path(s) to files to extract and upsert
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents. Will be merged with extracted metadata.
        ids : Optional[Union[str, List[str]]]
            Document IDs. If not provided, will use filename without extension.
        batch_size : int
            Batch size for processing, by default 100
        similarity_threshold : Optional[float]
            Skip chunks that are too similar to existing chunks
        extractor_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments passed to the extractor

        Returns
        -------
        List[str]
            List of document IDs that were upserted

        Raises
        ------
        FileNotFoundError
            If any of the specified files don't exist
        ValueError
            If extraction fails for any file
        """
        # Normalize file paths to list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]

        # Normalize other inputs
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]

        # Validate inputs
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")

        # Check files exist
        for file_path in file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

        # Prepare multipart form data
        url = self._build_url(f"/api/v1/{self.name}/upload")

        # Build form data
        form_data = {"batch_size": str(batch_size), "mode": "upsert"}  # Specify upsert mode

        if metadata is not None:
            form_data["metadata"] = json.dumps(metadata)

        if ids is not None:
            form_data["ids"] = json.dumps(ids)

        if similarity_threshold is not None:
            form_data["similarity_threshold"] = str(similarity_threshold)

        if extractor_kwargs:
            form_data["extractor_kwargs"] = json.dumps(extractor_kwargs)

        # Prepare files for streaming upload
        files = []
        file_handles = []
        result = {}
        try:
            for file_path in file_paths:
                file_handle = open(file_path, "rb")
                file_handles.append(file_handle)
                files.append(("files", (file_path.name, file_handle, "application/octet-stream")))

            # Make request with streaming files
            response = self._make_request_with_retry("POST", url, data=form_data, files=files)
            result = self._handle_response(response)
        finally:
            # Ensure all file handles are closed
            for file_handle in file_handles:
                try:
                    file_handle.close()
                except Exception:
                    pass  # Ignore close errors

        return result.get("document_ids", [])

    def insert_from_file(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
        errors: Literal["ignore", "raise"] = "raise",
        extractor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Insert new documents from files using file extraction.

        Parameters
        ----------
        file_paths : Union[str, Path, List[Union[str, Path]]]
            Path(s) to files to extract and insert
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents. Will be merged with extracted metadata.
        ids : Optional[Union[str, List[str]]]
            Document IDs. If not provided, will use filename without extension.
        batch_size : int
            Batch size for processing, by default 100
        similarity_threshold : Optional[float]
            Skip chunks that are too similar to existing chunks
        errors : Literal["ignore", "raise"]
            How to handle document ID conflicts, by default "raise"
        extractor_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments passed to the extractor

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted

        Raises
        ------
        FileNotFoundError
            If any of the specified files don't exist
        ValueError
            If extraction fails for any file
        DuplicateDocumentIDError
            If errors="raise" and document ID conflicts occur
        """
        # Normalize file paths to list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]

        # Normalize other inputs
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]

        # Validate inputs
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")

        # Check files exist
        for file_path in file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

        # Prepare multipart form data
        url = self._build_url(f"/api/v1/{self.name}/upload")

        # Build form data
        form_data = {"batch_size": str(batch_size), "mode": "insert", "errors": errors}  # Specify insert mode

        if metadata is not None:
            form_data["metadata"] = json.dumps(metadata)

        if ids is not None:
            form_data["ids"] = json.dumps(ids)

        if similarity_threshold is not None:
            form_data["similarity_threshold"] = str(similarity_threshold)

        if extractor_kwargs:
            form_data["extractor_kwargs"] = json.dumps(extractor_kwargs)

        # Prepare files for streaming upload
        files = []
        file_handles = []
        result = {}
        try:
            for file_path in file_paths:
                file_handle = open(file_path, "rb")
                file_handles.append(file_handle)
                files.append(("files", (file_path.name, file_handle, "application/octet-stream")))

            # Make request with streaming files
            response = self._make_request_with_retry("POST", url, data=form_data, files=files)
            result = self._handle_response(response)
        finally:
            # Ensure all file handles are closed
            for file_handle in file_handles:
                try:
                    file_handle.close()
                except Exception:
                    pass  # Ignore close errors

        return result.get("document_ids", [])

    def upsert_from_chunks(
        self,
        chunks_by_document: Dict[str, Union[List["Chunk"], List[str]]],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Upsert documents from pre-chunked data.

        Parameters
        ----------
        chunks_by_document : Dict[str, Union[List[Chunk], List[str]]]
            Dictionary mapping document IDs to their chunks. Chunks can be either:
            - List[Chunk]: Full Chunk objects with position information
            - List[str]: Simple strings that will be converted to Chunk objects
        metadata : Optional[Dict[str, Dict[str, Any]]], default=None
            Dictionary mapping document IDs to their metadata
        batch_size : int, default=100
            Number of embeddings to generate at once
        similarity_threshold : Optional[float], default=None
            If provided, filters out chunks that are too similar to existing chunks

        Returns
        -------
        List[str]
            List of document IDs that were upserted
        """
        # Convert Chunk objects to serializable format
        serializable_chunks = {}
        for doc_id, chunks in chunks_by_document.items():
            if chunks and hasattr(chunks[0], "__dict__"):
                # Convert Chunk objects to dicts
                serializable_chunks[doc_id] = [
                    (
                        {
                            "text": chunk.content,
                            "position": chunk.position.to_dict(),
                            "total_chunks": getattr(chunk, "total_chunks", len(chunks)),
                            "metadata": getattr(chunk, "metadata", {}),
                        }
                        if hasattr(chunk, "content")
                        else str(chunk)
                    )
                    for chunk in chunks
                ]
            else:
                # Already strings or serializable
                serializable_chunks[doc_id] = chunks

        # Prepare request payload
        payload = {"chunks_by_document": serializable_chunks, "batch_size": batch_size}

        if metadata is not None:
            payload["metadata"] = metadata

        if similarity_threshold is not None:
            payload["similarity_threshold"] = similarity_threshold

        url = self._build_url(f"/api/v1/{self.name}/documents/chunks")
        response = self._make_request_with_retry("POST", url, json=payload)
        result = self._handle_response(response)

        return result.get("ids", [])

    def insert_from_chunks(
        self,
        chunks_by_document: Dict[str, Union[List["Chunk"], List[str]]],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
        errors: Literal["ignore", "raise"] = "raise",
    ) -> List[str]:
        """
        Insert documents from pre-chunked data with conflict handling.

        Parameters
        ----------
        chunks_by_document : Dict[str, Union[List[Chunk], List[str]]]
            Dictionary mapping document IDs to their chunks
        metadata : Optional[Dict[str, Dict[str, Any]]], default=None
            Dictionary mapping document IDs to their metadata
        batch_size : int, default=100
            Number of embeddings to generate at once
        similarity_threshold : Optional[float], default=None
            If provided, filters out chunks that are too similar to existing chunks
        errors : Literal["ignore", "raise"], default="raise"
            How to handle document ID conflicts

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted

        Raises
        ------
        DuplicateDocumentIDError
            If a document ID already exists and errors="raise"
        """
        # Convert Chunk objects to serializable format
        serializable_chunks = {}
        for doc_id, chunks in chunks_by_document.items():
            if chunks and hasattr(chunks[0], "__dict__"):
                # Convert Chunk objects to dicts
                serializable_chunks[doc_id] = [
                    (
                        {
                            "text": chunk.content,
                            "position": chunk.position.to_dict(),
                            "total_chunks": getattr(chunk, "total_chunks", len(chunks)),
                            "metadata": getattr(chunk, "metadata", {}),
                        }
                        if hasattr(chunk, "content")
                        else str(chunk)
                    )
                    for chunk in chunks
                ]
            else:
                # Already strings or serializable
                serializable_chunks[doc_id] = chunks

        # Prepare request payload
        payload = {"chunks_by_document": serializable_chunks, "batch_size": batch_size, "errors": errors}

        if metadata is not None:
            payload["metadata"] = metadata

        if similarity_threshold is not None:
            payload["similarity_threshold"] = similarity_threshold

        url = self._build_url(f"/api/v1/{self.name}/documents/chunks/insert")
        response = self._make_request_with_retry("POST", url, json=payload)
        result = self._handle_response(response)

        return result.get("ids", [])

    def get(self, ids: Union[str, List[str]]) -> Union[Document, List[Document]]:
        """
        Retrieve documents by ID

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to retrieve

        Returns
        -------
        Union[Document, List[Document]]
            Retrieved document(s)

        Raises
        ------
        DocumentNotFoundError
            If any requested documents are not found
        """
        single_id = isinstance(ids, str)
        requested_ids = [ids] if single_id else ids

        if single_id:
            url = self._build_url(f"/api/v1/{self.name}/documents/{ids}")
            response = self._make_request_with_retry("GET", url)

            result = self._handle_response(response)
            doc = Document.from_dict(result)
            return doc
        else:
            url = self._build_url(f"/api/v1/{self.name}/documents?ids={','.join(requested_ids)}")
            response = self._make_request_with_retry("GET", url)

            result = self._handle_response(response)
            missing_ids = result["missing_ids"]
            if missing_ids:
                raise DocumentNotFoundError(f"Documents not found: {', '.join(missing_ids)}", missing_ids)

            documents = [Document.from_dict(d) for d in result["documents"]]

            # Ensure documents are returned in the same order as requested
            id_to_doc = {doc.id: doc for doc in documents}
            ordered_documents = [id_to_doc[doc_id] for doc_id in requested_ids]

            return ordered_documents

    def get_chunk_embeddings(self, chunk_ids: str | List[str]) -> np.ndarray:
        """Get embeddings for existing chunks in the database

        Parameters
        ----------
        chunk_ids : str | List[str]
            The chunks for which to retrieve the embeddings

        Returns
        -------
        numpy.ndarray

        """
        single_id = isinstance(chunk_ids, str)
        payload = {"ids": ([chunk_ids] if single_id else chunk_ids)}

        url = self._build_url(f"/api/v1/{self.name}/embeddings")
        response = self._make_request_with_retry("POST", url, json=payload)
        results = self._handle_response(response)

        embeddings = results.get("embeddings")[0] if single_id else results.get("embeddings")
        return np.array(embeddings)

    def exists(self, ids: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """
        Check if documents exist

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to check

        Returns
        -------
        Union[bool, List[bool]]
            Existence status for each ID
        """
        single_id = isinstance(ids, str)
        payload = {"ids": ([ids] if single_id else ids)}

        url = self._build_url(f"/api/v1/{self.name}/documents/exists")
        response = self._make_request_with_retry("POST", url, json=payload)
        results = self._handle_response(response)

        return results.get("exists")[0] if single_id else results.get("exists")

    def delete(self, ids: Union[str, List[str]]) -> int:
        """
        Delete documents

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to delete

        Returns
        -------
        int
            Number of documents deleted
        """
        if isinstance(ids, str):
            ids = [ids]

        # Use batch endpoint for multiple IDs (threshold: 2+ IDs)
        if len(ids) >= 2:
            url = self._build_url(f"/api/v1/{self.name}/documents/delete")
            payload = {"ids": ids}
            response = self._make_request_with_retry("POST", url, json=payload)
            result = self._handle_response(response)
            return result.get("deleted_count", 0)

        # Single ID: use original DELETE endpoint
        deleted_count = 0
        for doc_id in ids:
            url = self._build_url(f"/api/v1/{self.name}/documents/{doc_id}")
            response = self._make_request_with_retry("DELETE", url)
            result = self._handle_response(response)
            deleted_count += result.get("deleted_count", 0)

        return deleted_count

    def update(self, doc_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a document's content and/or metadata

        Parameters
        ----------
        doc_id : str
            Document ID to update
        content : Optional[str]
            New content (if None, content is not updated)
        metadata : Optional[Dict[str, Any]]
            New metadata (merged with existing)

        Returns
        -------
        bool
            True if document was updated, False if not found
        """
        if not content and not metadata:
            return False

        payload = {}
        if content is not None:
            payload["content"] = content
        if metadata is not None:
            payload["metadata"] = metadata

        url = self._build_url(f"/api/v1/{self.name}/documents/{doc_id}")
        response = self._make_request_with_retry("PUT", url, json=payload)

        try:
            result = self._handle_response(response)
            return result.get("updated", False)
        except DatabaseNotFoundError:
            return False

    def query(
        self,
        query: str,
        *,
        search_type: Literal["vector", "keyword", "hybrid"] = "vector",
        return_type: Literal["documents", "chunks", "context", "enriched"] = "documents",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        # NEW PARAMETERS:
        context_window: int = 2,
        semantic_dedup_threshold: Optional[float] = None,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: dict = None,
    ) -> List[QueryResult]:
        """
        Unified query interface for all search types

        Parameters
        ----------
        query : str
            Query text
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Literal['documents', 'chunks', 'context', 'enriched']
            Whether to return full documents, individual chunks, or chunks with context
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score threshold (0-1, higher=better)
        filters : Optional[Dict[str, Any]]
            Metadata filters
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)
        context_window : int
            Number of chunks before and after to include when return_type='context'
        semantic_dedup_threshold : Optional[float]
            Similarity threshold for semantic deduplication (0-1, higher=more similar)
        document_scoring_method : str
            Method for aggregating chunk scores into document scores
        document_scoring_options : dict, optional
            Parameters controlling the various scoring methods

        Returns
        -------
        List[QueryResult]
            Search results with normalized scores
        """
        # Prepare request payload
        payload = {
            "query": query,
            "search_type": search_type,
            "return_type": return_type,
            "k": k,
            "score_threshold": score_threshold,
            "vector_weight": vector_weight,
            "context_window": context_window,
            "document_scoring_method": document_scoring_method,
            "document_scoring_options": document_scoring_options,
        }

        if filters is not None:
            payload["filters"] = filters

        if semantic_dedup_threshold is not None:
            payload["semantic_dedup_threshold"] = semantic_dedup_threshold

        url = self._build_url(f"/api/v1/{self.name}/query")
        response = self._make_request_with_retry("POST", url, json=payload)
        result = self._handle_response(response)

        # Process results
        raw_results = result.get("results", [])
        return [QueryResult.from_dict(res) for res in raw_results]

    def query_multi_column(
        self,
        query: str,
        *,
        columns: Optional[List[str]] = None,
        search_type: Literal["vector", "keyword", "hybrid"] = "vector",
        return_type: Literal["documents", "chunks"] = "documents",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: dict = None,
    ) -> List[QueryResult]:
        """
        Query across multiple columns (main content + embedding-enabled metadata fields)

        Parameters
        ----------
        query : str
            Query text
        columns : Optional[List[str]]
            Specific columns to search. If None, searches all embedding-enabled fields
            plus main content. Use 'content' for main document content.
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Literal['documents', 'chunks']
            Whether to return full documents or individual chunks
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score threshold (0-1, higher=better)
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)
        document_scoring_method : DocumentScoringMethod
            Method for aggregating chunk scores into document scores
        document_scoring_options : dict, optional
            Parameters for the scoring method

        Returns
        -------
        List[QueryResult]
            Search results with column attribution
        """
        # Prepare request payload
        payload = {
            "query": query,
            "search_type": search_type,
            "return_type": return_type,
            "k": k,
            "score_threshold": score_threshold,
            "vector_weight": vector_weight,
            "document_scoring_method": document_scoring_method,
            "document_scoring_options": document_scoring_options,
        }

        if columns is not None:
            payload["columns"] = columns

        if filters is not None:
            payload["filters"] = filters

        url = self._build_url(f"/api/v1/{self.name}/query-multi-column")
        response = self._make_request_with_retry("POST", url, json=payload)
        result = self._handle_response(response)

        # Process results
        raw_results = result.get("results", [])
        return [QueryResult.from_dict(res) for res in raw_results]

    def filter(
        self,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Document]:
        """
        Filter documents using enhanced metadata filtering

        This method now supports advanced MongoDB-style filtering with operators
        like $gt, $lt, $contains, $exists, etc. Raw SQL support has been removed
        for security reasons.

        Parameters
        ----------
        where : Optional[Dict[str, Any]]
            Filter conditions using either simple format or MongoDB-style operators.

            Simple format::

                {"author": "John Doe", "year": 2023}

            Advanced format with operators::

                {
                    "author": {"$eq": "John Doe"},
                    "year": {"$gte": 2020, "$lte": 2024},
                    "tags": {"$contains": "python"},
                    "rating": {"$in": [4, 5]},
                    "$and": [
                        {"category": "tech"},
                        {"$or": [{"lang": "en"}, {"lang": "es"}]}
                    ]
                }

            Supported operators:

            - Comparison: $eq, $ne, $gt, $lt, $gte, $lte, $in, $nin
            - String: $like, $ilike, $contains, $startswith, $endswith
            - Existence: $exists, $not_exists
            - Type: $type
            - Logical: $and, $or, $not
            - JSON: $contains, $not_contains (for JSON fields)

        order_by : Optional[str]
            ORDER BY clause (field name with optional ASC/DESC)
            Examples: "created_at DESC", "author ASC", "rating"
        limit : Optional[int]
            Maximum number of results
        offset : int
            Number of results to skip

        Returns
        -------
        List[Document]
            Filtered documents

        Examples
        --------
        Simple filtering::

            # Simple equality
            docs = db.filter(where={"author": "John Doe"})

            # Multiple conditions (AND)
            docs = db.filter(where={"author": "John Doe", "year": 2023})

        Advanced filtering::

            # Range queries
            docs = db.filter(where={
                "year": {"$gte": 2020, "$lte": 2024},
                "rating": {"$gt": 4.0}
            })

            # String operations
            docs = db.filter(where={
                "title": {"$contains": "python"},
                "author": {"$startswith": "Dr."}
            })

            # List operations
            docs = db.filter(where={
                "category": {"$in": ["tech", "science"]},
                "tags": {"$contains": "tutorial"}
            })

            # Logical operations
            docs = db.filter(where={
                "$and": [
                    {"year": {"$gte": 2020}},
                    {"$or": [
                        {"author": "John Doe"},
                        {"author": "Jane Smith"}
                    ]}
                ]
            })

            # Existence checks
            docs = db.filter(where={
                "optional_field": {"$exists": False},
                "required_field": {"$exists": True}
            })

        Ordering and pagination::

            # Order by field
            docs = db.filter(
                where={"category": "tech"},
                order_by="created_at DESC",
                limit=10,
                offset=20
            )

        Notes
        -----
        - All queries are converted to safe parameterized SQL on the server
        - Field names are validated against the metadata schema
        - Raw SQL is no longer supported to prevent injection attacks
        - JSON fields support special operations like $contains
        """

        # Prepare request payload
        payload = {"where": where, "offset": offset}

        if order_by is not None:
            payload["order_by"] = order_by

        if limit is not None:
            payload["limit"] = limit

        url = self._build_url(f"/api/v1/{self.name}/filter")
        response = self._make_request_with_retry("POST", url, json=payload)
        result = self._handle_response(response)

        # Process results
        raw_docs = result.get("documents", [])
        return [Document.from_dict(doc) for doc in raw_docs]

    def query_builder(self) -> RemoteQueryBuilder:
        """
        Create a new RemoteQueryBuilder for this database.

        The RemoteQueryBuilder provides a fluent API for building complex queries
        that are executed server-side, including semantic filters.

        Returns
        -------
        RemoteQueryBuilder
            A new query builder instance

        Examples
        --------
        Basic query with semantic filter::

            results = (db.query_builder()
                .search("machine learning")
                .filter("year", gte_=2020)
                .semantic_filter("methodology", "neural networks", threshold=0.8)
                .limit(10)
                .execute())

        Complex multi-criteria query::

            results = (db.query_builder()
                .search("quantum computing", search_type="hybrid")
                .filter("publication_type", "paper")
                .filter("citations", gte_=50)
                .semantic_filter("approach", "quantum machine learning", threshold=0.75)
                .order_by("publication_date", "desc")
                .limit(25)
                .execute())
        """
        return RemoteQueryBuilder(self)

    def _execute_query_builder(self, query_state: Dict[str, Any]) -> List[QueryResult]:
        """
        Execute a query builder request on the server.

        Parameters
        ----------
        query_state : Dict[str, Any]
            The serialized query builder state

        Returns
        -------
        List[QueryResult]
            The query results
        """
        url = self._build_url(f"/api/v1/{self.name}/query_builder")
        response = self._make_request_with_retry("POST", url, json=query_state)
        result = self._handle_response(response)

        # Process results
        raw_results = result.get("results", [])
        return [QueryResult.from_dict(res) for res in raw_results]

    async def _execute_query_builder_async(self, query_state: Dict[str, Any]) -> List[QueryResult]:
        """
        Execute a query builder request on the server asynchronously.

        Parameters
        ----------
        query_state : Dict[str, Any]
            The serialized query builder state

        Returns
        -------
        List[QueryResult]
            The query results
        """
        url = self._build_url(f"/api/v1/{self.name}/query_builder")
        response = await self._make_async_request_with_retry("POST", url, json=query_state)
        result = self._handle_response(response)

        # Process results
        raw_results = result.get("results", [])
        return [QueryResult.from_dict(res) for res in raw_results]

    def save(self):
        """Save the database (no-op for remote client)"""
        # No-op for remote client - server handles saving automatically
        pass

    def update_metadata_schema(
        self, new_schema: Union[str, Dict[str, Any]], drop_columns: bool = False, column_mapping: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Update the metadata schema for the remote database

        This method allows you to add new metadata fields, modify existing ones,
        or remove fields from the schema. Existing document data is preserved.

        Parameters
        ----------
        new_schema : Union[str, Dict[str, Any]]
            The new metadata schema to apply. Can be:
            - str: Schema name from common schemas (e.g., 'research_papers')
            - Dict[str, MetadataField]: Complete field definitions
            - Dict[str, str]: Simple type-only definitions (e.g., {'field': 'text'})
            - Dict[str, tuple]: Tuple definitions (type, indexed) or (type, indexed, required)
            - Dict[str, dict]: Full field configuration objects
        drop_columns : bool, default=False
            Whether to actually drop columns that are no longer in the schema.
            If False, columns are kept but removed from schema for safety.
        column_mapping : dict, optional
            Optionally provide a mapping dict with old-column (key) -> new-column (value)

        Returns
        -------
        Dict[str, Any]
            Summary of changes made including:
            - added_fields: List of newly added field names
            - removed_fields: List of removed field names
            - modified_fields: List of modified fields with change details
            - populated_defaults: List of fields where default values were populated
            - dropped_columns: List of actually dropped columns (if drop_columns=True)
            - warnings: List of warnings about potential issues
            - errors: List of any errors encountered

        Examples
        --------
        Add new metadata fields::

            new_schema = {
                'category': {'type': 'text', 'indexed': True, 'required': True, 'default_value': 'general'},
                'rating': {'type': 'real', 'default_value': 0.0},
                'tags': {'type': 'json', 'default_value': []}
            }

            changes = db.update_metadata_schema(new_schema)
            print(f"Added fields: {changes['added_fields']}")
            print(f"Populated defaults: {changes['populated_defaults']}")

        Use shorthand syntax::

            new_schema = {
                'category': 'text',  # Simple type
                'priority': ('integer', False, True),  # (type, indexed, required)
                'rating': ('real', True)  # (type, indexed)
            }

            changes = db.update_metadata_schema(new_schema)

        Apply a common schema::

            changes = db.update_metadata_schema('research_papers')

        Notes
        -----
        - Field names cannot conflict with reserved columns: id, content, content_hash, created_at, updated_at
        - Removed fields are removed from the schema but columns are kept for data safety
        - Type changes are recorded but don't modify existing data (SQLite limitation)
        - Index changes are applied immediately
        - Changes are applied in a transaction and rolled back on error
        """
        # Handle different input formats
        if isinstance(new_schema, str):
            # Send schema name to server
            schema_data = new_schema
        elif isinstance(new_schema, dict):
            # Convert to server-compatible format
            schema_data = {}
            for field_name, field_def in new_schema.items():
                if isinstance(field_def, str):
                    # Simple type string
                    schema_data[field_name] = field_def
                elif isinstance(field_def, tuple):
                    # Tuple format: (type, indexed) or (type, indexed, required)
                    if len(field_def) == 2:
                        field_type, indexed = field_def
                        schema_data[field_name] = {"type": field_type, "indexed": indexed}
                    elif len(field_def) == 3:
                        field_type, indexed, required = field_def
                        schema_data[field_name] = {"type": field_type, "indexed": indexed, "required": required}
                    else:
                        raise ValueError(f"Tuple definition for '{field_name}' must have 2 or 3 elements")
                elif hasattr(field_def, "type"):
                    # MetadataField object
                    schema_data[field_name] = {
                        "type": field_def.type.value if hasattr(field_def.type, "value") else str(field_def.type),
                        "indexed": field_def.indexed,
                        "required": field_def.required,
                        "default_value": field_def.default_value,
                    }
                elif isinstance(field_def, dict):
                    # Dictionary configuration
                    schema_data[field_name] = field_def
                else:
                    raise ValueError(f"Invalid field definition for '{field_name}': {type(field_def)}")
        else:
            raise ValueError("new_schema must be a string, dictionary, or MetadataField mapping")

        if column_mapping is not None and not isinstance(column_mapping, dict):
            raise ValueError(f"column_mapping must be None or a dict, found: {type(column_mapping)}")

        # Prepare request payload
        payload = {"metadata_schema": schema_data, "drop_columns": drop_columns, "column_mapping": column_mapping}

        url = self._build_url(f"/api/v1/{self.name}/schema")
        response = self._make_request_with_retry("PUT", url, json=payload)

        result = self._handle_response(response)

        # Update local metadata schema cache
        if "new_schema" in result:
            self._metadata_schema = {}
            for field_name, field_config in result["new_schema"].items():
                self._metadata_schema[field_name] = MetadataField(
                    type=MetadataFieldType(field_config["type"]),
                    indexed=field_config.get("indexed", False),
                    required=field_config.get("required", False),
                    default_value=field_config.get("default_value"),
                )

        return result.get("changes", {})

    def get_metadata_schema_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current metadata schema

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - fields: Dict of field definitions
            - field_count: Number of fields
            - indexed_fields: List of indexed field names
            - required_fields: List of required field names
            - field_types: Summary of field types used

        Examples
        --------
        Get schema information::

            schema_info = db.get_metadata_schema_info()
            print(f"Total fields: {schema_info['field_count']}")
            print(f"Indexed fields: {schema_info['indexed_fields']}")
            print(f"Required fields: {schema_info['required_fields']}")
            print(f"Field types: {schema_info['field_types']}")

            # Detailed field information
            for field_name, field_info in schema_info['fields'].items():
                print(f"{field_name}: {field_info['type']} "
                      f"(indexed={field_info['indexed']}, required={field_info['required']})")
        """
        url = self._build_url(f"/api/v1/{self.name}/schema")
        response = self._make_request_with_retry("GET", url)
        result = self._handle_response(response)
        return result.get("schema_info", {})

    @property
    def healthy(self) -> bool:
        """
        Check if the remote database server is healthy and accessible.

        This property performs a lightweight ping to the server to verify
        connectivity and server responsiveness. Results are cached for 60 seconds
        to avoid excessive network requests.

        Returns
        -------
        bool
            True if server is healthy and accessible, False otherwise
        """
        try:
            return self.ping()
        except Exception:
            # If ping fails due to network issues, connection errors, etc.
            return False

    @property
    def closed(self) -> bool:
        """
        Check if the connection to the remote database is closed.

        Note: This is a legacy property name. For checking server health,
        use the 'healthy' property instead.

        Returns
        -------
        bool
            False (remote connections don't have a traditional "closed" state)
        """
        # Remote HTTP connections don't have a persistent "closed" state
        # Return False to indicate connection is not closed
        return False

    def ping(self, force=False):
        now = time.time()
        if now - self._last_ping_timestamp < 60 and self._last_ping_status and not force:
            return self._last_ping_status

        url = self._build_url("/api/v1/databases")
        response = self._make_request_with_retry("GET", url)
        data = self._handle_response(response)
        databases = data.get("databases", [])

        self._last_ping_status = self.name in databases
        self._last_ping_timestamp = now
        return self._last_ping_status

    def close(self):
        """Close the database connection and HTTP clients"""
        # Close sync client if it exists
        if self._sync_client is not None and not self._sync_client.is_closed:
            self._sync_client.close()
            self._sync_client = None

        # Close async client if it exists (requires running from async context or event loop)
        if self._client is not None and not self._client.is_closed:
            try:
                # Try to get current event loop to close async client properly
                import asyncio

                loop = asyncio.get_running_loop()
                # If we're in an async context, schedule the close
                loop.create_task(self._client.aclose())
                self._client = None
            except RuntimeError:
                # No event loop running - create one briefly to close the client
                try:
                    asyncio.run(self._client.aclose())
                    self._client = None
                except Exception:
                    # If closing fails, at least clear the reference to prevent further use
                    self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_async()

    def hybrid_query(
        self, query_text: str, k: int = 10, vector_weight: float = 0.7, metadata_filters: dict = None, **kwargs
    ) -> List[QueryResult]:
        """Legacy method for backward compatibility - use query() instead"""
        return self.query(query_text, search_type="hybrid", k=k, filters=metadata_filters, vector_weight=vector_weight)

    def keyword_search(self, query_text: str, k: int = 10, metadata_filters: dict = None) -> List[QueryResult]:
        """Legacy method for backward compatibility - use query() instead"""
        return self.query(query_text, search_type="keyword", k=k, filters=metadata_filters)

    @classmethod
    def database_exists(
        cls,
        db_name: str,
        base_url: str = "http://127.0.0.1:5000",
        api_key: str = None,
        authorization_header="Authorization",
    ) -> bool:
        """Check if a database exists on the server"""
        url = f"{base_url}/api/v1/databases"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers[authorization_header] = f"Bearer {api_key}"

        try:
            with httpx.Client() as client:
                response = client.get(url, headers=headers)
        except httpx.ConnectError as e:
            raise DatabaseError(f"Could not connect to remote database server: {str(e)}") from e

        if response.status_code == 200:
            result = response.json()
            return db_name in result.get("databases", [])
        return False

    #################
    # Async Methods #
    #################

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is available"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.request_timeout) if self.request_timeout else None,
                limits=self._connection_pool_limits,
                headers=self._get_headers(),
            )
        return self._client

    async def _make_request_with_retry_async(self, method: str, url: str, **kwargs) -> httpx.Response | None:
        """Make HTTP request with exponential backoff retry"""
        client = await self._ensure_client()

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.request(method, url, **kwargs)

                # Don't retry on 4xx errors (client errors)
                if 400 <= response.status_code < 500:
                    return response

                # Success or 5xx error that we might retry
                if response.status_code < 500:
                    return response

                # 5xx error - might retry
                if attempt == self.max_retries:
                    return response  # Last attempt, return even if error

                # Wait before retry with exponential backoff
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Request failed with {response.status_code}, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
                await asyncio.sleep(delay)

            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e

                if attempt == self.max_retries:
                    raise ConnectionError(f"Failed to connect after {self.max_retries + 1} attempts: {e}") from e

                # Wait before retry
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Request failed with {type(e).__name__}: {e}, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        return None

    async def get_stats_async(self) -> Dict[str, Any]:
        """Get database statistics"""
        url = self._build_url(f"/api/v1/{self.name}/info")
        response = await self._make_request_with_retry_async("GET", url)
        db_info = self._handle_response(response)
        return db_info.get("stats", {})

    async def upsert_async(
        self,
        documents: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
        **kwargs,
    ) -> List[str]:
        """
        Insert or update documents in the database

        Parameters
        ----------
        documents : Union[str, List[str]]
            Document text(s) to add
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents
        ids : Optional[Union[str, List[str]]]
            Document IDs (auto-generated if not provided)
        batch_size : int
            Batch size for processing, by default 100
        similarity_threshold : float, optional
            If provided, skip chunks which have semantic similarity greater than this value (pre-deduplication)

        Returns
        -------
        List[str]
            List of document IDs that were upserted
        """
        # Handle single document case
        if isinstance(documents, str):
            documents = [documents]
            if isinstance(metadata, dict):
                metadata = [metadata]
            if isinstance(ids, str):
                ids = [ids]

        # Prepare request payload
        payload = {"documents": documents, "batch_size": batch_size}

        if metadata is not None:
            payload["metadata"] = metadata

        if ids is not None:
            payload["ids"] = ids

        if similarity_threshold is not None:
            payload["similarity_threshold"] = similarity_threshold

        url = self._build_url(f"/api/v1/{self.name}/documents")
        response = await self._make_request_with_retry_async("POST", url, json=payload)
        result = self._handle_response(response)

        return result.get("ids", [])

    async def insert_async(
        self,
        documents: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
        errors: Literal["ignore", "raise"] = "raise",
        **kwargs,
    ) -> List[str]:
        """
        Insert new documents into the database

        Parameters
        ----------
        documents : Union[str, List[str]]
            Document text(s) to add
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents
        ids : Optional[Union[str, List[str]]]
            Document IDs (auto-generated if not provided)
        batch_size : int
            Batch size for processing, by default 100
        errors : Literal["ignore", "raise"]
            How to handle document ID conflicts, by default "raise"
        similarity_threshold : Optional[float]
            Skip chunks that are too similar to existing chunks

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted
        """

        # Handle single document case
        if isinstance(documents, str):
            documents = [documents]
            if isinstance(metadata, dict):
                metadata = [metadata]
            if isinstance(ids, str):
                ids = [ids]

        # Prepare request payload
        payload = {"documents": documents, "batch_size": batch_size, "errors": errors}

        if metadata is not None:
            payload["metadata"] = metadata

        if ids is not None:
            payload["ids"] = ids

        if similarity_threshold is not None:
            payload["similarity_threshold"] = similarity_threshold

        url = self._build_url(f"/api/v1/{self.name}/documents/insert")
        response = await self._make_request_with_retry_async("POST", url, json=payload)
        result = self._handle_response(response)

        return result.get("ids", [])

    async def upsert_from_file_async(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
        extractor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Insert or update documents from files using file extraction (async).

        Parameters
        ----------
        file_paths : Union[str, Path, List[Union[str, Path]]]
            Path(s) to files to extract and upsert
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents. Will be merged with extracted metadata.
        ids : Optional[Union[str, List[str]]]
            Document IDs. If not provided, will use filename without extension.
        batch_size : int
            Batch size for processing, by default 100
        similarity_threshold : Optional[float]
            Skip chunks that are too similar to existing chunks
        extractor_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments passed to the extractor

        Returns
        -------
        List[str]
            List of document IDs that were upserted

        Raises
        ------
        FileNotFoundError
            If any of the specified files don't exist
        ValueError
            If extraction fails for any file
        """
        # Normalize file paths to list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]

        # Normalize other inputs
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]

        # Validate inputs
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")

        # Check files exist
        for file_path in file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

        # Prepare multipart form data
        url = self._build_url(f"/api/v1/{self.name}/upload")

        # Build form data
        form_data = {"batch_size": str(batch_size), "mode": "upsert"}  # Specify upsert mode

        if metadata is not None:
            form_data["metadata"] = json.dumps(metadata)

        if ids is not None:
            form_data["ids"] = json.dumps(ids)

        if similarity_threshold is not None:
            form_data["similarity_threshold"] = str(similarity_threshold)

        if extractor_kwargs:
            form_data["extractor_kwargs"] = json.dumps(extractor_kwargs)

        # Prepare files for streaming upload
        files = []
        file_handles = []
        result = None
        try:
            for file_path in file_paths:
                file_handle = open(file_path, "rb")
                file_handles.append(file_handle)
                files.append(("files", (file_path.name, file_handle, "application/octet-stream")))

            # Make request with streaming files
            response = await self._make_request_with_retry_async("POST", url, data=form_data, files=files)
            result = self._handle_response(response)
        finally:
            # Ensure all file handles are closed
            for file_handle in file_handles:
                try:
                    file_handle.close()
                except Exception:
                    pass  # Ignore close errors

        if result is not None:
            return result.get("document_ids", [])
        return []

    async def insert_from_file_async(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
        errors: Literal["ignore", "raise"] = "raise",
        extractor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Insert new documents from files using file extraction (async).

        Parameters
        ----------
        file_paths : Union[str, Path, List[Union[str, Path]]]
            Path(s) to files to extract and insert
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents. Will be merged with extracted metadata.
        ids : Optional[Union[str, List[str]]]
            Document IDs. If not provided, will use filename without extension.
        batch_size : int
            Batch size for processing, by default 100
        similarity_threshold : Optional[float]
            Skip chunks that are too similar to existing chunks
        errors : Literal["ignore", "raise"]
            How to handle document ID conflicts, by default "raise"
        extractor_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments passed to the extractor

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted

        Raises
        ------
        FileNotFoundError
            If any of the specified files don't exist
        ValueError
            If extraction fails for any file
        DuplicateDocumentIDError
            If errors="raise" and document ID conflicts occur
        """
        # Normalize file paths to list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]

        # Normalize other inputs
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]

        # Validate inputs
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")

        # Check files exist
        for file_path in file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

        # Prepare multipart form data
        url = self._build_url(f"/api/v1/{self.name}/upload")

        # Build form data
        form_data = {"batch_size": str(batch_size), "mode": "insert", "errors": errors}  # Specify insert mode

        if metadata is not None:
            form_data["metadata"] = json.dumps(metadata)

        if ids is not None:
            form_data["ids"] = json.dumps(ids)

        if similarity_threshold is not None:
            form_data["similarity_threshold"] = str(similarity_threshold)

        if extractor_kwargs:
            form_data["extractor_kwargs"] = json.dumps(extractor_kwargs)

        # Prepare files for streaming upload
        files = []
        file_handles = []
        try:
            for file_path in file_paths:
                file_handle = open(file_path, "rb")
                file_handles.append(file_handle)
                files.append(("files", (file_path.name, file_handle, "application/octet-stream")))

            # Make request with streaming files
            response = await self._make_request_with_retry_async("POST", url, data=form_data, files=files)
            result = self._handle_response(response)
        finally:
            # Ensure all file handles are closed
            for file_handle in file_handles:
                try:
                    file_handle.close()
                except Exception:
                    pass  # Ignore close errors

        return result.get("document_ids", [])

    async def upsert_from_chunks_async(
        self,
        chunks_by_document: Dict[str, Union[List["Chunk"], List[str]]],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
        **kwargs,
    ) -> List[str]:
        """
        Upsert documents from pre-chunked data (async).

        Parameters
        ----------
        chunks_by_document : Dict[str, Union[List[Chunk], List[str]]]
            Dictionary mapping document IDs to their chunks. Chunks can be either:
            - List[Chunk]: Full Chunk objects with position information
            - List[str]: Simple strings that will be converted to Chunk objects
        metadata : Optional[Dict[str, Dict[str, Any]]], default=None
            Dictionary mapping document IDs to their metadata
        batch_size : int, default=100
            Number of embeddings to generate at once
        similarity_threshold : Optional[float], default=None
            If provided, filters out chunks that are too similar to existing chunks

        Returns
        -------
        List[str]
            List of document IDs that were upserted
        """
        # Convert Chunk objects to serializable format
        serializable_chunks = {}
        for doc_id, chunks in chunks_by_document.items():
            if chunks and hasattr(chunks[0], "__dict__"):
                # Convert Chunk objects to dicts
                serializable_chunks[doc_id] = [
                    (
                        {
                            "text": chunk.content,
                            "position": chunk.position.to_dict(),
                            "total_chunks": getattr(chunk, "total_chunks", len(chunks)),
                            "metadata": getattr(chunk, "metadata", {}),
                        }
                        if hasattr(chunk, "content")
                        else str(chunk)
                    )
                    for chunk in chunks
                ]
            else:
                # Already strings or serializable
                serializable_chunks[doc_id] = chunks

        # Prepare request payload
        payload = {"chunks_by_document": serializable_chunks, "batch_size": batch_size}

        if metadata is not None:
            payload["metadata"] = metadata

        if similarity_threshold is not None:
            payload["similarity_threshold"] = similarity_threshold

        url = self._build_url(f"/api/v1/{self.name}/documents/chunks")
        response = await self._make_request_with_retry_async("POST", url, json=payload)
        result = self._handle_response(response)

        return result.get("ids", [])

    async def insert_from_chunks_async(
        self,
        chunks_by_document: Dict[str, Union[List["Chunk"], List[str]]],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: int = 100,
        similarity_threshold: Optional[float] = None,
        errors: Literal["ignore", "raise"] = "raise",
        **kwargs,
    ) -> List[str]:
        """
        Insert documents from pre-chunked data with conflict handling (async).

        Parameters
        ----------
        chunks_by_document : Dict[str, Union[List[Chunk], List[str]]]
            Dictionary mapping document IDs to their chunks
        metadata : Optional[Dict[str, Dict[str, Any]]], default=None
            Dictionary mapping document IDs to their metadata
        batch_size : int, default=100
            Number of embeddings to generate at once
        similarity_threshold : Optional[float], default=None
            If provided, filters out chunks that are too similar to existing chunks
        errors : Literal["ignore", "raise"], default="raise"
            How to handle document ID conflicts

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted

        Raises
        ------
        DuplicateDocumentIDError
            If a document ID already exists and errors="raise"
        """
        # Convert Chunk objects to serializable format
        serializable_chunks = {}
        for doc_id, chunks in chunks_by_document.items():
            if chunks and hasattr(chunks[0], "__dict__"):
                # Convert Chunk objects to dicts
                serializable_chunks[doc_id] = [
                    (
                        {
                            "text": chunk.content,
                            "position": chunk.position.to_dict(),
                            "total_chunks": getattr(chunk, "total_chunks", len(chunks)),
                            "metadata": getattr(chunk, "metadata", {}),
                        }
                        if hasattr(chunk, "content")
                        else str(chunk)
                    )
                    for chunk in chunks
                ]
            else:
                # Already strings or serializable
                serializable_chunks[doc_id] = chunks

        # Prepare request payload
        payload = {"chunks_by_document": serializable_chunks, "batch_size": batch_size, "errors": errors}

        if metadata is not None:
            payload["metadata"] = metadata

        if similarity_threshold is not None:
            payload["similarity_threshold"] = similarity_threshold

        url = self._build_url(f"/api/v1/{self.name}/documents/chunks/insert")
        response = await self._make_request_with_retry_async("POST", url, json=payload)
        result = self._handle_response(response)

        return result.get("ids", [])

    async def get_async(self, ids: Union[str, List[str]]) -> Union[Document, List[Document], None]:
        """
        Retrieve documents by ID

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to retrieve

        Returns
        -------
        Union[Document, List[Document], None]
            Retrieved document(s) or None if not found
        """

        single_id = isinstance(ids, str)
        if single_id:
            url = self._build_url(f"/api/v1/{self.name}/documents/{ids}")
        else:
            url = self._build_url(f"/api/v1/{self.name}/documents?ids={','.join(ids)}")

        response = await self._make_request_with_retry_async("GET", url)
        result = self._handle_response(response)
        if single_id:
            return Document.from_dict(result)
        else:
            return [Document.from_dict(doc) for doc in result["documents"]]

    async def exists_async(self, ids: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """
        Check if documents exist

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to check

        Returns
        -------
        Union[bool, List[bool]]
            Existence status for each ID
        """
        single_id = isinstance(ids, str)
        check_ids = [ids] if single_id else ids

        url = self._build_url(f"/api/v1/{self.name}/documents/exists")
        response = await self._make_request_with_retry_async("POST", url, json={"ids": check_ids})
        results = self._handle_response(response)

        return results.get("exists")[0] if single_id else results.get("exists")

    async def delete_async(self, ids: Union[str, List[str]]) -> int:
        """
        Delete documents

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to delete

        Returns
        -------
        int
            Number of documents deleted
        """

        if isinstance(ids, str):
            ids = [ids]

        # Use batch endpoint for multiple IDs (threshold: 2+ IDs)
        if len(ids) >= 2:
            url = self._build_url(f"/api/v1/{self.name}/documents/delete")
            payload = {"ids": ids}
            response = await self._make_request_with_retry_async("POST", url, json=payload)
            result = self._handle_response(response)
            return result.get("deleted_count", 0)

        # Single ID: use original DELETE endpoint (keeping async concurrency for very small batches)
        deleted_count = 0

        async def delete_single(doc_id: str) -> int:
            url = self._build_url(f"/api/v1/{self.name}/documents/{doc_id}")
            response = await self._make_request_with_retry_async("DELETE", url)
            result = self._handle_response(response)
            return result.get("deleted_count", 0)

        tasks = [delete_single(doc_id) for doc_id in ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, int):
                deleted_count += result
            elif isinstance(result, Exception):
                logger.warning(f"Failed to delete document: {result}")

        return deleted_count

    async def count_async(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents matching filters asynchronously.

        Parameters
        ----------
        filters : Dict[str, Any], optional
            Metadata filters to apply

        Returns
        -------
        int
            Number of matching documents
        """
        url = self._build_url(f"/api/v1/{self.name}/documents/count")

        # Prepare payload
        payload = {}
        if filters is not None:
            payload["filters"] = filters

        response = await self._make_request_with_retry_async("POST", url, json=payload)
        result = self._handle_response(response)

        return result.get("count", 0)

    async def update_async(
        self, doc_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document's content and/or metadata

        Parameters
        ----------
        doc_id : str
            Document ID to update
        content : Optional[str]
            New content (if None, content is not updated)
        metadata : Optional[Dict[str, Any]]
            New metadata (merged with existing)

        Returns
        -------
        bool
            True if document was updated, False if not found
        """

        if not content and not metadata:
            return False

        payload = {}
        if content is not None:
            payload["content"] = content
        if metadata is not None:
            payload["metadata"] = metadata

        url = self._build_url(f"/api/v1/{self.name}/documents/{doc_id}")

        try:
            response = await self._make_request_with_retry_async("PUT", url, json=payload)
            result = self._handle_response(response)
            return result.get("updated", False)
        except DatabaseNotFoundError:
            return False

    async def query_async(
        self,
        query: str,
        *,
        search_type: Literal["vector", "keyword", "hybrid"] = "vector",
        return_type: Literal["documents", "chunks", "context"] = "documents",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        context_window: int = 2,
        semantic_dedup_threshold: Optional[float] = None,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: dict = None,
    ) -> List[QueryResult]:
        """
        Unified query interface for all search types

        Parameters
        ----------
        query : str
            Query text
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Literal['documents', 'chunks']
            Whether to return full documents or individual chunks
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score threshold (0-1, higher=better)
        filters : Optional[Dict[str, Any]]
            Metadata filters
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)
        context_window : int
            Number of chunks before and after to include when return_type='context'
        semantic_dedup_threshold : Optional[float]
            Similarity threshold for semantic deduplication (0-1, higher=more similar)
        document_scoring_method : str
            Method for aggregating chunk scores into document scores
        document_scoring_options : dict
            Optional parameters specific to each scoring method

        Returns
        -------
        List[QueryResult]
            Search results with normalized scores
        """

        # Prepare request payload
        payload = {
            "query": query,
            "search_type": search_type,
            "return_type": return_type,
            "k": k,
            "score_threshold": score_threshold,
            "vector_weight": vector_weight,
            "context_window": context_window,
            "semantic_dedup_threshold": semantic_dedup_threshold,
            "document_scoring_method": document_scoring_method,
            "document_scoring_options": document_scoring_options,
        }

        if filters is not None:
            payload["filters"] = filters

        url = self._build_url(f"/api/v1/{self.name}/query")
        response = await self._make_request_with_retry_async("POST", url, json=payload)
        result = self._handle_response(response)

        # Process results
        raw_results = result.get("results", [])
        return [QueryResult.from_dict(res) for res in raw_results]

    async def query_multi_column_async(
        self,
        query: str,
        *,
        columns: Optional[List[str]] = None,
        search_type: Literal["vector", "keyword", "hybrid"] = "vector",
        return_type: Literal["documents", "chunks"] = "documents",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: dict = None,
    ) -> List[QueryResult]:
        """
        Async query across multiple columns (main content + embedding-enabled metadata fields)

        Parameters
        ----------
        query : str
            Query text
        columns : Optional[List[str]]
            Specific columns to search. If None, searches all embedding-enabled fields
            plus main content. Use 'content' for main document content.
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Literal['documents', 'chunks']
            Whether to return full documents or individual chunks
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score threshold (0-1, higher=better)
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)
        document_scoring_method : DocumentScoringMethod
            Method for aggregating chunk scores into document scores
        document_scoring_options : dict, optional
            Parameters for the scoring method

        Returns
        -------
        List[QueryResult]
            Search results with column attribution
        """
        # Prepare request payload
        payload = {
            "query": query,
            "search_type": search_type,
            "return_type": return_type,
            "k": k,
            "score_threshold": score_threshold,
            "vector_weight": vector_weight,
            "document_scoring_method": document_scoring_method,
            "document_scoring_options": document_scoring_options,
        }

        if columns is not None:
            payload["columns"] = columns

        if filters is not None:
            payload["filters"] = filters

        url = self._build_url(f"/api/v1/{self.name}/query-multi-column")
        response = await self._make_request_with_retry_async("POST", url, json=payload)
        result = self._handle_response(response)

        # Process results
        raw_results = result.get("results", [])
        return [QueryResult.from_dict(res) for res in raw_results]

    async def filter_async(
        self,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Document]:
        """
        Filter documents using enhanced metadata filtering

        Parameters
        ----------
        where : Optional[Dict[str, Any]]
            Filter conditions using either simple format or MongoDB-style operators
        order_by : Optional[str]
            ORDER BY clause (field name with optional ASC/DESC)
        limit : Optional[int]
            Maximum number of results
        offset : int
            Number of results to skip

        Returns
        -------
        List[Document]
            Filtered documents
        """

        # Prepare request payload
        payload = {"where": where, "offset": offset}

        if order_by is not None:
            payload["order_by"] = order_by

        if limit is not None:
            payload["limit"] = limit

        url = self._build_url(f"/api/v1/{self.name}/filter")
        response = await self._make_request_with_retry_async("POST", url, json=payload)
        result = self._handle_response(response)

        # Process results
        raw_docs = result.get("documents", [])
        return [Document.from_dict(doc) for doc in raw_docs]

    async def save_async(self):
        """No-op for remote databases"""
        pass

    async def close_async(self):
        """Close HTTP clients"""
        # Close sync client if it exists
        if self._sync_client is not None and not self._sync_client.is_closed:
            self._sync_client.close()
            self._sync_client = None

        # Close async client if it exists
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def update_metadata_schema_async(
        self, new_schema: Union[str, Dict[str, Any]], drop_columns: bool = False, column_mapping: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Update the metadata schema for the database asynchronously

        This method allows you to add new metadata fields, modify existing ones,
        or remove fields from the schema. Existing document data is preserved.

        Parameters
        ----------
        new_schema : Union[str, Dict[str, Any]]
            The new metadata schema to apply. Can be:
            - str: Schema name from common schemas (e.g., 'research_papers')
            - Dict with field definitions
        drop_columns : bool, default=False
            Whether to actually drop columns that are no longer in the schema.
            If False, columns are kept but removed from schema for safety.
        column_mapping : dict, optional
            Optionally provide a mapping dict with old-column (key) -> new-column (value)

        Returns
        -------
        Dict[str, Any]
            Summary of changes made including:
            - added_fields: List of newly added field names
            - removed_fields: List of removed field names
            - modified_fields: List of modified fields with change details
            - populated_defaults: List of fields where default values were populated
            - dropped_columns: List of actually dropped columns (if drop_columns=True)
            - warnings: List of warnings about potential issues
            - errors: List of any errors encountered

        Examples
        --------
        Add new metadata fields::

            new_schema = {
                'category': {
                    'type': 'text',
                    'indexed': True
                },
                'priority': {
                    'type': 'integer',
                    'default_value': 0
                }
            }

            changes = await db.update_metadata_schema_async(new_schema)
            print(f"Added fields: {changes['added_fields']}")

        Apply a common schema::

            changes = await db.update_metadata_schema_async('research_papers')
        """
        # Handle different input formats
        if isinstance(new_schema, str):
            # Send schema name to server
            schema_data = new_schema
        elif isinstance(new_schema, dict):
            # Convert to server-compatible format
            schema_data = {}
            for field_name, field_def in new_schema.items():
                if isinstance(field_def, str):
                    # Simple type string
                    schema_data[field_name] = field_def
                elif isinstance(field_def, tuple):
                    # Tuple format: (type, indexed) or (type, indexed, required)
                    if len(field_def) == 2:
                        field_type, indexed = field_def
                        schema_data[field_name] = {"type": field_type, "indexed": indexed}
                    elif len(field_def) == 3:
                        field_type, indexed, required = field_def
                        schema_data[field_name] = {"type": field_type, "indexed": indexed, "required": required}
                    else:
                        raise ValueError(f"Tuple definition for '{field_name}' must have 2 or 3 elements")
                elif hasattr(field_def, "type"):
                    # MetadataField object
                    schema_data[field_name] = {
                        "type": field_def.type.value if hasattr(field_def.type, "value") else str(field_def.type),
                        "indexed": field_def.indexed,
                        "required": getattr(field_def, "required", False),
                        "default_value": getattr(field_def, "default_value", None),
                    }
                elif isinstance(field_def, dict):
                    # Already in dict format
                    schema_data[field_name] = field_def
                else:
                    raise ValueError(f"Invalid field definition for '{field_name}': {type(field_def)}")
        else:
            raise ValueError("new_schema must be a string (schema name) or dict")

        url = self._build_url(f"/api/v1/{self.name}/update_schema")
        payload = {"new_schema": schema_data, "drop_columns": drop_columns, "column_mapping": column_mapping or {}}

        response = await self._make_request_with_retry_async("POST", url, json=payload)
        result = self._handle_response(response)

        # Update local metadata schema cache
        if "new_schema" in result:
            self._metadata_schema = {}
            for field_name, field_config in result["new_schema"].items():
                self._metadata_schema[field_name] = MetadataField(
                    type=MetadataFieldType(field_config["type"]),
                    indexed=field_config.get("indexed", False),
                    required=field_config.get("required", False),
                    default_value=field_config.get("default_value"),
                )

        return result.get("changes", {})

    async def get_metadata_schema_info_async(self) -> Dict[str, Any]:
        """
        Get detailed information about the current metadata schema asynchronously

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - fields: Dict of field definitions
            - field_count: Number of fields
            - indexed_fields: List of indexed field names
            - required_fields: List of required field names
            - field_types: Summary of field types used

        Examples
        --------
        Get schema information::

            schema_info = await db.get_metadata_schema_info_async()
            print(f"Total fields: {schema_info['field_count']}")
            print(f"Indexed fields: {schema_info['indexed_fields']}")
            print(f"Required fields: {schema_info['required_fields']}")
            print(f"Field types: {schema_info['field_types']}")

            # Detailed field information
            for field_name, field_info in schema_info['fields'].items():
                print(f"{field_name}: {field_info['type']} "
                      f"(indexed={field_info['indexed']}, required={field_info['required']})")
        """
        url = self._build_url(f"/api/v1/{self.name}/schema")
        response = await self._make_request_with_retry_async("GET", url)
        result = self._handle_response(response)
        return result.get("schema_info", {})

    ## Tuning
    def get_sqlite_tuning(self) -> Dict[str, Any]:
        """Get current SQLite tuning configuration from remote server."""
        response = self._make_request_with_retry("GET", f"/api/v1/{self.name}/tuning")
        return self._handle_response(response)

    def set_sqlite_tuning(self, profile: str, overrides: Optional[Dict[str, Any]] = None, persist: bool = True) -> None:
        """Apply SQLite tuning profile via remote server."""
        payload = {"profile": profile, "overrides": overrides or {}, "persist": persist}

        response = self._make_request_with_retry("PUT", f"/api/v1/{self.name}/tuning", json=payload)
        self._handle_response(response)

    def sqlite_checkpoint(self, mode: str = "PASSIVE") -> None:
        """Run SQLite WAL checkpoint via remote server."""
        payload = {"mode": mode}
        response = self._make_request_with_retry("POST", f"/api/v1/{self.name}/maintenance/checkpoint", json=payload)
        self._handle_response(response)

    def sqlite_optimize(self) -> None:
        """Run SQLite PRAGMA optimize via remote server."""
        response = self._make_request_with_retry("POST", f"/api/v1/{self.name}/maintenance/optimize")
        self._handle_response(response)

    def sqlite_vacuum(self) -> None:
        """Run SQLite VACUUM via remote server."""
        response = self._make_request_with_retry("POST", f"/api/v1/{self.name}/maintenance/vacuum")
        self._handle_response(response)

    def sqlite_incremental_vacuum(self, pages: int = 2000) -> None:
        """Run incremental VACUUM via remote server."""
        payload = {"pages": pages}
        response = self._make_request_with_retry(
            "POST", f"/api/v1/{self.name}/maintenance/incremental_vacuum", json=payload
        )
        self._handle_response(response)

    def analyze_system_resources(self) -> Dict[str, Any]:
        """Analyze remote server system resources."""
        response = self._make_request_with_retry("GET", "/api/system/resources")
        return self._handle_response(response)

    def checkpoint_if_wal_large(self, wal_mb_threshold: int = 128) -> bool:
        """Check if remote WAL is large and checkpoint if needed."""
        payload = {"threshold_mb": wal_mb_threshold}
        response = self._make_request_with_retry(
            "POST", f"/api/v1/{self.name}/maintenance/checkpoint_if_large", json=payload
        )
        data = self._handle_response(response)
        return data.get("checkpointed", False)
