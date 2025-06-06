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
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Literal, List, Optional, Dict, Any

import httpx
import numpy as np

from localvectordb.core import MetadataField, MetadataFieldType, QueryResult, Document, BaseVectorDB
from localvectordb.embeddings import EmbeddingProvider
from localvectordb.exceptions import (
    DatabaseNotFoundError, DuplicateDocumentIDError, EmbeddingError, BaseLocalVectorDBException, DatabaseError
)

logger = logging.getLogger(__name__)


class RemoteEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider that proxies requests to a LocalVectorDB server.

    This provider mimics the interface of local embedding providers but makes
    HTTP requests to the server's embedding endpoint. This allows RemoteVectorDB
    to seamlessly support semantic filtering and other embedding-dependent features.

    Parameters
    ----------
    db_name : str
        Name of the database on the server
    base_url : str
        Base URL of the LocalVectorDB server
    api_key : Optional[str]
        API key for authentication
    request_timeout : Optional[int]
        Timeout for HTTP requests in seconds
    authorization_header : str
        Authorization header name, by default "Authorization"

    Examples
    --------
    Direct usage (typically handled automatically by RemoteVectorDB)::

        provider = RemoteEmbeddingProvider(
            db_name="research_papers",
            base_url="http://localhost:5000",
            api_key="your_api_key"
        )

        embeddings = provider.embed_sync(["text to embed"])
        dimension = provider.get_dimension()
    """

    def __init__(
            self,
            db_name: str,
            base_url: str,
            api_key: Optional[str] = None,
            request_timeout: Optional[int] = None,
            authorization_header: str = "Authorization",
            model: str = "remote",  # Required by base class
            **kwargs
    ):
        # Initialize base class with dummy model name
        super().__init__(model, **kwargs)

        self.db_name = db_name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.request_timeout = request_timeout or 300
        self.authorization_header = authorization_header

        # Cache for database info to avoid repeated requests
        self._db_info_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 300  # 5 minutes cache TTL

        # Derived info from database
        self._provider_name: Optional[str] = None
        self._model_name: Optional[str] = None
        self._dimension: Optional[int] = None
        self._validated: bool = False

        # Load initial database info
        self._ensure_db_info()

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers including authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.authorization_header] = f"Bearer {self.api_key}"
        return headers

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and raise appropriate exceptions."""
        if response.status_code == 200:
            return response.json()

        try:
            error_data = response.json()
            error_msg = error_data.get("error", str(response.status_code))
        except Exception:
            error_msg = f"HTTP {response.status_code}: {response.text}"

        if response.status_code == 404:
            raise RuntimeError(f"Database '{self.db_name}' not found on server")
        elif response.status_code == 401:
            raise RuntimeError("Authentication failed. Check your API key.")
        else:
            raise RuntimeError(f"Server error: {error_msg}")

    def _ensure_db_info(self, force_refresh: bool = False) -> None:
        """Ensure database info is loaded and cached."""
        current_time = time.time()

        if (not force_refresh and
                self._db_info_cache and
                current_time - self._cache_timestamp < self._cache_ttl):
            return

        try:
            self._load_db_info()
            self._cache_timestamp = current_time
        except Exception as e:
            logger.error(f"Failed to load database info for {self.db_name}: {e}")
            if not self._db_info_cache:
                # If we have no cached info at all, re-raise the error
                raise RuntimeError(f"Cannot connect to database '{self.db_name}': {e}")
            # Otherwise, use stale cache and log warning
            logger.warning(f"Using stale cache for database {self.db_name} due to error: {e}")

    def _load_db_info(self) -> None:
        """Load database information from server."""
        url = f"{self.base_url}/api/v1/{self.db_name}/info"

        with httpx.Client() as client:
            response = client.get(
                url,
                headers=self._get_headers(),
                timeout=self.request_timeout
            )

        db_info = self._handle_response(response)
        self._db_info_cache = db_info

        # Extract relevant configuration
        config = db_info.get("config", {})
        self._provider_name = config.get("embedding_provider", "unknown")
        self._model_name = config.get("embedding_model", "unknown")
        self._dimension = config.get("embedding_dimension", 0)

        if self._dimension <= 0:
            raise RuntimeError(f"Invalid embedding dimension: {self._dimension}")

        self._validated = True
        logger.debug(f"Loaded database info: provider={self._provider_name}, "
                     f"model={self._model_name}, dimension={self._dimension}")


    @property
    def provider_name(self) -> str:
        """Return the underlying embedding provider name."""
        self._ensure_db_info()
        return f"remote-{self._provider_name}"

    @property
    def model(self) -> str:
        """Return the underlying embedding model name."""
        self._ensure_db_info()
        return self._model_name

    @property
    def max_batch_size(self) -> int:
        """Maximum batch size for this provider."""
        # Conservative default for remote provider to avoid timeouts
        return 100

    def validate_model(self) -> bool:
        """Check if the remote model is available."""
        try:
            self._ensure_db_info()
            return self._validated
        except Exception:
            return False

    def get_dimension(self) -> int:
        """Get embedding dimension from the remote database."""
        self._ensure_db_info()
        return self._dimension

    async def embed_batch(
            self,
            texts: List[str],
            batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Generate embeddings using the remote server."""
        if not texts:
            return np.array([]).reshape(0, self.get_dimension())

        batch_size = batch_size or self.max_batch_size
        all_embeddings = []

        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                try:
                    embeddings = await self._embed_batch_request(client, batch)
                    all_embeddings.extend(embeddings)

                except Exception as e:
                    logger.error(f"Error getting embeddings for batch {i // batch_size + 1}: {e}")
                    raise RuntimeError(f"Failed to get embeddings from server: {e}")

        return np.array(all_embeddings, dtype=np.float32)

    async def _embed_batch_request(
            self,
            client: httpx.AsyncClient,
            texts: List[str]
    ) -> List[List[float]]:
        """Make a single embedding request to the server."""
        url = f"{self.base_url}/api/v1/{self.db_name}/embeddings"

        payload = {"texts": texts}

        response = await client.post(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=self.request_timeout
        )

        result = self._handle_response(response)
        embeddings = result.get("embeddings", [])

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Server returned {len(embeddings)} embeddings for {len(texts)} texts"
            )

        return embeddings

    def embed_sync(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Synchronous wrapper for embed_batch."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, run in thread pool
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.embed_batch(texts, batch_size))
                    return future.result()
            else:
                return loop.run_until_complete(self.embed_batch(texts, batch_size))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.embed_batch(texts, batch_size))

    def get_config_info(self) -> Dict[str, Any]:
        """Get detailed configuration information about the remote embedding setup."""
        self._ensure_db_info()
        return {
            "provider_name": self._provider_name,
            "model_name": self._model_name,
            "dimension": self._dimension,
            "remote_database": self.db_name,
            "server_url": self.base_url,
            "max_batch_size": self.max_batch_size,
            "cache_ttl": self._cache_ttl,
            "last_cache_update": self._cache_timestamp
        }

    def refresh_cache(self) -> None:
        """Force refresh of the database info cache."""
        self._ensure_db_info(force_refresh=True)

    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to the remote server and return diagnostic info."""
        start_time = time.time()

        try:
            # Test basic connectivity
            self._ensure_db_info(force_refresh=True)

            # Test embedding functionality with a simple request
            test_embeddings = self.embed_sync(["test connection"])

            end_time = time.time()

            return {
                "status": "success",
                "response_time_seconds": end_time - start_time,
                "server_reachable": True,
                "authentication_valid": True,
                "embedding_functional": True,
                "test_embedding_dimension": len(test_embeddings[0]) if len(test_embeddings) > 0 else 0,
                "server_info": {
                    "provider": self._provider_name,
                    "model": self._model_name,
                    "dimension": self._dimension
                }
            }

        except Exception as e:
            end_time = time.time()
            return {
                "status": "error",
                "response_time_seconds": end_time - start_time,
                "error": str(e),
                "server_reachable": "unknown",
                "authentication_valid": "unknown",
                "embedding_functional": False
            }

    def __repr__(self) -> str:
        """String representation of the provider."""
        status = "connected" if self._validated else "disconnected"
        return (f"RemoteEmbeddingProvider(db='{self.db_name}', "
                f"server='{self.base_url}', status='{status}')")


class RemoteVectorDB(BaseVectorDB):
    """Client for interacting with a LocalVectorDB v1.0 server.

    This client provides the same document-focused interface as LocalVectorDB v1.0
    but connects to a remote server via HTTP.

    Parameters
    ----------
    name : str
        Name of the database
    base_url : str
        URL of the LocalVectorDB server (e.g., "http://localhost:5000")
    api_key : str, optional
        API key for authentication
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
            request_timeout: int = None,
            authorization_header: str = "Authorization",
            max_retries: int = 3,
            retry_delay: float = 1.0,
            connection_pool_limits: Optional[httpx.Limits] = None
    ):
        self.name = name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connection_pool_limits = connection_pool_limits or httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0
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
        self._authorization_header = authorization_header

        self._last_ping_timestamp = 0
        self._last_ping_status = False

        # State variables to be loaded from server
        self._embedding_dimension = 0

        self._remote_embedding_provider = RemoteEmbeddingProvider(
            db_name=self.name,
            base_url=self.base_url,
            api_key=self.api_key,
            request_timeout=self.request_timeout,
            authorization_header=self._authorization_header
        )

        # Check if database exists and create if needed
        if create_if_not_exists:
            self._ensure_database_exists()
        else:
            # Load existing database info
            self._load_database_info()

    def _make_request_with_retry(
            self,
            method: str,
            url: str,
            **kwargs
    ) -> httpx.Response | None:
        """Make HTTP request with exponential backoff retry"""

        last_exception = None

        # Configure httpx client with connection pooling
        client_kwargs = {
            'timeout': httpx.Timeout(self.request_timeout or 30.0),
            'limits': self._connection_pool_limits,
            'headers': self._get_headers()
        }

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(**client_kwargs) as client:
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
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Request failed with {response.status_code}, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    time.sleep(delay)

            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e

                if attempt == self.max_retries:
                    raise ConnectionError(f"Failed to connect after {self.max_retries + 1} attempts: {e}")

                # Wait before retry
                delay = self.retry_delay * (2 ** attempt)
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

    def _handle_response(self, response: httpx.Response) -> dict:
        """Handle API response and raise appropriate exceptions"""
        if response.status_code == 200:
            return response.json()

        try:
            error_data = response.json()
            error_type = error_data.get("type", "unknown")
            error_msg = error_data.get("error", str(response.status_code))

            # Map error type to appropriate exception
            error_map = {
                "database_not_found": DatabaseNotFoundError,
                "duplicate_document_id": DuplicateDocumentIDError,
                "embedding_error": EmbeddingError,
            }

            # Raise the appropriate exception if we recognize the type
            if error_type in error_map:
                raise error_map[error_type](error_msg)

        except (ValueError, KeyError):
            # Fallback if we can't parse the response
            error_msg = response.text or f"HTTP Error: {response.status_code}"

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
            self.metadata_schema[field_name] = MetadataField(
                type=MetadataFieldType(field_config["type"]),
                indexed=field_config.get("indexed", False),
                required=field_config.get("required", False),
                default_value=field_config.get("default_value")
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
                    "default_value": field_def.default_value
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
            "enable_fts": self._enable_fts
        }

        response = self._make_request_with_retry("POST", url, json=payload)

        created_db_info = self._handle_response(response)
        config = created_db_info.get("config", {})
        self._embedding_dimension = config.get("embedding_dimension", 0)

    @property
    def embedding_provider(self) -> RemoteEmbeddingProvider:
        """Return the remote embedding provider instance."""
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
            similarity_threshold: Optional[float] = None
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
        payload = {
            "documents": documents,
            "batch_size": batch_size
        }

        if metadata is not None:
            payload["metadata"] = metadata

        if ids is not None:
            payload["ids"] = ids

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
            errors: Literal["ignore", "raise"] = "raise",
            similarity_threshold: Optional[float] = None
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
        payload = {
            "documents": documents,
            "batch_size": batch_size,
            "errors": errors
        }

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

    def get(self, ids: Union[str, List[str]]) -> Union[Document, List[Document], None]:
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

            response = self._make_request_with_retry("GET", url)

            try:
                result = self._handle_response(response)
                doc = Document.from_dict(result)
                return doc
            except DatabaseNotFoundError:
                return None
        else:
            # Handle multiple IDs - make individual requests for each ID
            # TODO: make server handle multiple!
            docs = []
            for doc_id in ids:
                doc = self.get(doc_id)
                if doc is not None:
                    docs.append(doc)
            return docs

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

        deleted_count = 0
        for doc_id in ids:
            url = self._build_url(f"/api/v1/{self.name}/documents/{doc_id}")
            response = self._make_request_with_retry("DELETE", url)
            result = self._handle_response(response)
            deleted_count += result.get("deleted_count", 0)

        return deleted_count

    def update(
            self,
            doc_id: str,
            content: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
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
            search_type: Literal['vector', 'keyword', 'hybrid'] = 'vector',
            return_type: Literal['documents', 'chunks'] = 'documents',
            k: int = 10,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
            vector_weight: float = 0.7,  # For hybrid search
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
            "vector_weight": vector_weight
        }

        if filters is not None:
            payload["filters"] = filters

        url = self._build_url(f"/api/v1/{self.name}/query")
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
            offset: int = 0
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
        payload = {
            "where": where,
            "offset": offset
        }

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

    def save(self):
        """Save the database (no-op for remote client)"""
        # No-op for remote client - server handles saving automatically
        pass

    def update_metadata_schema(
            self,
            new_schema: Union[str, Dict[str, Any]],
            drop_columns: bool = False
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
                        schema_data[field_name] = {
                            'type': field_type,
                            'indexed': indexed
                        }
                    elif len(field_def) == 3:
                        field_type, indexed, required = field_def
                        schema_data[field_name] = {
                            'type': field_type,
                            'indexed': indexed,
                            'required': required
                        }
                    else:
                        raise ValueError(f"Tuple definition for '{field_name}' must have 2 or 3 elements")
                elif hasattr(field_def, 'type'):
                    # MetadataField object
                    schema_data[field_name] = {
                        'type': field_def.type.value if hasattr(field_def.type, 'value') else str(field_def.type),
                        'indexed': field_def.indexed,
                        'required': field_def.required,
                        'default_value': field_def.default_value
                    }
                elif isinstance(field_def, dict):
                    # Dictionary configuration
                    schema_data[field_name] = field_def
                else:
                    raise ValueError(f"Invalid field definition for '{field_name}': {type(field_def)}")
        else:
            raise ValueError("new_schema must be a string, dictionary, or MetadataField mapping")

        # Prepare request payload
        payload = {
            'metadata_schema': schema_data,
            'drop_columns': drop_columns
        }

        url = self._build_url(f"/api/v1/{self.name}/schema")
        response = self._make_request_with_retry("PUT", url, json=payload)

        result = self._handle_response(response)

        # Update local metadata schema cache
        if 'new_schema' in result:
            self._metadata_schema = {}
            for field_name, field_config in result['new_schema'].items():
                self.metadata_schema[field_name] = MetadataField(
                    type=MetadataFieldType(field_config['type']),
                    indexed=field_config.get('indexed', False),
                    required=field_config.get('required', False),
                    default_value=field_config.get('default_value')
                )

        return result.get('changes', {})

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
        return result.get('schema_info', {})

    @property
    def closed(self):
        return self.ping()

    def ping(self, force=False):
        now = time.time()
        if now - self._last_ping_timestamp < 60 and self._last_ping_status and not force:
            return self._last_ping_status

        url = self._build_url(f"/api/v1/databases")
        response = self._make_request_with_retry("GET", url)
        databases = response.json().get("databases", [])

        self._last_ping_status = self.name in databases
        self._last_ping_timestamp = now
        return self._last_ping_status

    def close(self):
        """Close the database connection"""
        # Doesn't do anything since it's remote!
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def hybrid_query(
            self,
            query_text: str,
            k: int = 10,
            vector_weight: float = 0.7,
            metadata_filters: dict = None,
            **kwargs
    ) -> List[QueryResult]:
        """Legacy method for backward compatibility - use query() instead"""
        return self.query(
            query_text,
            search_type="hybrid",
            k=k,
            filters=metadata_filters,
            vector_weight=vector_weight
        )

    def keyword_search(
            self,
            query_text: str,
            k: int = 10,
            metadata_filters: dict = None
    ) -> List[QueryResult]:
        """Legacy method for backward compatibility - use query() instead"""
        return self.query(
            query_text,
            search_type="keyword",
            k=k,
            filters=metadata_filters
        )

    @classmethod
    def database_exists(cls, db_name: str, base_url: str = "http://127.0.0.1:5000", api_key: str = None,
                        authorization_header="Authorization") -> bool:
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
