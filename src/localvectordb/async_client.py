# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb/async_client.py
"""
AsyncRemoteVectorDB - Async client for LocalVectorDB v1.0 Server

This module provides an async client interface to interact with a LocalVectorDB server.
It implements the same document-focused interface as AsyncLocalVectorDB but connects
to a remote server via HTTP using async requests.

Features:
- Drop-in async replacement for RemoteVectorDB
- Native async HTTP requests using httpx
- Async context manager support
- Same API as AsyncLocalVectorDB for seamless switching
- Optimized batch operations
- Connection pooling and timeouts
- Retry logic with exponential backoff

Examples
--------
Basic usage with context manager::

    async with AsyncRemoteVectorDB(
        "my_db",
        "http://localhost:5000",
        api_key="your_api_key"
    ) as db:
        # Insert documents
        doc_ids = await db.upsert([
            "This is a test document",
            "Another document for testing"
        ])

        # Query documents
        results = await db.query("test document", k=5)

        # Get documents
        docs = await db.get(doc_ids)

Factory function usage::

    db = await create_async_remote_vectordb(
        "my_db",
        "http://localhost:5000",
        api_key="your_api_key"
    )
    try:
        stats = await db.get_stats()
        print(f"Database has {stats['documents']} documents")
    finally:
        await db.close()

Batch operations with retry::

    async with AsyncRemoteVectorDB(
        "my_db",
        "http://localhost:5000",
        max_retries=3,
        timeout=60.0
    ) as db:
        # Batch upsert with optimized async requests
        await db.upsert(large_document_list, batch_size=50)
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Union, Literal, Any

import httpx

from localvectordb.client import RemoteEmbeddingProvider
from localvectordb.core import MetadataField, MetadataFieldType, QueryResult, Document, AsyncBaseVectorDB
from localvectordb.exceptions import (
    DatabaseNotFoundError, DuplicateDocumentIDError, EmbeddingError,
    BaseLocalVectorDBException, DatabaseError
)

logger = logging.getLogger(__name__)


class AsyncRemoteVectorDB(AsyncBaseVectorDB):
    """
    Async client for interacting with a LocalVectorDB v1.0 server.

    This client provides the same document-focused interface as AsyncLocalVectorDB
    but connects to a remote server via HTTP using async requests.

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
    timeout : float, optional
        Timeout for HTTP requests to the server in seconds, by default 30.0
    max_retries : int, optional
        Maximum number of retries for failed requests, by default 3
    retry_delay : float, optional
        Base delay between retries in seconds, by default 1.0
    authorization_header : str, default="Authorization"
        The server can be configured to accept alternate headers
    connection_limits : httpx.Limits, optional
        Connection pool limits for httpx client
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
            timeout: float = 30.0,
            max_retries: int = 3,
            retry_delay: float = 1.0,
            authorization_header: str = "Authorization",
            connection_limits: Optional[httpx.Limits] = None
    ):
        self.name = name
        self.base_url = base_url.rstrip('/')

        api_key_env_var = "LVDB_API_KEY"
        # Allow the user to specify an environment variable by prefixing $
        if api_key and api_key.startswith("$") and api_key[1:].isupper():
            api_key_env_var = api_key[1:]
            api_key = None

        self.api_key = api_key or os.getenv(api_key_env_var)

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.authorization_header = authorization_header

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

        # HTTP client configuration
        self._connection_limits = connection_limits or httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0
        )

        # State variables
        self._client: Optional[httpx.AsyncClient] = None
        self._embedding_dimension = 0
        self._closed = False
        self._initialized = False

        self._remote_embedding_provider = RemoteEmbeddingProvider(
            db_name=self.name,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            authorization_header=self.authorization_header
        )

        # Store initialization parameters for lazy initialization
        self._create_if_not_exists = create_if_not_exists

    async def __aenter__(self):
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


    def _get_headers(self) -> dict:
        """Get headers for API requests including authentication if provided"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.authorization_header] = f"Bearer {self.api_key}"
        return headers

    def _build_url(self, endpoint: str) -> str:
        """Build a full URL for the given endpoint"""
        return f"{self.base_url}{endpoint}"

    async def _handle_response(self, response: httpx.Response) -> dict:
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

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is available"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=self._connection_limits,
                headers=self._get_headers()
            )
        return self._client

    async def _make_request_with_retry(
            self,
            method: str,
            url: str,
            **kwargs
    ) -> httpx.Response | None:
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
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Request failed with {response.status_code}, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
                await asyncio.sleep(delay)

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
                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        return None

    async def _ensure_initialized(self):
        """Lazy initialization of the remote database connection"""
        if self._initialized and not self._closed:
            return

        if self._closed:
            raise DatabaseError("Database connection has been closed")

        try:
            # Check if database exists and create if needed
            if self._create_if_not_exists:
                await self._ensure_database_exists()
            else:
                # Load existing database info
                await self._load_database_info()

            self._initialized = True
            logger.info(f"AsyncRemoteVectorDB '{self.name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AsyncRemoteVectorDB: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")

    async def _ensure_database_exists(self) -> None:
        """Check if database exists and create if it doesn't"""
        try:
            # Try to get database info
            await self._load_database_info()
        except DatabaseNotFoundError:
            # Database doesn't exist, create it
            await self._create_database()

    async def _load_database_info(self) -> None:
        """Load database information from server"""
        url = self._build_url(f"/api/v1/{self.name}/info")

        response = await self._make_request_with_retry("GET", url)
        db_info = await self._handle_response(response)

        # Update configuration from server
        config = db_info.get("config", {})
        self._embedding_provider = config.get("embedding_provider", self._embedding_provider)
        self._embedding_model = config.get("embedding_model", self._embedding_model)
        self._embedding_dimension = config.get("embedding_dimension", 0)
        self._chunking_method = config.get("chunking_method", self._chunking_method)
        self._chunk_size = config.get("chunk_size", self._chunk_size)
        self._chunk_overlap = config.get("chunk_overlap", self._chunk_overlap)
        self._enable_fts = config.get("fts_enabled", self._enable_fts)

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

    async def _create_database(self) -> None:
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

        response = await self._make_request_with_retry("POST", url, json=payload)
        created_db_info = await self._handle_response(response)
        config = created_db_info.get("config", {})
        self._embedding_dimension = config.get("embedding_dimension", 0)

    @property
    def embedding_provider(self) -> RemoteEmbeddingProvider:
        """Return the remote embedding provider instance."""
        return self._remote_embedding_provider

    @property
    def metadata_schema(self) -> Dict[str, MetadataField]:
        return self._metadata_schema

    # Property accessors (synchronous, safe to access)
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

    @property
    def closed(self) -> bool:
        """Check if the connection is closed"""
        return self._closed

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        await self._ensure_initialized()

        url = self._build_url(f"/api/v1/{self.name}/info")
        response = await self._make_request_with_retry("GET", url)
        db_info = await self._handle_response(response)
        return db_info.get("stats", {})

    # Core async methods
    async def upsert(
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
            If provided, skip chunks which have semantic similarity greater than this value (pre-deduplication)

        Returns
        -------
        List[str]
            List of document IDs that were upserted
        """
        await self._ensure_initialized()

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
        response = await self._make_request_with_retry("POST", url, json=payload)
        result = await self._handle_response(response)

        return result.get("ids", [])

    async def insert(
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
        await self._ensure_initialized()

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
        response = await self._make_request_with_retry("POST", url, json=payload)
        result = await self._handle_response(response)

        return result.get("ids", [])

    async def get(self, ids: Union[str, List[str]]) -> Union[Document, List[Document], None]:
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
        await self._ensure_initialized()

        single_id = isinstance(ids, str)
        if single_id:
            url = self._build_url(f"/api/v1/{self.name}/documents/{ids}")

            try:
                response = await self._make_request_with_retry("GET", url)
                result = await self._handle_response(response)
                doc = Document.from_dict(result)
                return doc
            except DatabaseNotFoundError:
                return None
        else:
            # Handle multiple IDs - make individual requests for each ID
            docs = []
            for doc_id in ids:
                doc = await self.get(doc_id)
                if doc is not None:
                    docs.append(doc)
            return docs

    async def exists(self, ids: Union[str, List[str]]) -> Union[bool, List[bool]]:
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
        await self._ensure_initialized()

        single_id = isinstance(ids, str)
        check_ids = [ids] if single_id else ids

        url = self._build_url(f"/api/v1/{self.name}/documents/exists")
        response = await self._make_request_with_retry(
            "POST", url, json={"ids": check_ids}
        )
        results = await self._handle_response(response)

        return results.get("exists")[0] if single_id else results.get("exists")

    async def delete(self, ids: Union[str, List[str]]) -> int:
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
        await self._ensure_initialized()

        if isinstance(ids, str):
            ids = [ids]

        deleted_count = 0
        # Use asyncio.gather for concurrent deletions
        tasks = []

        async def delete_single(doc_id: str) -> int:
            url = self._build_url(f"/api/v1/{self.name}/documents/{doc_id}")
            response = await self._make_request_with_retry("DELETE", url)
            result = await self._handle_response(response)
            return result.get("deleted_count", 0)

        tasks = [delete_single(doc_id) for doc_id in ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, int):
                deleted_count += result
            elif isinstance(result, Exception):
                logger.warning(f"Failed to delete document: {result}")

        return deleted_count

    async def update(
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
        await self._ensure_initialized()

        if not content and not metadata:
            return False

        payload = {}
        if content is not None:
            payload["content"] = content
        if metadata is not None:
            payload["metadata"] = metadata

        url = self._build_url(f"/api/v1/{self.name}/documents/{doc_id}")

        try:
            response = await self._make_request_with_retry("PUT", url, json=payload)
            result = await self._handle_response(response)
            return result.get("updated", False)
        except DatabaseNotFoundError:
            return False

    async def query(
            self,
            query: str,
            *,
            search_type: Literal['vector', 'keyword', 'hybrid'] = 'vector',
            return_type: Literal['documents', 'chunks', 'context'] = 'documents',
            k: int = 10,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
            vector_weight: float = 0.7,  # For search_type='hybrid'
            context_window: int = 2,     # For return_type='context'
            semantic_dedup_threshold: Optional[float] = None,
            document_scoring_method: Literal[
                "best", "average", "worst", "weighted_average", "frequency_boost"] = "frequency_boost"
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

        Returns
        -------
        List[QueryResult]
            Search results with normalized scores
        """
        await self._ensure_initialized()

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
        response = await self._make_request_with_retry("POST", url, json=payload)
        result = await self._handle_response(response)

        # Process results
        raw_results = result.get("results", [])
        return [QueryResult.from_dict(res) for res in raw_results]

    async def filter(
            self,
            where: Optional[Dict[str, Any]] = None,
            order_by: Optional[str] = None,
            limit: Optional[int] = None,
            offset: int = 0
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
        await self._ensure_initialized()

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
        response = await self._make_request_with_retry("POST", url, json=payload)
        result = await self._handle_response(response)

        # Process results
        raw_docs = result.get("documents", [])
        return [Document.from_dict(doc) for doc in raw_docs]

    async def save(self):
        """Save the database (no-op for remote client)"""
        # No-op for remote client - server handles saving automatically
        pass

    async def update_metadata_schema(
            self,
            new_schema: Union[str, Dict[str, Any]],
            drop_columns: bool = False,
            column_mapping: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Update the metadata schema for the remote database

        Parameters
        ----------
        new_schema : Union[str, Dict[str, Any]]
            The new metadata schema to apply
        drop_columns : bool, default=False
            Whether to actually drop columns that are no longer in the schema
        column_mapping : dict, optional
            Optionally provide a mapping dict with old-column (key) -> new-column (value)

        Returns
        -------
        Dict[str, Any]
            Summary of changes made
        """
        await self._ensure_initialized()

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
        response = await self._make_request_with_retry("PUT", url, json=payload)
        result = await self._handle_response(response)

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

    async def get_metadata_schema_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current metadata schema

        Returns
        -------
        Dict[str, Any]
            Dictionary containing schema information
        """
        await self._ensure_initialized()

        url = self._build_url(f"/api/v1/{self.name}/schema")
        response = await self._make_request_with_retry("GET", url)
        result = await self._handle_response(response)
        return result.get('schema_info', {})

    async def close(self):
        """Close the database connection"""
        self._closed = True
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # Legacy compatibility methods
    async def hybrid_query(
            self,
            query_text: str,
            k: int = 10,
            vector_weight: float = 0.7,
            metadata_filters: dict = None,
            **kwargs
    ) -> List[QueryResult]:
        """Legacy method for backward compatibility - use query() instead"""
        return await self.query(
            query_text,
            search_type="hybrid",
            k=k,
            filters=metadata_filters,
            vector_weight=vector_weight
        )

    async def keyword_search(
            self,
            query_text: str,
            k: int = 10,
            metadata_filters: dict = None
    ) -> List[QueryResult]:
        """Legacy method for backward compatibility - use query() instead"""
        return await self.query(
            query_text,
            search_type="keyword",
            k=k,
            filters=metadata_filters
        )

    @classmethod
    async def database_exists(
            cls,
            db_name: str,
            base_url: str = "http://127.0.0.1:5000",
            api_key: str = None,
            authorization_header: str = "Authorization"
    ) -> bool:
        """Check if a database exists on the server"""
        url = f"{base_url}/api/v1/databases"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers[authorization_header] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
        except httpx.ConnectError as e:
            raise DatabaseError(f"Could not connect to remote database: {str(e)}") from e

        if response.status_code == 200:
            result = response.json()
            return db_name in result.get("databases", [])
        return False


# Factory function for convenient creation
async def create_async_remote_vectordb(
        name: str,
        base_url: str = "http://127.0.0.1:5000",
        **kwargs
) -> AsyncRemoteVectorDB:
    """
    Factory function to create and initialize an AsyncRemoteVectorDB.

    Parameters
    ----------
    name : str
        Database name
    base_url : str
        Base URL for the remote server
    **kwargs
        All other parameters passed to AsyncRemoteVectorDB constructor

    Returns
    -------
    AsyncRemoteVectorDB
        Initialized async remote database instance
    """
    db = AsyncRemoteVectorDB(name, base_url, **kwargs)
    await db._ensure_initialized()
    return db


# Export main classes and functions
__all__ = [
    'AsyncRemoteVectorDB',
    'create_async_remote_vectordb'
]