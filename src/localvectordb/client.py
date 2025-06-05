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
import time
from typing import Union, Any, Optional, Literal, Dict, List

import httpx

from localvectordb.core import MetadataField, MetadataFieldType, QueryResult, Document
from localvectordb.exceptions import (
    DatabaseNotFoundError, DuplicateDocumentIDError, EmbeddingError, BaseLocalVectorDBException, DatabaseError
)


class RemoteVectorDB:
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
            authorization_header: str = "Authorization"
    ):
        self.name = name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.request_timeout = request_timeout

        # Configuration
        self.metadata_schema = metadata_schema or {}
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
        # self._closed = False

        # Check if database exists and create if needed
        if create_if_not_exists:
            self._ensure_database_exists()
        else:
            # Load existing database info
            self._load_database_info()


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
        with httpx.Client() as client:
            response = client.get(url, headers=self._get_headers(), timeout=self.request_timeout)

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
        self.metadata_schema = {}
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

        with httpx.Client() as client:
            response = client.post(url, json=payload, headers=self._get_headers(), timeout=self.request_timeout)

        created_db_info = self._handle_response(response)
        config = created_db_info.get("config", {})
        self._embedding_dimension = config.get("embedding_dimension", 0)

    @property
    def embedding_model(self) -> str:
        """Return the embedding model name"""
        return self._embedding_model

    @property
    def embedding_provider(self) -> str:
        """Return the provider name"""
        return self._embedding_provider

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
    def stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        url = self._build_url(f"/api/v1/{self.name}/info")
        with httpx.Client() as client:
            response = client.get(url, headers=self._get_headers(), timeout=self.request_timeout)
        db_info = self._handle_response(response)
        return db_info.get("stats", {})

    def upsert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100
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
        with httpx.Client() as client:
            response = client.post(url, json=payload, headers=self._get_headers(), timeout=self.request_timeout)
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
        with httpx.Client() as client:
            response = client.post(url, json=payload, headers=self._get_headers(), timeout=self.request_timeout)
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

            with httpx.Client() as client:
                response = client.get(url, headers=self._get_headers(), timeout=self.request_timeout)

            try:
                result = self._handle_response(response)
                doc = Document.from_dict(result)
                return doc
            except DatabaseNotFoundError:
                return None
        else:
            # Handle multiple IDs - make individual requests for each ID
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
        check_ids = [ids] if single_id else ids

        url = self._build_url(f"/api/v1/{self.name}/documents/exists")
        with httpx.Client() as client:
            response = client.post(url, headers=self._get_headers(), json={"ids": check_ids},
                                   timeout=self.request_timeout)
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
            with httpx.Client() as client:
                response = client.delete(url, headers=self._get_headers(), timeout=self.request_timeout)
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
        with httpx.Client() as client:
            response = client.put(url, json=payload, headers=self._get_headers(), timeout=self.request_timeout)

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
        with httpx.Client() as client:
            response = client.post(url, json=payload, headers=self._get_headers(), timeout=self.request_timeout)
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
        with httpx.Client() as client:
            response = client.post(url, json=payload, headers=self._get_headers(), timeout=self.request_timeout)
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
        with httpx.Client() as client:
            response = client.put(url, json=payload, headers=self._get_headers(), timeout=self.request_timeout)

        result = self._handle_response(response)

        # Update local metadata schema cache
        if 'new_schema' in result:
            self.metadata_schema = {}
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
        with httpx.Client() as client:
            response = client.get(url, headers=self._get_headers(), timeout=self.request_timeout)

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
        with httpx.Client() as client:
            response = client.post(url, headers=self._get_headers(), timeout=self.request_timeout)
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
            raise DatabaseError(f"Could not connect to remote database: {str(e)}") from e

        if response.status_code == 200:
            result = response.json()
            return db_name in result.get("databases", [])
        return False
