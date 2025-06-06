# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb/factory.py
"""Enhanced Factory function for LocalVectorDB v1.0 with Async Support

This module provides factory functions that automatically choose between
local and remote database implementations, with support for both sync and async variants.
"""

from pathlib import Path
from typing import Union, Literal

from localvectordb.client import RemoteVectorDB
from localvectordb.database import LocalVectorDB


def VectorDB(
        name: str,
        base_path: Union[str, Path],
        async_mode: bool = False,
        **kwargs
):
    """
    Enhanced factory function that returns the appropriate VectorDB instance
    based on whether base_path looks like a URL or a local path, with optional async support.

    This factory automatically handles the differences between local and remote
    implementations, and between sync and async variants, making it easy to switch between them.

    Parameters
    ----------
    name : str
        Name of the database
    base_path : Union[str, Path]
        Path or URL to the database. If it starts with 'http://' or 'https://',
        a RemoteVectorDB or AsyncRemoteVectorDB will be created. Otherwise,
        a LocalVectorDB or AsyncLocalVectorDB will be created.
    async_mode : bool, default=False
        Whether to return async variants of the database classes.
        If True, returns AsyncLocalVectorDB or AsyncRemoteVectorDB.
        If False, returns LocalVectorDB or RemoteVectorDB.
    **kwargs : dict
        Additional arguments to pass to the appropriate constructor.

        For LocalVectorDB/AsyncLocalVectorDB, these include:
        - metadata_schema: Dict[str, MetadataField] - Schema for metadata fields
        - embedding_provider: str - Provider for embeddings ("ollama", "openai")
        - embedding_model: str - Model name for embeddings
        - embedding_config: Dict[str, Any] - Config for embedding provider
        - chunking_method: str - Method for chunking ("sentences", "tokens", etc.)
        - chunk_size: int - Maximum tokens per chunk
        - chunk_overlap: int - Overlap between chunks
        - enable_gpu: bool - Whether to use GPU for FAISS
        - enable_fts: bool - Whether to enable full-text search
        - create_if_not_exists: bool - Whether to create if not exists

        For RemoteVectorDB/AsyncRemoteVectorDB, these include:
        - api_key: str - API key for authentication
        - create_if_not_exists: bool - Whether to create if not exists
        - metadata_schema: Dict[str, MetadataField] - Schema for metadata fields
        - embedding_provider: str - Provider for embeddings
        - embedding_model: str - Model name for embeddings
        - embedding_config: Dict[str, Any] - Config for embedding provider
        - chunking_method: str - Method for chunking
        - chunk_size: int - Maximum tokens per chunk
        - chunk_overlap: int - Overlap between chunks
        - enable_gpu: bool - Whether to use GPU on server
        - enable_fts: bool - Whether to enable full-text search
        - timeout: float - Timeout for HTTP requests
        - max_retries: int - Number of retry attempts
        - retry_delay: float - Base delay between retries

    Returns
    -------
    Union[LocalVectorDB, RemoteVectorDB, AsyncLocalVectorDB, AsyncRemoteVectorDB]
        An instance of the appropriate vector database class

    Examples
    --------
    Sync local database::

        from localvectordb import VectorDB
        from localvectordb.core import MetadataField, MetadataFieldType

        # Create a sync local database
        db = VectorDB(
            "my_docs",
            "./vector_storage",
            metadata_schema={
                'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                'date': MetadataField(type=MetadataFieldType.DATE, indexed=True)
            },
            embedding_model="nomic-embed-text",
            chunk_size=500
        )

    Async local database::

        # Create an async local database
        async_db = VectorDB(
            "my_docs",
            "./vector_storage",
            async_mode=True,
            embedding_model="nomic-embed-text",
            chunk_size=500
        )

        # Use with async context manager
        async with async_db as db:
            results = await db.query("search term", k=10)

    Sync remote database::

        # Create a sync remote database connection
        db = VectorDB(
            "my_docs",
            "http://localhost:5000",
            api_key="your_api_key",
            metadata_schema={
                'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                'date': MetadataField(type=MetadataFieldType.DATE, indexed=True)
            }
        )

    Async remote database::

        # Create an async remote database connection
        async_db = VectorDB(
            "my_docs",
            "http://localhost:5000",
            async_mode=True,
            api_key="your_api_key",
            max_retries=5,
            timeout=60.0
        )

        async with async_db as db:
            doc_ids = await db.upsert(["Document 1", "Document 2"])

    Seamless switching between sync and async::

        def create_database(use_remote=False, use_async=False):
            if use_remote:
                base_path = "http://localhost:5000"
                extra_kwargs = {"api_key": "your_api_key", "max_retries": 3}
            else:
                base_path = "./local_storage"
                extra_kwargs = {"enable_gpu": True}

            return VectorDB(
                "my_database",
                base_path,
                async_mode=use_async,
                embedding_model="nomic-embed-text",
                chunk_size=500,
                **extra_kwargs
            )

        # Creates LocalVectorDB
        sync_local_db = create_database(use_remote=False, use_async=False)

        # Creates AsyncLocalVectorDB
        async_local_db = create_database(use_remote=False, use_async=True)

        # Creates RemoteVectorDB
        sync_remote_db = create_database(use_remote=True, use_async=False)

        # Creates AsyncRemoteVectorDB
        async_remote_db = create_database(use_remote=True, use_async=True)

    Notes
    -----
    - The factory function automatically filters out incompatible parameters
      for each implementation
    - Both sync and async implementations provide the same document-focused API
    - Local databases require appropriate dependencies (FAISS, SQLite)
    - Remote databases require a running LocalVectorDB server
    - Async variants are recommended for I/O-intensive applications and web servers

    Raises
    ------
    ImportError
        If required dependencies are not available for the chosen implementation
    ValueError
        If invalid parameters are provided for the chosen implementation
    """

    # Convert base_path to string for URL checking
    base_path_str = str(base_path)

    # Check if base_path is a URL
    if base_path_str.lower().startswith(('http://', 'https://')):
        # Remote database
        base_url = base_path_str

        # Filter out LocalVectorDB-specific kwargs that don't apply to RemoteVectorDB
        remote_kwargs = {k: v for k, v in kwargs.items()
                         if k not in [
                             'connection_pool_size',  # Local-only parameter
                         ]}

        if async_mode:
            # Import here to avoid circular imports and handle missing async client
            try:
                from localvectordb.async_client import AsyncRemoteVectorDB
                return AsyncRemoteVectorDB(name=name, base_url=base_url, **remote_kwargs)
            except ImportError as e:
                raise ImportError(
                    "AsyncRemoteVectorDB requires the async client module. "
                    f"Import failed: {e}"
                )
        else:
            return RemoteVectorDB(name=name, base_url=base_url, **remote_kwargs)
    else:
        # Local database

        # Filter out RemoteVectorDB-specific kwargs that don't apply to LocalVectorDB
        local_kwargs = {k: v for k, v in kwargs.items()
                        if k not in [
                            'api_key',  # Remote-only parameter
                            'timeout',  # Remote-only parameter (renamed in remote)
                            'max_retries',  # Remote-only parameter
                            'retry_delay',  # Remote-only parameter
                            'authorization_header',  # Remote-only parameter
                            'connection_limits',  # Remote-only parameter
                        ]}

        if async_mode:
            # Import here to avoid circular imports
            try:
                from localvectordb.async_database import AsyncLocalVectorDB
                return AsyncLocalVectorDB(name=name, base_path=base_path, **local_kwargs)
            except ImportError as e:
                raise ImportError(
                    "AsyncLocalVectorDB requires the async database module. "
                    f"Import failed: {e}"
                )
        else:
            return LocalVectorDB(name=name, base_path=base_path, **local_kwargs)


async def AsyncVectorDB(
        name: str,
        base_path: Union[str, Path],
        **kwargs
):
    """
    Async factory function that returns initialized async VectorDB instances.

    This is a convenience function that automatically initializes async databases
    and is equivalent to VectorDB(..., async_mode=True) followed by initialization.

    Parameters
    ----------
    name : str
        Database name
    base_path : Union[str, Path]
        Path or URL to the database
    **kwargs
        All other parameters passed to the appropriate async constructor

    Returns
    -------
    Union[AsyncLocalVectorDB, AsyncRemoteVectorDB]
        Initialized async database instance

    Examples
    --------
    Async local database::

        async_db = await AsyncVectorDB("my_db", "./local_path")
        try:
            results = await async_db.query("search term")
        finally:
            await async_db.close()

    Async remote database::

        async_db = await AsyncVectorDB(
            "my_db",
            "http://localhost:5000",
            api_key="your_api_key"
        )
        try:
            stats = await async_db.get_stats()
        finally:
            await async_db.close()

    Notes
    -----
    - This function automatically initializes the database connection
    - Remember to call close() or use async context managers
    - For most use cases, prefer the async context manager pattern
    """

    # Create the async database instance
    db = VectorDB(name, base_path, async_mode=True, **kwargs)

    # Initialize it
    await db._ensure_initialized()

    return db


def create_vectordb(
        name: str,
        base_path: Union[str, Path],
        database_type: Literal["local", "remote"] = "auto",
        async_mode: bool = False,
        **kwargs
):
    """
    Explicit factory function with clear database type specification.

    This function provides an alternative to VectorDB() when you want to be
    explicit about the database type rather than inferring from the base_path.

    Parameters
    ----------
    name : str
        Database name
    base_path : Union[str, Path]
        Path or URL to the database
    database_type : Literal["local", "remote", "auto"], default="auto"
        Explicitly specify database type:
        - "local": Force LocalVectorDB/AsyncLocalVectorDB
        - "remote": Force RemoteVectorDB/AsyncRemoteVectorDB
        - "auto": Auto-detect based on base_path (same as VectorDB())
    async_mode : bool, default=False
        Whether to return async variants
    **kwargs
        Additional arguments passed to the constructor

    Returns
    -------
    Union[LocalVectorDB, RemoteVectorDB, AsyncLocalVectorDB, AsyncRemoteVectorDB]
        Database instance of the specified type

    Examples
    --------
    Force local database even with URL-like path::

        # This creates a LocalVectorDB even though the path looks like a URL
        db = create_vectordb(
            "my_db",
            "http_server_backup",
            database_type="local"
        )

    Force remote database with local-looking path::

        # This creates a RemoteVectorDB even though the path looks local
        db = create_vectordb(
            "my_db",
            "localhost:5000",
            database_type="remote",
            api_key="your_api_key"
        )
    """

    if database_type == "auto":
        return VectorDB(name, base_path, async_mode=async_mode, **kwargs)
    elif database_type == "local":
        # Force local database
        local_kwargs = {k: v for k, v in kwargs.items()
                        if k not in [
                            'api_key', 'timeout', 'max_retries', 'retry_delay',
                            'authorization_header', 'connection_limits'
                        ]}

        if async_mode:
            from localvectordb.async_database import AsyncLocalVectorDB
            return AsyncLocalVectorDB(name=name, base_path=base_path, **local_kwargs)
        else:
            return LocalVectorDB(name=name, base_path=base_path, **local_kwargs)
    elif database_type == "remote":
        # Force remote database
        remote_kwargs = {k: v for k, v in kwargs.items()
                         if k not in ['connection_pool_size']}

        # Ensure base_path is treated as URL
        base_url = str(base_path)
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"

        if async_mode:
            from localvectordb.async_client import AsyncRemoteVectorDB
            return AsyncRemoteVectorDB(name=name, base_url=base_url, **remote_kwargs)
        else:
            return RemoteVectorDB(name=name, base_url=base_url, **remote_kwargs)
    else:
        raise ValueError(f"Invalid database_type: {database_type}. Must be 'local', 'remote', or 'auto'")

