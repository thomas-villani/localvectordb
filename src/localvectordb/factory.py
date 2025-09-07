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
import os.path
from pathlib import Path
from typing import Union
from urllib.parse import urlparse, parse_qs

from localvectordb.client import RemoteVectorDB
from localvectordb.core import AnyVectorDB
from localvectordb.database import LocalVectorDB


def VectorDB(
        name: str,
        base_path: Union[str, Path],
        **kwargs
) -> AnyVectorDB:
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
    **kwargs : dict
        Additional arguments to pass to the appropriate constructor.

        These include:
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

        For RemoteVectorDB, these include:
        - api_key: str - API key for authentication
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

        # Async built in
        async with db as async_db:
            doc_ids = await async_db.upsert_async(["Document 1", "Document 2"])


    Notes
    -----
    - The factory function automatically filters out incompatible parameters
      for each implementation
    - Local databases require appropriate dependencies (FAISS, SQLite)
    - Remote databases require a running LocalVectorDB server

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


        return LocalVectorDB(name=name, base_path=base_path, **local_kwargs)

def _fix_types(item):
    if isinstance(item, list):
        return [_fix_types(i) for i in item]
    if isinstance(item, str):
        if item.replace("-","").isdigit():
            return int(item)
        if item.replace(".", "").replace("-","").isdigit():
            return float(item)
        if item.lower() == "false":
            return False
        if item.lower() == "true":
            return True
    return item


def from_uri(db_uri: str) -> AnyVectorDB:
    parsed = urlparse(db_uri)
    if parsed.scheme != "lvdb":
        raise ValueError(f"Invalid database URI: scheme must be 'lvdb', found: '{parsed.scheme}'")

    # The following should only be true if using absolute path
    absolute_path = parsed.hostname is None

    if absolute_path:
        full_path, db_name = parsed.path.rsplit("/", 1)
        if not os.path.exists(full_path):
            raise ValueError(f"Invalid database URI: db folder not found: '{full_path}'")
    else:
        complete = parsed.netloc + parsed.path
        full_path, db_name = complete.rsplit("/", 1)
        if parsed.port or not os.path.exists(full_path):
            full_path = "http://" + full_path

    if not db_name:
        raise ValueError("Must provide a valid database URI, expected database name specified as path.")

    query_params = {}
    if parsed.query:
        parsed_query = parse_qs(parsed.query)
        query_params = {k: _fix_types(v if len(v) > 1 else v[0]) for k, v in parsed_query.items()}

    return VectorDB(db_name, full_path, **query_params)