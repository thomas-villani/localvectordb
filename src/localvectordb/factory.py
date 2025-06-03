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
"""Factory function for LocalVectorDB v1.0

This module provides a factory function that automatically chooses between
local and remote database implementations based on the base_path parameter.
"""

from pathlib import Path
from typing import Union

from localvectordb.client import RemoteVectorDB
from localvectordb.database import LocalVectorDB


def VectorDB(name: str, base_path: Union[str, Path], **kwargs):
    """
    Factory function that returns either a LocalVectorDB or RemoteVectorDB instance
    based on whether base_path looks like a URL or a local path.

    This factory automatically handles the differences between local and remote
    implementations, making it easy to switch between them.

    Parameters
    ----------
    name : str
        Name of the database
    base_path : Union[str, Path]
        Path or URL to the database. If it starts with 'http://' or 'https://',
        a RemoteVectorDB will be created. Otherwise, a LocalVectorDB will be created.
    **kwargs : dict
        Additional arguments to pass to the appropriate constructor.

        For LocalVectorDB, these include:
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
        - request_timeout: int - Timeout for HTTP requests

    Returns
    -------
    Union[LocalVectorDB, RemoteVectorDB]
        An instance of the appropriate vector database class

    Examples
    --------
    Local database::

        from localvectordb import VectorDB
        from localvectordb.core import MetadataField, MetadataFieldType

        # Create a local database
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

    Remote database::

        # Create a remote database connection
        db = VectorDB(
            "my_docs",
            "http://localhost:5000",
            api_key="your_api_key",
            metadata_schema={
                'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                'date': MetadataField(type=MetadataFieldType.DATE, indexed=True)
            }
        )

    Seamless switching::

        # Use the same code for both local and remote
        def create_database(use_remote=False):
            if use_remote:
                base_path = "http://localhost:5000"
                extra_kwargs = {"api_key": "your_api_key"}
            else:
                base_path = "./local_storage"
                extra_kwargs = {"enable_gpu": True}

            return VectorDB(
                "my_database",
                base_path,
                embedding_model="nomic-embed-text",
                chunk_size=500,
                **extra_kwargs
            )

        # Creates LocalVectorDB
        local_db = create_database(use_remote=False)

        # Creates RemoteVectorDB
        remote_db = create_database(use_remote=True)

    Notes
    -----
    - The factory function automatically filters out incompatible parameters
      for each implementation
    - Both implementations provide the same document-focused API
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
        # Remote database - use RemoteVectorDB
        base_url = base_path_str

        # Filter out LocalVectorDB-specific kwargs that don't apply to RemoteVectorDB
        remote_kwargs = {k: v for k, v in kwargs.items()
                         if k not in [
                             'connection_pool_size',  # Local-only parameter
                         ]}


        return RemoteVectorDB(name=name, base_url=base_url, **remote_kwargs)
    else:
        # Local database - use LocalVectorDB v1.0

        # Filter out RemoteVectorDB-specific kwargs that don't apply to LocalVectorDB
        local_kwargs = {k: v for k, v in kwargs.items()
                        if k not in [
                            'api_key',  # Remote-only parameter
                            'request_timeout',  # Remote-only parameter
                        ]}

        return LocalVectorDB(name=name, base_path=base_path, **local_kwargs)
