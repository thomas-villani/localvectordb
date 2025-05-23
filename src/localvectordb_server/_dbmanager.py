# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb_server/_dbmanager.py
#  Copyright (c) 2023-2025 Tom Villani, Ph.D. All rights reserved.
#
"""
Updated database manager for LocalVectorDB with document-first architecture,
structured metadata support, and unified query interface.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Union, Literal, Optional, Any

from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.exceptions import DatabaseNotFoundError
from localvectordb.utils import make_filename_safe
from werkzeug.exceptions import BadRequest

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for a vector database instance"""
    name: str
    metadata_schema: Optional[Dict[str, MetadataField]] = None
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_config: Optional[Dict[str, Any]] = None
    chunking_method: str = "sentences"
    chunk_size: int = 500
    chunk_overlap: int = 1
    enable_gpu: bool = False
    enable_fts: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> 'DatabaseConfig':
        """Create config from dictionary, with validation"""


        try:
            name = make_filename_safe(data['name'])
            if not name:
                raise ValueError('Must provide valid database name')

            # Parse metadata schema if provided
            metadata_schema = None
            if 'metadata_schema' in data:
                metadata_schema = cls._parse_metadata_schema(data['metadata_schema'])

            return cls(
                name=name,
                metadata_schema=metadata_schema,
                embedding_provider=data.get('embedding_provider', 'ollama'),
                embedding_model=data.get('embedding_model', 'nomic-embed-text'),
                embedding_config=data.get('embedding_config', {}),
                chunking_method=data.get('chunking_method', 'sentences'),
                chunk_size=data.get('chunk_size', 500),
                chunk_overlap=data.get('chunk_overlap', 1),
                enable_gpu=data.get('enable_gpu', False),
                enable_fts=data.get('enable_fts', True)
            )
        except (KeyError, ValueError) as e:
            raise BadRequest(f"Invalid database configuration: {str(repr(e))}")

    @staticmethod
    def _parse_metadata_schema(schema_data: Dict[str, Any]) -> Dict[str, MetadataField]:
        """Parse metadata schema from request data"""
        if not schema_data:
            return {}

        parsed_schema = {}
        for field_name, field_config in schema_data.items():
            if isinstance(field_config, str):
                # Simple string type
                field_type = MetadataFieldType(field_config)
                parsed_schema[field_name] = MetadataField(type=field_type)
            elif isinstance(field_config, dict):
                # Full field configuration
                field_type = MetadataFieldType(field_config.get('type', 'text'))
                parsed_schema[field_name] = MetadataField(
                    type=field_type,
                    indexed=field_config.get('indexed', False),
                    required=field_config.get('required', False),
                    default_value=field_config.get('default_value')
                )
            else:
                from werkzeug.exceptions import BadRequest
                raise BadRequest(f"Invalid metadata field configuration for '{field_name}'")

        return parsed_schema

    def validate(self) -> None:
        """Validate configuration parameters"""
        from werkzeug.exceptions import BadRequest

        if not self.name or not isinstance(self.name, str):
            raise BadRequest("Database name must be a non-empty string")

        if self.embedding_provider not in ("ollama", "openai"):
            raise BadRequest("embedding_provider must be 'ollama' or 'openai'")

        if self.chunk_size <= 0:
            raise BadRequest("chunk_size must be a positive integer")

        if self.chunk_overlap < 0:
            raise BadRequest("chunk_overlap must be a non-negative integer")

        # Validate chunking method
        valid_methods = [
            "sentences", "words", "characters", "tokens",
            "lines", "sections", "paragraphs", "code-blocks"
        ]
        if self.chunking_method not in valid_methods:
            raise BadRequest(f"chunking_method must be one of {valid_methods}")


class DatabaseManager:
    """Manages multiple vector databases with timeout-based cleanup"""

    def __init__(self, app):
        self.app = app
        self.config = app.config
        self.databases: Dict[str, tuple["LocalVectorDB", datetime]] = {}
        self.lock = threading.Lock()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        # Ensure database directory exists
        db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
        db_path.mkdir(parents=True, exist_ok=True)

    def create_db(self, db_config: DatabaseConfig) -> "LocalVectorDB":
        """Create a new database with specified configuration"""
        db_config.validate()

        if db_config.name in self.databases:
            from werkzeug.exceptions import BadRequest
            raise BadRequest(f"Database '{db_config.name}' already exists")

        from localvectordb.database import LocalVectorDB

        # Create new database instance
        db = LocalVectorDB(
            name=db_config.name,
            base_path=Path(self.config.get("DB_ROOT_DIR", ".lvdb")),
            metadata_schema=db_config.metadata_schema,
            embedding_provider=db_config.embedding_provider,
            embedding_model=db_config.embedding_model,
            embedding_config=db_config.embedding_config,
            chunking_method=db_config.chunking_method,
            chunk_size=db_config.chunk_size,
            chunk_overlap=db_config.chunk_overlap,
            enable_gpu=db_config.enable_gpu,
            enable_fts=db_config.enable_fts,
            create_if_not_exists=True
        )

        self.databases[db_config.name] = (db, datetime.now())
        logger.info(f"Created new database: {db_config.name}")
        return db

    def get_db(self, name: str) -> "LocalVectorDB":
        """Get an existing database instance"""

        if name not in self.databases:
            # Check if database exists on disk
            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
            db_sqlite_file = db_path / f"{name}.sqlite"

            if not db_sqlite_file.exists():
                from werkzeug.exceptions import NotFound
                raise NotFound(f"Database '{name}' not found")

            from localvectordb.database import LocalVectorDB

            # Load existing database
            try:
                db = LocalVectorDB(
                    name=name,
                    base_path=db_path,
                    create_if_not_exists=False
                )
                self.databases[name] = (db, datetime.now())
                logger.info(f"Loaded existing database: {name}")
            except DatabaseNotFoundError:
                from werkzeug.exceptions import NotFound
                raise NotFound(f"Database '{name}' not found")
        else:
            # Update last access time
            db, _ = self.databases[name]
            self.databases[name] = (db, datetime.now())

        return self.databases[name][0]

    def list_databases(self) -> list[str]:
        """List all available databases"""
        db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
        if not db_path.exists():
            return []

        return [d.stem for d in db_path.iterdir() if d.suffix.lower() == ".sqlite"]

    def search_databases(
            self,
            query: str,
            database_names: Optional[list[str]] = None,
            search_type: Literal["vector", "keyword", "hybrid"] = "vector",
            return_type: Literal["documents", "chunks"] = "documents",
            k: int = 10,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
            vector_weight: float = 0.7
    ) -> Dict[str, Union[list, str]]:
        """Search across multiple databases using the unified query interface

        Parameters
        ----------
        query : str
            Query text
        database_names : Optional[list[str]]
            List of databases to search. If None, searches all databases.
        search_type : Literal["vector", "keyword", "hybrid"]
            Type of search to perform
        return_type : Literal["documents", "chunks"]
            Whether to return full documents or individual chunks
        k : int
            Maximum number of results to return per database
        score_threshold : float
            Minimum score threshold (0-1, higher=better)
        filters : Optional[Dict[str, Any]]
            Metadata filters
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)

        Returns
        -------
        Dict[str, Union[list, str]]
            Dictionary mapping database names to either a list of results or an error message
        """
        if database_names is None:
            database_names = self.list_databases()

        results = {}

        # Search each database
        for db_name in database_names:
            try:
                # Get database instance
                db = self.get_db(db_name)

                # Perform search using unified query interface
                db_results = db.query(
                    query=query,
                    search_type=search_type,
                    return_type=return_type,
                    k=k,
                    score_threshold=score_threshold,
                    filters=filters,
                    vector_weight=vector_weight
                )

                results[db_name] = db_results

            except Exception as e:
                logger.error(f"Error searching database {db_name}: {e}")
                results[db_name] = f"Search failed: {str(e)}"

        return results

    def get_embeddings_for_model(
            self,
            query_texts: Union[str, list[str]],
            provider: str,
            model: str
    ) -> list[list[float]]:
        """Get embeddings for query texts using specified provider and model

        Parameters
        ----------
        query_texts : Union[str, list[str]]
            Text or list of texts to embed
        provider : str
            Embedding provider ('ollama' or 'openai')
        model : str
            Model name

        Returns
        -------
        list[list[float]]
            List of embedding vectors
        """
        try:
            from localvectordb.embeddings import EmbeddingRegistry

            if isinstance(query_texts, str):
                query_texts = [query_texts]

            # Create embedding provider
            embedding_provider = EmbeddingRegistry.create_provider(provider, model)

            # Get embeddings
            embeddings = embedding_provider.embed_sync(query_texts)

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error getting embeddings for model {model}: {e}")
            raise RuntimeError(f"Failed to get embeddings: {e}")

    def _cleanup_loop(self):
        """Periodically check for and cleanup inactive databases"""
        while True:
            self._cleanup_inactive()
            threading.Event().wait(60)  # Check every minute

    def _cleanup_inactive(self):
        """Close inactive database connections"""
        now = datetime.now()
        timeout = timedelta(seconds=self.config.get("DB_TIMEOUT", 3600))  # Default 1 hour

        with self.lock:
            to_remove = []
            for name, (db, last_access) in self.databases.items():
                if now - last_access > timeout:
                    logger.info(f"Closing inactive database: {name}")
                    try:
                        db.close()
                    except Exception as e:
                        logger.error(f"Error closing database {name}: {e}")
                    to_remove.append(name)

            for name in to_remove:
                del self.databases[name]

    def close_all(self):
        """Close all database connections"""
        with self.lock:
            for name, (db, _) in self.databases.items():
                logger.info(f"Closing database: {name}")
                try:
                    db.close()
                except Exception as e:
                    logger.error(f"Error closing database {name}: {e}")
            self.databases.clear()

    def get_database_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a specific database"""
        try:
            db = self.get_db(name)
            return db.stats
        except Exception as e:
            logger.error(f"Error getting stats for database {name}: {e}")
            return {"error": str(e)}

    def get_all_database_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all databases"""
        stats = {}
        for db_name in self.list_databases():
            try:
                stats[db_name] = self.get_database_stats(db_name)
            except Exception as e:
                logger.error(f"Error getting stats for database {db_name}: {e}")
                stats[db_name] = {"error": str(e)}

        return stats

    def backup_database(self, name: str, backup_path: str):
        """Create a backup of a database"""
        try:
            db = self.get_db(name)

            # For now, we'll just copy the files
            # In a more sophisticated implementation, we might use the export functionality
            import shutil

            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
            source_sqlite = db_path / f"{name}.sqlite"
            source_faiss = db_path / f"{name}.faiss"

            backup_path = Path(backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)

            if source_sqlite.exists():
                shutil.copy2(source_sqlite, backup_path / f"{name}.sqlite")

            if source_faiss.exists():
                shutil.copy2(source_faiss, backup_path / f"{name}.faiss")

            logger.info(f"Database {name} backed up to {backup_path}")

        except Exception as e:
            logger.error(f"Error backing up database {name}: {e}")
            raise

    def restore_database(self, name: str, backup_path: str):
        """Restore a database from backup"""
        try:
            import shutil

            backup_path = Path(backup_path)
            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))

            source_sqlite = backup_path / f"{name}.sqlite"
            source_faiss = backup_path / f"{name}.faiss"

            if not source_sqlite.exists():
                raise FileNotFoundError(f"Backup file {source_sqlite} not found")

            # Close database if it's currently open
            if name in self.databases:
                db, _ = self.databases[name]
                db.close()
                del self.databases[name]

            # Copy files
            if source_sqlite.exists():
                shutil.copy2(source_sqlite, db_path / f"{name}.sqlite")

            if source_faiss.exists():
                shutil.copy2(source_faiss, db_path / f"{name}.faiss")

            logger.info(f"Database {name} restored from {backup_path}")

        except Exception as e:
            logger.error(f"Error restoring database {name}: {e}")
            raise