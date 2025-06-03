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

"""
Updated database manager for LocalVectorDB with document-first architecture,
structured metadata support, and unified query interface.
"""

import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Union, Literal, Optional, Any

from localvectordb.core import MetadataFieldType
from localvectordb.exceptions import DatabaseNotFoundError
from localvectordb_server.config import DatabaseSettings, EmbeddingSettings

logger = logging.getLogger(__name__)

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

    def create_db(self, new_db_name,
                  metadata_schema: dict[str, MetadataFieldType] | None,
                  db_config: DatabaseSettings,
                  embedding_config: EmbeddingSettings) -> "LocalVectorDB":
        """Create a new database with specified configuration"""

        if new_db_name in self.databases:
            from werkzeug.exceptions import BadRequest
            raise BadRequest(f"Database '{new_db_name}' already exists")

        from localvectordb.database import LocalVectorDB

        # Create new database instance
        db = LocalVectorDB(
            name=new_db_name,
            base_path=Path(self.config.get("DB_ROOT_DIR", ".lvdb")),
            metadata_schema=metadata_schema,
            embedding_provider=embedding_config.provider,
            embedding_model=embedding_config.model,
            embedding_config=embedding_config.config,
            chunking_method=db_config.chunking_method,
            chunk_size=db_config.chunk_size,
            chunk_overlap=db_config.chunk_overlap,
            enable_gpu=db_config.enable_gpu,
            enable_fts=db_config.enable_fts,
            create_if_not_exists=True
        )

        self.databases[new_db_name] = (db, datetime.now())
        logger.info(f"Created new database: {new_db_name}")
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
