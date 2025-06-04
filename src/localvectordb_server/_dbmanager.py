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
Enhanced database manager for LocalVectorDB with improved error handling,
structured logging, performance monitoring, and recovery strategies.
"""

import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Union, Literal, Optional, Any, Tuple

from localvectordb.core import MetadataFieldType
from localvectordb.exceptions import DatabaseNotFoundError, DatabaseError
from localvectordb_server.config import DatabaseSettings, EmbeddingSettings
from localvectordb_server._logcfg import DatabaseLogger, log_performance
from localvectordb_server._error_handlers import APIError

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()


class DatabaseManager:
    """
    Enhanced database manager with error handling, monitoring, and recovery

    Features:
    - Comprehensive error handling and recovery
    - Performance monitoring and logging
    - Connection health checks
    - Automatic cleanup of inactive connections
    - Database statistics and health monitoring
    """

    def __init__(self, app):
        self.app = app
        self.config = app.config
        self.databases: Dict[str, Tuple["LocalVectorDB", datetime]] = {}
        self.lock = threading.RLock()
        self._shutdown_event = threading.Event()

        # Health monitoring
        self._last_health_check = datetime.now()
        self._health_check_interval = timedelta(minutes=5)

        # Error tracking
        self._error_counts = {}
        self._last_errors = {}

        # Ensure database directory exists
        db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
        try:
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Database directory ready: {db_path}")
        except Exception as e:
            logger.error(f"Failed to create database directory {db_path}: {e}")
            raise DatabaseError(f"Cannot create database directory: {e}")

        # Start background threads
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background monitoring and cleanup tasks"""
        try:
            # Cleanup thread
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="db-cleanup"
            )
            self._cleanup_thread.start()
            logger.info("Database cleanup thread started")

            # Health monitoring thread
            self._health_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name="db-health"
            )
            self._health_thread.start()
            logger.info("Database health monitoring thread started")

        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")

    def delete_db(self, db_name):
        try:
            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
            if not db_path.exists():
                logger.warning(f"Database directory does not exist: {db_path}")
                return False, "Database directory not found"

            db = self.databases.get(db_name)
            if db:
                db[0].close()

            db_logger.log_query(
                "delete_database_start",
                database_name=db_name
            )
            db_faiss_file = db_path / f"{db_name}.faiss"
            db_sqlite_file = db_path / f"{db_name}.sqlite"

            if db_sqlite_file.exists():
                os.remove(db_sqlite_file)
            else:
                logger.warning(f"Database {db_name} was not found.")
                return False, "Database file not found"

            # Faiss may not exist if the database has no records yet
            if db_faiss_file.exists():
                os.remove(db_faiss_file)

            db_logger.log_query(
                "delete_database_success",
                database_name=db_name
            )

            return True, None

        except Exception as e:
            logger.error(f"Error deleting database: {e}")
            db_logger.log_error("delete_database_failed", e)

            raise APIError(
                message=f"Failed to delete database: {str(e)}",
                error_code="DATABASE_LIST_FAILED",
                status_code=500,
                recoverable=False,
                details={"original_error": str(e)}
            )


    @log_performance("create_database")
    def create_db(
            self, new_db_name: str,
            metadata_schema: Optional[Dict[str, MetadataFieldType]],
            db_config: DatabaseSettings,
            embedding_config: EmbeddingSettings
            ) -> "LocalVectorDB":
        """
        Create a new database with comprehensive error handling and validation

        Parameters
        ----------
        new_db_name : str
            Name for the new database
        metadata_schema : Optional[Dict[str, MetadataFieldType]]
            Metadata schema configuration
        db_config : DatabaseSettings
            Database configuration
        embedding_config : EmbeddingSettings
            Embedding provider configuration

        Returns
        -------
        LocalVectorDB
            Created database instance

        Raises
        ------
        APIError
            If database creation fails
        """
        with self.lock:
            # Validate database name
            if not self._validate_database_name(new_db_name):
                raise APIError(
                    message=f"Invalid database name: '{new_db_name}'",
                    error_code="INVALID_DATABASE_NAME",
                    status_code=400,
                    recoverable=True
                )

            # Check if database already exists
            if new_db_name in self.databases:
                raise APIError(
                    message=f"Database '{new_db_name}' already exists in memory",
                    error_code="DATABASE_ALREADY_EXISTS",
                    status_code=409,
                    recoverable=True
                )

            # Check if database files exist on disk
            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
            db_sqlite_file = db_path / f"{new_db_name}.sqlite"
            if db_sqlite_file.exists():
                raise APIError(
                    message=f"Database '{new_db_name}' already exists on disk",
                    error_code="DATABASE_FILE_EXISTS",
                    status_code=409,
                    recoverable=True
                )

            try:
                from localvectordb.database import LocalVectorDB

                db_logger.log_query(
                    "create_database_start",
                    database_name=new_db_name,
                    embedding_provider=embedding_config.provider,
                    embedding_model=embedding_config.model,
                    chunk_size=db_config.chunk_size
                )

                # Create new database instance with comprehensive error handling
                db = LocalVectorDB(
                    name=new_db_name,
                    base_path=db_path,
                    metadata_schema=metadata_schema,
                    embedding_provider=embedding_config.provider,
                    embedding_model=embedding_config.model,
                    embedding_config=embedding_config.config,
                    chunking_method=db_config.chunking_method,
                    chunk_size=db_config.chunk_size,
                    chunk_overlap=db_config.chunk_overlap,
                    enable_gpu=db_config.enable_gpu,
                    enable_fts=db_config.enable_fts,
                    connection_pool_size=db_config.connection_pool_size,
                    create_if_not_exists=True
                )

                # Store in memory with access timestamp
                self.databases[new_db_name] = (db, datetime.now())

                db_logger.log_query(
                    "create_database_success",
                    database_name=new_db_name,
                    database_path=str(db_path),
                    stats=db.stats
                )

                logger.info(f"Successfully created database: {new_db_name}")
                return db

            except Exception as e:
                db_logger.log_error(
                    "create_database_failed",
                    e,
                    database_name=new_db_name,
                    embedding_provider=embedding_config.provider,
                    embedding_model=embedding_config.model
                )

                self._record_error(new_db_name, e)

                # Clean up any partially created files
                self._cleanup_failed_database(new_db_name)

                # Convert to appropriate API error
                if "not available" in str(e).lower() or "not found" in str(e).lower():
                    raise APIError(
                        message=f"Embedding model '{embedding_config.model}' not available",
                        error_code="EMBEDDING_MODEL_UNAVAILABLE",
                        status_code=503,
                        recoverable=True,
                        details={
                            "provider": embedding_config.provider,
                            "model": embedding_config.model
                        }
                    )
                else:
                    raise APIError(
                        message=f"Failed to create database: {str(e)}",
                        error_code="DATABASE_CREATION_FAILED",
                        status_code=500,
                        recoverable=False,
                        details={"original_error": str(e)}
                    )

    @log_performance("get_database")
    def get_db(self, name: str) -> "LocalVectorDB":
        """
        Get an existing database instance with enhanced error handling

        Parameters
        ----------
        name : str
            Database name

        Returns
        -------
        LocalVectorDB
            Database instance

        Raises
        ------
        APIError
            If database not found or cannot be loaded
        """
        with self.lock:
            # Check if database is already loaded
            if name in self.databases:
                db, _ = self.databases[name]

                # Check if database is still healthy
                try:
                    if self._check_database_health(db):
                        # Update last access time
                        self.databases[name] = (db, datetime.now())
                        return db
                    else:
                        # Database is unhealthy, remove from cache
                        logger.warning(f"Database {name} failed health check, reloading")
                        try:
                            db.close()
                        except Exception:
                            pass
                        del self.databases[name]

                except Exception as e:
                    logger.error(f"Health check failed for database {name}: {e}")
                    # Remove unhealthy database from cache
                    try:
                        db.close()
                    except Exception:
                        pass
                    del self.databases[name]

            # Load database from disk
            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
            db_sqlite_file = db_path / f"{name}.sqlite"

            if not db_sqlite_file.exists():
                raise APIError(
                    message=f"Database '{name}' not found",
                    error_code="DATABASE_NOT_FOUND",
                    status_code=404,
                    recoverable=True
                )

            try:
                from localvectordb.database import LocalVectorDB

                db_logger.log_query("load_database", database_name=name)

                # Load existing database
                db = LocalVectorDB(
                    name=name,
                    base_path=db_path,
                    create_if_not_exists=False
                )

                # Verify database is functional
                if not self._check_database_health(db):
                    raise DatabaseError(f"Database {name} failed post-load health check")

                self.databases[name] = (db, datetime.now())

                db_logger.log_query("load_database_success", database_name=name, stats=db.stats)
                logger.info(f"Successfully loaded database: {name}")

                return db

            except DatabaseNotFoundError:
                raise APIError(
                    message=f"Database '{name}' not found",
                    error_code="DATABASE_NOT_FOUND",
                    status_code=404,
                    recoverable=True
                )
            except Exception as e:
                db_logger.log_error("load_database_failed", e, database_name=name)
                self._record_error(name, e)

                raise APIError(
                    message=f"Failed to load database '{name}': {str(e)}",
                    error_code="DATABASE_LOAD_FAILED",
                    status_code=500,
                    recoverable=False,
                    details={"original_error": str(e)}
                )

    def list_databases(self) -> list[str]:
        """
        List all available databases with error handling

        Returns
        -------
        list[str]
            List of database names
        """
        try:
            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
            if not db_path.exists():
                logger.warning(f"Database directory does not exist: {db_path}")
                return []

            databases = []
            for db_file in db_path.iterdir():
                if db_file.suffix.lower() == ".sqlite":
                    databases.append(db_file.stem)

            logger.debug(f"Found {len(databases)} databases in {db_path}")
            return sorted(databases)

        except Exception as e:
            logger.error(f"Error listing databases: {e}")
            db_logger.log_error("list_databases_failed", e)

            raise APIError(
                message=f"Failed to list databases: {str(e)}",
                error_code="DATABASE_LIST_FAILED",
                status_code=500,
                recoverable=False,
                details={"original_error": str(e)}
            )

    def _validate_database_name(self, name: str) -> bool:
        """
        Validate database name

        Parameters
        ----------
        name : str
            Database name to validate

        Returns
        -------
        bool
            True if name is valid
        """
        if not name or not isinstance(name, str):
            return False

        # Check length
        if len(name) < 1 or len(name) > 64:
            return False

        # Check for invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
        if any(char in name for char in invalid_chars):
            return False

        # Check for reserved names
        reserved_names = ['con', 'prn', 'aux', 'nul']
        if name.lower() in reserved_names:
            return False

        return True

    def _check_database_health(self, db: "LocalVectorDB") -> bool:
        """
        Check if a database instance is healthy

        Parameters
        ----------
        db : LocalVectorDB
            Database instance to check

        Returns
        -------
        bool
            True if healthy, False otherwise
        """
        try:
            # Check if database is closed
            if db.closed:
                return False

            # Try a simple operation
            stats = db.stats

            # Basic sanity checks
            if stats['documents'] < 0 or stats['chunks'] < 0:
                return False

            return True

        except Exception as e:
            logger.debug(f"Database health check failed: {e}")
            return False

    def _cleanup_failed_database(self, db_name: str):
        """
        Clean up files from a failed database creation

        Parameters
        ----------
        db_name : str
            Name of the database to clean up
        """
        try:
            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
            files_to_remove = [
                db_path / f"{db_name}.sqlite",
                db_path / f"{db_name}.faiss"
            ]

            for file_path in files_to_remove:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Cleaned up file: {file_path}")

        except Exception as e:
            logger.error(f"Error cleaning up failed database {db_name}: {e}")

    def _record_error(self, db_name: str, error: Exception):
        """
        Record error for monitoring and recovery decisions

        Parameters
        ----------
        db_name : str
            Database name
        error : Exception
            Error that occurred
        """
        current_time = datetime.now()

        if db_name not in self._error_counts:
            self._error_counts[db_name] = 0

        self._error_counts[db_name] += 1
        self._last_errors[db_name] = {
            'error': str(error),
            'type': type(error).__name__,
            'timestamp': current_time
        }

    def _cleanup_loop(self):
        """Periodically check for and cleanup inactive databases"""
        logger.info("Database cleanup loop started")

        while not self._shutdown_event.wait(60):  # Check every minute
            try:
                self._cleanup_inactive()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def _cleanup_inactive(self):
        """Close inactive database connections"""
        now = datetime.now()
        timeout = timedelta(seconds=self.config.get("DB_TIMEOUT", 3600))  # Default 1 hour

        with self.lock:
            to_remove = []
            for name, (db, last_access) in self.databases.items():
                if now - last_access > timeout:
                    db_logger.log_query("cleanup_inactive_database", database_name=name,
                                        idle_time_seconds=(now - last_access).total_seconds())

                    try:
                        db.close()
                    except Exception as e:
                        logger.error(f"Error closing database {name}: {e}")

                    to_remove.append(name)

            for name in to_remove:
                del self.databases[name]
                logger.info(f"Cleaned up inactive database: {name}")

    def _health_check_loop(self):
        """Periodically check database health"""
        logger.info("Database health monitoring started")

        while not self._shutdown_event.wait(300):  # Check every 5 minutes
            try:
                self._perform_health_checks()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    def _perform_health_checks(self):
        """Perform health checks on active databases"""
        now = datetime.now()

        if now - self._last_health_check < self._health_check_interval:
            return

        self._last_health_check = now

        with self.lock:
            unhealthy_dbs = []

            for name, (db, last_access) in self.databases.items():
                try:
                    if not self._check_database_health(db):
                        unhealthy_dbs.append(name)
                        logger.warning(f"Database {name} failed health check")

                except Exception as e:
                    logger.error(f"Health check error for {name}: {e}")
                    unhealthy_dbs.append(name)

            # Remove unhealthy databases
            for name in unhealthy_dbs:
                try:
                    db, _ = self.databases[name]
                    db.close()
                except Exception:
                    pass

                del self.databases[name]
                db_logger.log_query("removed_unhealthy_database", database_name=name)

    def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get database manager statistics

        Returns
        -------
        Dict[str, Any]
            Manager statistics and health information
        """
        with self.lock:
            active_dbs = len(self.databases)
            total_dbs = len(self.list_databases())

            # Calculate uptime
            start_time = getattr(self, '_start_time', datetime.now())
            uptime = (datetime.now() - start_time).total_seconds()

            stats = {
                'active_databases': active_dbs,
                'total_databases': total_dbs,
                'uptime_seconds': uptime,
                'error_counts': dict(self._error_counts),
                'last_health_check': self._last_health_check.isoformat(),
                'background_threads': {
                    'cleanup_running': self._cleanup_thread.is_alive() if hasattr(self, '_cleanup_thread') else False,
                    'health_check_running': self._health_thread.is_alive() if hasattr(self, '_health_thread') else False
                }
            }

            # Add database-specific stats
            db_stats = {}
            for name, (db, last_access) in self.databases.items():
                try:
                    db_stats[name] = {
                        'last_access': last_access.isoformat(),
                        'idle_seconds': (datetime.now() - last_access).total_seconds(),
                        'stats': db.stats
                    }
                except Exception as e:
                    db_stats[name] = {'error': str(e)}

            stats['databases'] = db_stats

            return stats

    def close_all(self):
        """Close all database connections and stop background tasks"""
        logger.info("Shutting down database manager")

        # Signal background threads to stop
        self._shutdown_event.set()

        # Close all databases
        with self.lock:
            for name, (db, _) in self.databases.items():
                logger.info(f"Closing database: {name}")
                try:
                    db.close()
                except Exception as e:
                    logger.error(f"Error closing database {name}: {e}")

            self.databases.clear()

        # Wait for background threads to finish
        for thread_name, thread in [
            ('cleanup', getattr(self, '_cleanup_thread', None)),
            ('health', getattr(self, '_health_thread', None))
        ]:
            if thread and thread.is_alive():
                logger.info(f"Waiting for {thread_name} thread to finish")
                try:
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        logger.warning(f"{thread_name} thread did not finish gracefully")
                except Exception as e:
                    logger.error(f"Error joining {thread_name} thread: {e}")

        logger.info("Database manager shutdown complete")

    # Search and operation methods with enhanced error handling
    @log_performance("search_databases")
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
        """
        Search across multiple databases with enhanced error handling
        """
        if database_names is None:
            database_names = self.list_databases()

        results = {}

        db_logger.log_query("multi_database_search",
                            query_length=len(query),
                            database_count=len(database_names),
                            search_type=search_type)

        # Search each database with individual error handling
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

            except APIError as e:
                logger.warning(f"API error searching database {db_name}: {e.message}")
                results[db_name] = {"error": e.message, "error_code": e.error_code}
            except Exception as e:
                logger.error(f"Unexpected error searching database {db_name}: {e}")
                db_logger.log_error("search_database_failed", e, database_name=db_name)
                results[db_name] = {"error": f"Search failed: {str(e)}"}

        return results

    @log_performance("get_embeddings")
    def get_embeddings_for_model(
            self,
            query_texts: Union[str, list[str]],
            provider: str,
            model: str
    ) -> list[list[float]]:
        """
        Get embeddings with enhanced error handling
        """
        try:
            from localvectordb.embeddings import EmbeddingRegistry

            if isinstance(query_texts, str):
                query_texts = [query_texts]

            db_logger.log_query("get_embeddings",
                                provider=provider,
                                model=model,
                                text_count=len(query_texts))

            # Create embedding provider
            embedding_provider = EmbeddingRegistry.create_provider(provider, model)

            # Get embeddings
            embeddings = embedding_provider.embed_sync(query_texts)

            return embeddings.tolist()

        except Exception as e:
            db_logger.log_error("get_embeddings_failed", e, provider=provider, model=model)
            raise APIError(
                message=f"Failed to get embeddings: {str(e)}",
                error_code="EMBEDDING_GENERATION_FAILED",
                status_code=503,
                recoverable=True,
                details={"provider": provider, "model": model}
            )