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
Enhanced database manager for LocalVectorDB with multi-worker coordination.
Uses cachelib for shared database registry across workers.
"""

import atexit
import json
import logging
import os
import sys
import threading
import time
import types
import weakref
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from cachelib import DynamoDbCache, FileSystemCache, MemcachedCache, MongoDbCache, RedisCache, SimpleCache, UWSGICache

from localvectordb.core import MetadataFieldType
from localvectordb.exceptions import DatabaseError, DatabaseNotFoundError
from localvectordb_server._error_handlers import APIError
from localvectordb_server._logcfg import DatabaseLogger, log_performance
from localvectordb_server.config import DatabaseSettings, EmbeddingSettings

if TYPE_CHECKING:
    from localvectordb.database import LocalVectorDB

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()

# Global registry to track all DatabaseManager instances for cleanup
_active_managers: weakref.WeakSet["DatabaseManager"] = weakref.WeakSet()
_atexit_registered = False


def _cleanup_all_managers():
    """Emergency cleanup function called on process exit"""
    logger.info("Process exit detected, cleaning up all database managers")

    # Create a list copy since the WeakSet may change during iteration
    managers_to_close = list(_active_managers)

    for manager in managers_to_close:
        try:
            if hasattr(manager, "_shutdown_event") and not manager._shutdown_event.is_set():
                logger.info(
                    f"Emergency shutdown of database manager (worker: {getattr(manager, 'worker_id', 'unknown')})"
                )
                manager.close_all()
        except Exception as e:
            # Use print since logging might not work during shutdown
            print(f"Error during emergency database manager cleanup: {e}")

    logger.info("Emergency cleanup completed")


class DatabaseRegistryError(Exception):
    """Raised when database registry operations fail"""

    pass


class CrossPlatformFileLock:
    """Cross-platform file locking implementation"""

    def __init__(self, lock_file: Path, timeout: float = 30.0):
        self.lock_file = Path(lock_file)
        self.timeout = timeout
        self.file_handle: Optional[IO[str]] = None
        self._is_locked = False
        self.is_windows = sys.platform == "win32"

        # Platform-specific imports
        self.msvcrt: Optional[types.ModuleType] = None
        self.fcntl: Optional[types.ModuleType] = None
        if self.is_windows:
            import msvcrt

            self.msvcrt = msvcrt
        else:
            import fcntl

            self.fcntl = fcntl

        # Ensure lock directory exists
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the file lock"""
        if self._is_locked:
            return True

        try:
            self.file_handle = open(self.lock_file, "w")  # noqa: SIM115

            if self.is_windows:
                return self._acquire_windows(blocking)
            else:
                return self._acquire_unix(blocking)

        except Exception as e:
            logger.error(f"Failed to acquire lock {self.lock_file}: {e}")
            self._cleanup()
            if blocking:
                raise DatabaseRegistryError(f"Failed to acquire lock: {e}") from e
            return False

    def _acquire_windows(self, blocking: bool) -> bool:
        """Acquire lock on Windows using msvcrt"""
        assert self.msvcrt is not None
        assert self.file_handle is not None
        start_time = time.time()

        while True:
            try:
                self.msvcrt.locking(self.file_handle.fileno(), self.msvcrt.LK_NBLCK, 1)
                self._is_locked = True
                logger.debug(f"Acquired Windows lock: {self.lock_file}")
                return True

            except OSError:
                if not blocking:
                    return False

                if time.time() - start_time > self.timeout:
                    raise DatabaseRegistryError(f"Lock acquisition timeout after {self.timeout}s") from None

                time.sleep(0.1)

    def _acquire_unix(self, blocking: bool) -> bool:
        """Acquire lock on Unix systems using fcntl"""
        assert self.fcntl is not None
        assert self.file_handle is not None
        try:
            if blocking:
                start_time = time.time()
                while True:
                    try:
                        self.fcntl.flock(self.file_handle.fileno(), self.fcntl.LOCK_EX | self.fcntl.LOCK_NB)
                        self._is_locked = True
                        logger.debug(f"Acquired Unix lock: {self.lock_file}")
                        return True
                    except (OSError, IOError):
                        if time.time() - start_time > self.timeout:
                            raise DatabaseRegistryError(f"Lock acquisition timeout after {self.timeout}s") from None
                        time.sleep(0.1)
            else:
                self.fcntl.flock(self.file_handle.fileno(), self.fcntl.LOCK_EX | self.fcntl.LOCK_NB)
                self._is_locked = True
                logger.debug(f"Acquired Unix lock: {self.lock_file}")
                return True

        except (OSError, IOError) as e:
            if not blocking:
                return False
            raise DatabaseRegistryError(f"Failed to acquire Unix lock: {e}") from e

    def release(self):
        """Release the file lock"""
        if not self._is_locked or not self.file_handle:
            return

        try:
            if self.is_windows:
                assert self.msvcrt is not None
                self.msvcrt.locking(self.file_handle.fileno(), self.msvcrt.LK_UNLCK, 1)
            else:
                assert self.fcntl is not None
                self.fcntl.flock(self.file_handle.fileno(), self.fcntl.LOCK_UN)

            self._is_locked = False
            logger.debug(f"Released lock: {self.lock_file}")

        except Exception as e:
            logger.error(f"Error releasing lock {self.lock_file}: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up file handle"""
        if self.file_handle:
            try:
                self.file_handle.close()
            except Exception:
                pass
            finally:
                self.file_handle = None

    def __enter__(self):
        if not self.acquire():
            raise DatabaseRegistryError(f"Failed to acquire lock: {self.lock_file}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class DatabaseLockManager:
    """Cross-platform lock manager for FAISS index coordination"""

    def __init__(self, base_path: Path, lock_timeout: float = 30.0):
        self.base_path = Path(base_path)
        self.lock_dir = self.base_path / ".locks"
        self.lock_timeout = lock_timeout
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def acquire_read_lock(self, db_name: str):
        """Acquire a read lock for database operations"""
        lock_file = self.lock_dir / f"{db_name}.lock"

        logger.debug(f"Acquiring read lock for {db_name}")
        with CrossPlatformFileLock(lock_file, self.lock_timeout):
            logger.debug(f"Acquired read lock for {db_name}")
            try:
                yield
            finally:
                logger.debug(f"Released read lock for {db_name}")

    @contextmanager
    def acquire_write_lock(self, db_name: str):
        """Acquire a write lock for database operations"""
        lock_file = self.lock_dir / f"{db_name}.lock"

        logger.debug(f"Acquiring write lock for {db_name}")
        with CrossPlatformFileLock(lock_file, self.lock_timeout):
            logger.debug(f"Acquired write lock for {db_name}")
            try:
                yield
            finally:
                logger.debug(f"Released write lock for {db_name}")


class DatabaseRegistry:
    """Database registry implementation using cachelib backends"""

    def __init__(self, cache_instance):
        """
        Initialize with a cachelib cache instance

        Parameters
        ----------
        cache_instance : cachelib.BaseCache
            Configured cache instance (SimpleCache, RedisCache, etc.)
        """
        self.cache = cache_instance
        self.registry_key = "lvdb:registry"
        self.metadata_key_prefix = "lvdb:metadata:"

    def _get_registry_set(self) -> set:
        """Get the set of registered database names"""
        registry_data = self.cache.get(self.registry_key)
        if registry_data is None:
            return set()

        if isinstance(registry_data, str):
            # Handle JSON-serialized data
            try:
                return set(json.loads(registry_data))
            except (json.JSONDecodeError, TypeError):
                logger.warning("Corrupted registry data, resetting")
                return set()
        elif isinstance(registry_data, (list, set)):
            return set(registry_data)
        else:
            logger.warning(f"Unexpected registry data type: {type(registry_data)}")
            return set()

    def _save_registry_set(self, db_names: set) -> None:
        """Save the set of registered database names"""
        # Convert to list for JSON serialization compatibility
        registry_data = list(db_names)

        # Use JSON for cross-platform compatibility
        if hasattr(self.cache, "set"):
            # Direct set operation
            self.cache.set(self.registry_key, json.dumps(registry_data))
        else:
            # Fallback for caches that don't support set
            self.cache[self.registry_key] = json.dumps(registry_data)

    def register_database(self, name: str, metadata: Dict[str, Any]) -> None:
        """Register a database with metadata"""
        try:
            # Add to registry set
            registry = self._get_registry_set()
            registry.add(name)
            self._save_registry_set(registry)

            # Store metadata
            metadata_key = f"{self.metadata_key_prefix}{name}"
            metadata_json = json.dumps(metadata, default=str)  # default=str handles datetime objects

            if hasattr(self.cache, "set"):
                self.cache.set(metadata_key, metadata_json)
            else:
                self.cache[metadata_key] = metadata_json

            logger.info(f"Registered database '{name}' in shared registry")

        except Exception as e:
            logger.error(f"Failed to register database '{name}': {e}")
            raise DatabaseRegistryError(f"Failed to register database: {e}") from e

    def unregister_database(self, name: str) -> None:
        """Unregister a database"""
        try:
            # Remove from registry set
            registry = self._get_registry_set()
            registry.discard(name)
            self._save_registry_set(registry)

            # Remove metadata
            metadata_key = f"{self.metadata_key_prefix}{name}"
            if hasattr(self.cache, "delete"):
                self.cache.delete(metadata_key)
            else:
                try:
                    del self.cache[metadata_key]
                except KeyError:
                    pass

            logger.info(f"Unregistered database '{name}' from shared registry")

        except Exception as e:
            logger.error(f"Failed to unregister database '{name}': {e}")
            raise DatabaseRegistryError(f"Failed to unregister database: {e}") from e

    def list_databases(self) -> List[str]:
        """List all registered databases"""
        try:
            registry = self._get_registry_set()
            return sorted(list(registry))
        except Exception as e:
            logger.error(f"Failed to list databases: {e}")
            return []

    def get_database_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a database"""
        try:
            metadata_key = f"{self.metadata_key_prefix}{name}"
            metadata_json = self.cache.get(metadata_key)

            if metadata_json is None:
                return None

            if isinstance(metadata_json, str):
                result: Dict[str, Any] = json.loads(metadata_json)
                return result
            elif isinstance(metadata_json, dict):
                # Already deserialized
                return metadata_json
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to get metadata for database '{name}': {e}")
            return None

    def database_exists(self, name: str) -> bool:
        """Check if a database is registered"""
        try:
            registry = self._get_registry_set()
            return name in registry
        except Exception as e:
            logger.error(f"Failed to check existence of database '{name}': {e}")
            return False

    def update_database_metadata(self, name: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for an existing database"""
        try:
            if not self.database_exists(name):
                raise DatabaseRegistryError(f"Database '{name}' not registered")

            metadata_key = f"{self.metadata_key_prefix}{name}"
            metadata_json = json.dumps(metadata, default=str)

            if hasattr(self.cache, "set"):
                self.cache.set(metadata_key, metadata_json)
            else:
                self.cache[metadata_key] = metadata_json

            logger.debug(f"Updated metadata for database '{name}'")

        except Exception as e:
            logger.error(f"Failed to update metadata for database '{name}': {e}")
            raise DatabaseRegistryError(f"Failed to update metadata: {e}") from e


class DatabaseManager:
    """
    Enhanced database manager with multi-worker coordination, error handling,
    monitoring, and recovery

    Features:
    - Multi-worker coordination using cachelib registry
    - Cross-platform file locking for FAISS coordination
    - Comprehensive error handling and recovery
    - Performance monitoring and logging
    - Connection health checks
    - Automatic cleanup of inactive connections
    - Database statistics and health monitoring
    """

    def __init__(self, app):
        global _atexit_registered

        self.app = app
        self.config = app.config

        # Register this instance for emergency cleanup
        _active_managers.add(self)

        # Register global atexit handler (only once)
        if not _atexit_registered:
            atexit.register(_cleanup_all_managers)
            _atexit_registered = True
            logger.info("Registered atexit handler for database cleanup")

        # Initialize shared registry using cachelib
        self.registry = self._create_registry()

        # Initialize lock manager for FAISS coordination
        db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
        self.lock_manager = DatabaseLockManager(db_path)

        # Process-local database cache (existing functionality)
        self.databases: Dict[str, Tuple["LocalVectorDB", datetime]] = {}
        self.lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._shutdown_complete = False

        # Worker identification for coordination
        self.worker_id = f"worker-{os.getpid()}-{threading.get_ident()}"

        # Registry synchronization
        self._last_registry_sync = datetime.now(UTC)
        self._registry_sync_interval = timedelta(seconds=30)  # Sync every 30 seconds

        # Health monitoring
        self._last_health_check = datetime.now(UTC)
        self._health_check_interval = timedelta(minutes=5)

        # Error tracking
        self._error_counts = {}
        self._last_errors = {}
        self._start_time = datetime.now(UTC)

        # Ensure database directory exists
        try:
            db_path.mkdir(parents=True, exist_ok=True)
            # logger.info(f"Database directory ready: {db_path}")
        except Exception as e:
            # logger.error(f"Failed to create database directory {db_path}: {e}")
            raise DatabaseError(f"Cannot create database directory: {e}") from e

        # Start background services
        self._start_background_services()

        # Initial registry sync from filesystem
        self._sync_registry_from_filesystem()

    def _create_registry(self) -> DatabaseRegistry:
        """Create database registry using cachelib"""
        if self.app.config_obj.server.use_single_cache:
            from localvectordb_server._cache import cache

            return DatabaseRegistry(cache.cache)

        # Get registry configuration from server settings
        registry_type = self.app.config_obj.server.db_registry_type

        registry_settings = self.app.config_obj.server.db_registry_settings

        if registry_type == self.app.config_obj.server.cache_type and not registry_settings:
            registry_settings = self.app.config_obj.server.cache_settings

        registry_settings = registry_settings or {}
        logger.info(f"Initializing database registry with type: {registry_type}")

        registry_config_kwargs: Dict[str, Any] = {}
        for key, value in registry_settings.items():
            if isinstance(value, str) and value.startswith("$"):
                registry_config_kwargs[key] = os.getenv(value[1:])
            else:
                registry_config_kwargs[key] = value

        # Create cache instance based on type
        cache_instance: Union[
            SimpleCache, FileSystemCache, RedisCache, MemcachedCache, UWSGICache, DynamoDbCache, MongoDbCache
        ]
        if registry_type == "SimpleCache" or registry_type == "memory":
            cache_instance = SimpleCache(**registry_config_kwargs)
        elif registry_type == "FileSystemCache":
            registry_config_kwargs["cache_dir"] = registry_config_kwargs.get("cache_dir", "./.lvdb/registry_cache")
            cache_instance = FileSystemCache(**registry_config_kwargs)
        elif registry_type == "RedisCache":
            cache_instance = RedisCache(**registry_config_kwargs)
        elif registry_type == "MemcachedCache":
            cache_instance = MemcachedCache(**registry_config_kwargs)
        elif registry_type == "UWSGICache":
            cache_instance = UWSGICache(**registry_config_kwargs)
        elif registry_type == "DynamoDbCache":
            cache_instance = DynamoDbCache(**registry_config_kwargs)
        elif registry_type == "MongoDbCache":
            cache_instance = MongoDbCache(**registry_config_kwargs)
        else:
            logger.warning(f"Unknown registry type '{registry_type}', falling back to memory")
            cache_instance = SimpleCache()

        return DatabaseRegistry(cache_instance)

    def _start_background_services(self):
        """Start background monitoring and cleanup tasks"""
        try:
            # Registry sync thread
            self._sync_thread = threading.Thread(target=self._registry_sync_loop, daemon=True, name="db-registry-sync")
            self._sync_thread.start()

            # Cleanup thread
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True, name="db-cleanup")
            self._cleanup_thread.start()
            logger.info("Database cleanup thread started")

            # Health monitoring thread
            self._health_thread = threading.Thread(target=self._health_check_loop, daemon=True, name="db-health")
            self._health_thread.start()
            logger.info("Database health monitoring thread started")

        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")

    def _registry_sync_loop(self):
        """Background thread to sync registry periodically"""
        while not self._shutdown_event.wait(self._registry_sync_interval.total_seconds()):
            try:
                self._sync_registry_from_filesystem()
            except Exception as e:
                logger.error(f"Error in registry sync loop: {e}")

    def _sync_registry_from_filesystem(self):
        """Sync registry with actual filesystem state"""
        try:
            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
            if not db_path.exists():
                return

            # Get databases from filesystem
            fs_databases = set()
            for db_file in db_path.glob("*.sqlite"):
                fs_databases.add(db_file.stem)

            # Get databases from registry
            registry_databases = set(self.registry.list_databases())

            # Register new databases found on filesystem
            new_databases = fs_databases - registry_databases
            for db_name in new_databases:
                self._register_database_from_filesystem(db_name)

            # Unregister databases that no longer exist on filesystem
            missing_databases = registry_databases - fs_databases
            for db_name in missing_databases:
                logger.info(f"Database '{db_name}' no longer exists on filesystem, unregistering")
                self.registry.unregister_database(db_name)

            self._last_registry_sync = datetime.now(UTC)

        except Exception as e:
            logger.error(f"Failed to sync registry from filesystem: {e}")

    def _register_database_from_filesystem(self, db_name: str):
        """Register a database found on filesystem"""
        try:
            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
            sqlite_path = db_path / f"{db_name}.sqlite"
            faiss_path = db_path / f"{db_name}.faiss"

            # Try to extract metadata from database
            metadata = {
                "name": db_name,
                "created_at": datetime.fromtimestamp(sqlite_path.stat().st_ctime).isoformat(),
                "last_modified": datetime.fromtimestamp(sqlite_path.stat().st_mtime).isoformat(),
                "discovered_by": self.worker_id,
                "file_paths": {"sqlite": str(sqlite_path), "faiss": str(faiss_path) if faiss_path.exists() else None},
            }

            # Try to load database config
            try:
                from localvectordb.database import LocalVectorDB

                with self.lock_manager.acquire_read_lock(db_name):
                    temp_db = LocalVectorDB(name=db_name, base_path=db_path, create_if_not_exists=False)
                    stats = temp_db.get_stats()
                    metadata.update(
                        {
                            "embedding_model": stats.get("embedding_model"),
                            "embedding_provider": stats.get("embedding_provider"),
                            "embedding_dimension": stats.get("embedding_dimension"),
                            "chunk_size": stats.get("chunk_size"),
                            "chunking_method": stats.get("chunking_method"),
                            "chunk_overlap": stats.get("chunk_overlap"),
                            "documents": stats.get("documents", 0),
                            "chunks": stats.get("chunks", 0),
                        }
                    )
                    temp_db.close()
            except Exception as e:
                logger.warning(f"Could not extract metadata for database '{db_name}': {e}")

            self.registry.register_database(db_name, metadata)
            logger.info(f"Discovered and registered database '{db_name}'")

        except Exception as e:
            logger.error(f"Failed to register database '{db_name}' from filesystem: {e}")

    @log_performance("create_database")
    def create_db(
        self,
        new_db_name: str,
        metadata_schema: Optional[Dict[str, MetadataFieldType]],
        db_config: DatabaseSettings,
        embedding_config: EmbeddingSettings,
    ) -> "LocalVectorDB":
        """Create a new database with coordination and comprehensive error handling"""

        with self.lock:
            # Validate database name
            if not self._validate_database_name(new_db_name):
                raise APIError(
                    message=f"Invalid database name: '{new_db_name}'",
                    error_code="INVALID_DATABASE_NAME",
                    status_code=400,
                    recoverable=True,
                )

            # Check if database already exists in registry
            if self.registry.database_exists(new_db_name):
                raise APIError(
                    message=f"Database '{new_db_name}' already exists",
                    error_code="DATABASE_ALREADY_EXISTS",
                    status_code=409,
                    recoverable=True,
                )

            from localvectordb.database import LocalVectorDB

            # Create database with write lock
            with self.lock_manager.acquire_write_lock(new_db_name):
                # Double-check after acquiring lock
                if self.registry.database_exists(new_db_name):
                    raise APIError(
                        message=f"Database '{new_db_name}' already exists",
                        error_code="DATABASE_ALREADY_EXISTS",
                        status_code=409,
                        recoverable=True,
                    )

                try:
                    db_logger.log_query(
                        "create_database_start",
                        database_name=new_db_name,
                        embedding_provider=embedding_config.provider,
                        embedding_model=embedding_config.model,
                        chunk_size=db_config.chunk_size,
                    )

                    # Some of the config parameters need to be included in the `embedding_config` kwarg.
                    embedding_config_dict = dict(
                        timeout=embedding_config.timeout,
                        max_retries=embedding_config.max_retries,
                        base_url=embedding_config.base_url,
                        api_key=embedding_config.api_key,
                        **embedding_config.config,
                    )
                    if embedding_config.base_url:
                        embedding_config_dict["base_url"] = embedding_config.base_url
                    if embedding_config.api_key:
                        embedding_config_dict["api_key"] = embedding_config.api_key

                    # Create new database instance
                    db = LocalVectorDB(
                        name=new_db_name,
                        base_path=self.config.get("DB_ROOT_DIR", ".lvdb"),
                        metadata_schema=metadata_schema,
                        embedding_provider=embedding_config.provider,
                        embedding_model=embedding_config.model,
                        embedding_config=embedding_config_dict,
                        chunking_method=db_config.chunking_method,
                        chunk_size=db_config.chunk_size,
                        chunk_overlap=db_config.chunk_overlap,
                        enable_gpu=db_config.enable_gpu,
                        enable_fts=db_config.enable_fts,
                        connection_pool_size=db_config.connection_pool_size,
                        create_if_not_exists=True,
                        faiss_index_type=db_config.faiss_index_type,
                        faiss_index_hnsw_flat_neighbors=db_config.faiss_index_hnsw_flat_neighbors,
                        faiss_index_lsh_bits=db_config.faiss_index_lsh_bits,
                    )

                    # Register in shared registry
                    db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
                    metadata = {
                        "name": new_db_name,
                        "created_at": datetime.now(UTC).isoformat(),
                        "last_modified": datetime.now(UTC).isoformat(),
                        "created_by": self.worker_id,
                        "embedding_model": embedding_config.model,
                        "embedding_provider": embedding_config.provider,
                        "embedding_dimension": db.embedding_dimension,
                        "chunk_size": db_config.chunk_size,
                        "chunking_method": db_config.chunking_method,
                        "chunk_overlap": db_config.chunk_overlap,
                        "file_paths": {
                            "sqlite": str(db_path / f"{new_db_name}.sqlite"),
                            "faiss": str(db_path / f"{new_db_name}.faiss"),
                        },
                    }

                    self.registry.register_database(new_db_name, metadata)

                    # Cache locally
                    self.databases[new_db_name] = (db, datetime.now(UTC))

                    db_logger.log_query(
                        "create_database_success",
                        database_name=new_db_name,
                        database_path=str(db_path),
                        stats=db.get_stats(),
                    )

                    logger.info(f"Successfully created database: {new_db_name}")
                    return db

                except Exception as e:
                    db_logger.log_error(
                        "create_database_failed",
                        e,
                        database_name=new_db_name,
                        embedding_provider=embedding_config.provider,
                        embedding_model=embedding_config.model,
                    )

                    self._record_error(new_db_name, e)
                    self._cleanup_failed_database(new_db_name)

                    # Convert to appropriate API error
                    if "not available" in str(e).lower() or "not found" in str(e).lower():
                        raise APIError(
                            message=f"Embedding model '{embedding_config.model}' not available",
                            error_code="EMBEDDING_MODEL_UNAVAILABLE",
                            status_code=503,
                            recoverable=True,
                            details={"provider": embedding_config.provider, "model": embedding_config.model},
                        ) from e
                    else:
                        raise APIError(
                            message=f"Failed to create database: {str(e)}",
                            error_code="DATABASE_CREATION_FAILED",
                            status_code=500,
                            recoverable=False,
                            details={"original_error": str(e)},
                        ) from e

    @log_performance("get_database")
    def get_db(self, name: str) -> "LocalVectorDB":
        """Get an existing database instance with coordination and enhanced error handling"""

        with self.lock:
            # Check local cache first
            if name in self.databases:
                db, _ = self.databases[name]

                # Check if database is still healthy
                try:
                    if self._check_database_health(db):
                        # Update last access time
                        self.databases[name] = (db, datetime.now(UTC))
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

            # Check if database exists in registry
            if not self.registry.database_exists(name):
                # Try syncing from filesystem first
                self._sync_registry_from_filesystem()

                if not self.registry.database_exists(name):
                    raise APIError(
                        message=f"Database '{name}' not found",
                        error_code="DATABASE_NOT_FOUND",
                        status_code=404,
                        recoverable=True,
                    )

            # Load database with read lock
            from localvectordb.database import LocalVectorDB

            with self.lock_manager.acquire_read_lock(name):
                try:
                    db_logger.log_query("load_database", database_name=name)

                    db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
                    db = LocalVectorDB(name=name, base_path=db_path, create_if_not_exists=False)

                    # Verify database is functional
                    if not self._check_database_health(db):
                        raise DatabaseError(f"Database {name} failed post-load health check")

                    # Cache locally
                    self.databases[name] = (db, datetime.now(UTC))

                    db_logger.log_query("load_database_success", database_name=name, stats=db.get_stats())
                    logger.info(f"Successfully loaded database: {name}")
                    return db

                except DatabaseNotFoundError as e:
                    # Database was in registry but not on filesystem
                    logger.warning(f"Database '{name}' in registry but not found on filesystem")
                    self.registry.unregister_database(name)
                    raise APIError(
                        message=f"Database '{name}' not found",
                        error_code="DATABASE_NOT_FOUND",
                        status_code=404,
                        recoverable=True,
                    ) from e
                except Exception as e:
                    db_logger.log_error("load_database_failed", e, database_name=name)
                    self._record_error(name, e)

                    raise APIError(
                        message=f"Failed to load database '{name}': {str(e)}",
                        error_code="DATABASE_LOAD_FAILED",
                        status_code=500,
                        recoverable=False,
                        details={"original_error": str(e)},
                    ) from e

    def list_databases(self) -> List[str]:
        """List all available databases from shared registry with enhanced error handling"""
        try:
            # Ensure we have recent data
            if datetime.now(UTC) - self._last_registry_sync > self._registry_sync_interval:
                self._sync_registry_from_filesystem()

            result: List[str] = self.registry.list_databases()
            return result
        except Exception as e:
            logger.error(f"Failed to list databases from registry: {e}")
            db_logger.log_error("list_databases_failed", e)
            # Fallback to filesystem scan
            return self._fallback_list_databases()

    def _fallback_list_databases(self) -> List[str]:
        """Fallback method to list databases from filesystem"""
        try:
            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
            if not db_path.exists():
                return []
            return [d.stem for d in db_path.iterdir() if d.suffix.lower() == ".sqlite"]
        except Exception as e:
            logger.error(f"Error in fallback list databases: {e}")
            raise APIError(
                message=f"Failed to list databases: {str(e)}",
                error_code="DATABASE_LIST_FAILED",
                status_code=500,
                recoverable=False,
                details={"original_error": str(e)},
            ) from e

    def delete_database(self, name: str) -> bool:
        """Delete a database with coordination and enhanced error handling"""

        # Returning False for a non-existent database is intentional (idempotent delete).
        if not self.registry.database_exists(name):
            return False

        with self.lock_manager.acquire_write_lock(name):
            try:
                # Close database if it's cached locally
                if name in self.databases:
                    db, _ = self.databases[name]
                    db.close()
                    del self.databases[name]

                db_logger.log_query("delete_database_start", database_name=name)

                # Delete database files
                db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
                sqlite_file = db_path / f"{name}.sqlite"
                faiss_file = db_path / f"{name}.faiss"

                if sqlite_file.exists():
                    sqlite_file.unlink()
                if faiss_file.exists():
                    faiss_file.unlink()

                # Unregister from shared registry
                self.registry.unregister_database(name)

                db_logger.log_query("delete_database_success", database_name=name)
                logger.info(f"Deleted database: {name}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete database '{name}': {e}")
                db_logger.log_error("delete_database_failed", e, database_name=name)
                raise APIError(
                    message=f"Failed to delete database: {str(e)}",
                    error_code="DATABASE_DELETE_FAILED",
                    status_code=500,
                    recoverable=False,
                    details={"original_error": str(e)},
                ) from e

    @log_performance("search_databases")
    def search_databases(
        self,
        query: str,
        database_names: Optional[List[str]] = None,
        search_type: Literal["vector", "keyword", "hybrid"] = "vector",
        return_type: Literal["documents", "chunks"] = "documents",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        context_window: int = 2,
    ) -> Dict[str, Union[List, str]]:
        """Search across multiple databases with enhanced error handling"""

        if database_names is None:
            database_names = self.list_databases()

        results = {}

        db_logger.log_query(
            "multi_database_search",
            query_length=len(query),
            database_count=len(database_names or []),
            search_type=search_type,
        )

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
                    vector_weight=vector_weight,
                    context_window=context_window,
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
        self, query_texts: Union[str, List[str]], provider: str, model: str
    ) -> List[List[float]]:
        """Get embeddings with enhanced error handling"""
        try:
            from localvectordb.embeddings import EmbeddingRegistry

            if isinstance(query_texts, str):
                query_texts = [query_texts]

            db_logger.log_query("get_embeddings", provider=provider, model=model, text_count=len(query_texts))

            # Create embedding provider
            embedding_provider = EmbeddingRegistry.create_provider(provider, model)

            # Get embeddings
            embeddings = embedding_provider.embed_sync(query_texts)

            result: List[List[float]] = embeddings.tolist()
            return result

        except Exception as e:
            db_logger.log_error("get_embeddings_failed", e, provider=provider, model=model)
            raise APIError(
                message=f"Failed to get embeddings: {str(e)}",
                error_code="EMBEDDING_GENERATION_FAILED",
                status_code=503,
                recoverable=True,
                details={"provider": provider, "model": model},
            ) from e

    def _validate_database_name(self, name: str) -> bool:
        """
        Validate database name for security and filesystem compatibility.

        Security checks include:
        - Path traversal prevention (.., null bytes, control characters)
        - Invalid filesystem characters
        - Windows reserved names
        - Unicode normalization attacks

        Parameters
        ----------
        name : str
            Database name to validate

        Returns
        -------
        bool
            True if name is valid, False otherwise
        """
        if not name or not isinstance(name, str):
            return False

        # Check length
        if len(name) < 1 or len(name) > 64:
            return False

        # Security: Check for null bytes (path manipulation)
        if "\x00" in name:
            return False

        # Security: Check for control characters (ASCII 0-31)
        if any(ord(char) < 32 for char in name):
            return False

        # Security: Check for path traversal sequences
        if ".." in name:
            return False

        # Security: Check for hidden file indicators
        if name.startswith("."):
            return False

        # Check for invalid characters (filesystem and path separators)
        invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]
        if any(char in name for char in invalid_chars):
            return False

        # Security: Check for Unicode path separators and lookalikes
        # U+2215 DIVISION SLASH, U+2044 FRACTION SLASH, U+29F8 BIG SOLIDUS
        # U+FF0F FULLWIDTH SOLIDUS, U+FF3C FULLWIDTH REVERSE SOLIDUS
        unicode_path_chars = ["\u2215", "\u2044", "\u29f8", "\uff0f", "\uff3c"]
        if any(char in name for char in unicode_path_chars):
            return False

        # Security: Check for Unicode homoglyphs that could confuse users
        # (e.g., Cyrillic 'а' vs Latin 'a' in reserved names)
        # Normalize to NFKC and recheck
        import unicodedata

        normalized_name = unicodedata.normalize("NFKC", name)
        if normalized_name != name:
            # Name contains characters that normalize differently
            # This could indicate Unicode tricks
            return False

        # Check for reserved names (Windows compatibility)
        reserved_names = [
            "con",
            "prn",
            "aux",
            "nul",
            "com1",
            "com2",
            "com3",
            "com4",
            "com5",
            "com6",
            "com7",
            "com8",
            "com9",
            "lpt1",
            "lpt2",
            "lpt3",
            "lpt4",
            "lpt5",
            "lpt6",
            "lpt7",
            "lpt8",
            "lpt9",
        ]
        # Check base name (before any extension)
        base_name = name.lower().split(".")[0]
        if base_name in reserved_names:
            return False

        # Ensure name contains only safe characters: alphanumeric, underscore, hyphen
        # This is a whitelist approach for defense in depth
        import re

        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", name):
            return False

        return True

    def _check_database_health(self, db: "LocalVectorDB") -> bool:
        """Check if a database instance is healthy"""
        try:
            # Check if database is closed
            if db.closed:
                return False

            # Try a simple operation
            stats = db.get_stats()

            # Basic sanity checks
            if stats["documents"] < 0 or stats["chunks"] < 0:
                return False

            return True

        except Exception as e:
            logger.debug(f"Database health check failed: {e}")
            return False

    def _cleanup_failed_database(self, db_name: str):
        """Clean up files from a failed database creation"""
        try:
            db_path = Path(self.config.get("DB_ROOT_DIR", ".lvdb"))
            files_to_remove = [db_path / f"{db_name}.sqlite", db_path / f"{db_name}.faiss"]

            for file_path in files_to_remove:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Cleaned up file: {file_path}")

        except Exception as e:
            logger.error(f"Error cleaning up failed database {db_name}: {e}")

    def _record_error(self, db_name: str, error: Exception):
        """Record error for monitoring and recovery decisions"""
        current_time = datetime.now(UTC)

        if db_name not in self._error_counts:
            self._error_counts[db_name] = 0

        self._error_counts[db_name] += 1
        self._last_errors[db_name] = {"error": str(error), "type": type(error).__name__, "timestamp": current_time}

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
        now = datetime.now(UTC)
        timeout = timedelta(seconds=self.config.get("DB_TIMEOUT", 3600))  # Default 1 hour

        with self.lock:
            to_remove = []
            for name, (db, last_access) in self.databases.items():
                if now - last_access > timeout:
                    db_logger.log_query(
                        "cleanup_inactive_database",
                        database_name=name,
                        idle_time_seconds=(now - last_access).total_seconds(),
                    )

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
        now = datetime.now(UTC)

        if now - self._last_health_check < self._health_check_interval:
            return

        self._last_health_check = now

        with self.lock:
            unhealthy_dbs = []

            for name, (db, _last_access) in self.databases.items():
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
        """Get database manager statistics"""
        with self.lock:
            active_dbs = len(self.databases)
            total_dbs = len(self.list_databases())

            # Calculate uptime
            uptime = (datetime.now(UTC) - self._start_time).total_seconds()

            stats = {
                "active_databases": active_dbs,
                "total_databases": total_dbs,
                "uptime_seconds": uptime,
                "worker_id": self.worker_id,
                "registry_type": getattr(self.app.config_obj.server, "db_registry_type", "memory"),
                "error_counts": dict(self._error_counts),
                "last_health_check": self._last_health_check.isoformat(),
                "last_registry_sync": self._last_registry_sync.isoformat(),
                "background_threads": {
                    "cleanup_running": self._cleanup_thread.is_alive() if hasattr(self, "_cleanup_thread") else False,
                    "health_check_running": (
                        self._health_thread.is_alive() if hasattr(self, "_health_thread") else False
                    ),
                    "registry_sync_running": self._sync_thread.is_alive() if hasattr(self, "_sync_thread") else False,
                },
            }

            # Add database-specific stats
            db_stats = {}
            for name, (db, last_access) in self.databases.items():
                try:
                    db_stats[name] = {
                        "last_access": last_access.isoformat(),
                        "idle_seconds": (datetime.now(UTC) - last_access).total_seconds(),
                        "stats": db.get_stats(),
                    }
                except Exception as e:
                    db_stats[name] = {"error": str(e)}

            stats["databases"] = db_stats

            return stats

    def close_all(self):
        """Close all database connections and stop background tasks"""
        if self._shutdown_complete:
            return

        logger.info(f"Shutting down database manager (worker: {self.worker_id})")

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
            ("cleanup", getattr(self, "_cleanup_thread", None)),
            ("health", getattr(self, "_health_thread", None)),
            ("registry-sync", getattr(self, "_sync_thread", None)),
        ]:
            if thread and thread.is_alive():
                logger.info(f"Waiting for {thread_name} thread to finish")
                try:
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        logger.warning(f"{thread_name} thread did not finish gracefully")
                except Exception as e:
                    logger.error(f"Error joining {thread_name} thread: {e}")

        self._shutdown_complete = True
        logger.info(f"Database manager shutdown complete (worker: {self.worker_id})")

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection"""
        try:
            if not self._shutdown_complete:
                logger.debug(
                    "DatabaseManager garbage collected, ensuring cleanup "
                    f"(worker: {getattr(self, 'worker_id', 'unknown')})"
                )
                self.close_all()
        except Exception as e:
            # Use print since logging might not work during GC
            print(f"Error in DatabaseManager destructor: {e}")
