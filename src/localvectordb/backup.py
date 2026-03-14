# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/backup.py

"""Backup, restore, and recovery system for LocalVectorDB.

This module provides comprehensive backup and recovery capabilities for LocalVectorDB,
including full backups, incremental backups, and point-in-time recovery. The system
leverages SQLite's built-in backup API and FAISS's save/load functionality.

Classes:
    BackupManager: Core backup and restore functionality
    BackupMetadata: Backup metadata structure
    BackupConfig: Configuration for backup operations
"""

import gc
import hashlib
import json
import logging
import shutil
import sqlite3
import tarfile
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np

from localvectordb.versioning import VersionManager

from .database._faiss_utils import get_faiss_external_ids
from .utils import parse_iso8601

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Backup type enumeration."""
    FULL = "full"
    INCREMENTAL = "incremental"


class CompressionAlgorithm(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    BZIP = "bzip2"


class BackupMetadata:
    """
    Metadata for backup files.

    Contains information about the backup including version, timestamp,
    checksums, and configuration details needed for restoration.

    Parameters
    ----------
    backup_id : str
        Unique identifier for the backup
    backup_type : BackupType
        Type of backup (full or incremental)
    database_name : str
        Name of the source database
    database_version : str
        Version of the source database schema
    created_at : datetime
        Timestamp when backup was created
    file_paths : Dict[str, str]
        Mapping of component names to file paths in the backup
    checksums : Dict[str, str]
        SHA-256 checksums for each component
    compression_algorithm : CompressionAlgorithm
        Compression algorithm used
    size_bytes : int
        Total size of the backup in bytes
    parent_backup_id : str, optional
        ID of parent backup (for incremental backups)
    metadata : Dict[str, Any], optional
        Additional metadata
    archive_checksum : str, optional
        SHA-256 checksum of the entire archive file for integrity verification
    manifest_checksum : str, optional
        SHA-256 checksum of the manifest.json for tampering detection
    """

    def __init__(
            self,
            backup_id: str,
            backup_type: BackupType,
            database_name: str,
            database_version: str,
            created_at: datetime,
            file_paths: Dict[str, str],
            checksums: Dict[str, str],
            compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
            size_bytes: int = 0,
            parent_backup_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            archive_checksum: Optional[str] = None,
            manifest_checksum: Optional[str] = None
    ):
        self.backup_id = backup_id
        self.backup_type = backup_type
        self.database_name = database_name
        self.database_version = database_version
        self.created_at = created_at
        self.file_paths = file_paths
        self.checksums = checksums
        self.compression_algorithm = compression_algorithm
        self.size_bytes = size_bytes
        self.parent_backup_id = parent_backup_id
        self.metadata = metadata or {}
        self.archive_checksum = archive_checksum
        self.manifest_checksum = manifest_checksum

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            'backup_id': self.backup_id,
            'backup_type': self.backup_type.value,
            'database_name': self.database_name,
            'database_version': self.database_version,
            'created_at': self.created_at.isoformat(),
            'file_paths': self.file_paths,
            'checksums': self.checksums,
            'compression_algorithm': self.compression_algorithm.value,
            'size_bytes': self.size_bytes,
            'parent_backup_id': self.parent_backup_id,
            'metadata': self.metadata,
            'archive_checksum': self.archive_checksum,
            'manifest_checksum': self.manifest_checksum
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupMetadata':
        """Create BackupMetadata from dictionary."""
        return cls(
            backup_id=data['backup_id'],
            backup_type=BackupType(data['backup_type']),
            database_name=data['database_name'],
            database_version=data['database_version'],
            created_at=parse_iso8601(data['created_at']),
            file_paths=data['file_paths'],
            checksums=data['checksums'],
            compression_algorithm=CompressionAlgorithm(data['compression_algorithm']),
            size_bytes=data['size_bytes'],
            parent_backup_id=data.get('parent_backup_id'),
            metadata=data.get('metadata', {}),
            archive_checksum=data.get('archive_checksum'),  # Backward compatibility
            manifest_checksum=data.get('manifest_checksum')  # Backward compatibility
        )


class BackupConfig:
    """
    Configuration for backup operations.

    Parameters
    ----------
    backup_location : Union[str, Path]
        Directory to store backup files
    compression_algorithm : CompressionAlgorithm
        Compression algorithm to use
    verify_integrity : bool
        Whether to verify backup integrity after creation
    retention_days : int
        Number of days to retain backups
    max_backup_size_gb : float
        Maximum backup size in GB (0 = unlimited)
    include_faiss_index : bool
        Whether to include FAISS index in backups
    """

    def __init__(
            self,
            backup_location: Union[str, Path] = "./backups",
            compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
            verify_integrity: bool = True,
            retention_days: int = 30,
            max_backup_size_gb: float = 0.0,
            include_faiss_index: bool = True
    ):
        self.backup_location: Path = Path(backup_location)
        self.compression_algorithm = compression_algorithm
        self.verify_integrity = verify_integrity
        self.retention_days = retention_days
        self.max_backup_size_gb = max_backup_size_gb
        self.include_faiss_index = include_faiss_index

        # Ensure backup directory exists
        self.backup_location.mkdir(parents=True, exist_ok=True)


class BackupManager:
    """
    Main backup and restore manager for LocalVectorDB.

    Provides comprehensive backup and recovery capabilities including full backups,
    incremental backups, and point-in-time recovery using SQLite's backup API
    and FAISS index management.

    Parameters
    ----------
    database_path : Union[str, Path]
        Path to the LocalVectorDB database file
    faiss_index_path : Union[str, Path], optional
        Path to the FAISS index file. If None, inferred from database_path.
    config : BackupConfig, optional
        Backup configuration. If None, uses default configuration.

    Examples
    --------
    Create a full backup::

        manager = BackupManager("/path/to/mydb.sqlite")
        backup_id = manager.create_backup(BackupType.FULL)
        print(f"Backup created: {backup_id}")

    Restore from backup::

        manager.restore_backup(backup_id, "/path/to/restore/location")

    List available backups::

        backups = manager.list_backups()
        for backup in backups:
            print(f"{backup.backup_id}: {backup.created_at}")
    """

    def __init__(
            self,
            database_path: Union[str, Path],
            faiss_index_path: Optional[Union[str, Path]] = None,
            config: Optional[BackupConfig] = None
    ):
        self.database_path = Path(database_path)

        if faiss_index_path is None:
            # Infer FAISS index path from database path
            self.faiss_index_path = self.database_path.with_suffix('.faiss')
        else:
            self.faiss_index_path = Path(faiss_index_path)

        self.config = config or BackupConfig()
        self.database_name = self.database_path.stem

        # Initialize version manager
        self.version_manager = VersionManager(self.database_path)

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _create_backup_manifest(
            self,
            backup_id: str,
            backup_type: BackupType,
            temp_dir: Path,
            parent_backup_id: Optional[str] = None
    ) -> BackupMetadata:
        """Create backup manifest with metadata and checksums."""

        # Get database version
        db_version = self.version_manager.get_database_version()

        # Calculate checksums for all files in temp directory
        file_paths = {}
        checksums = {}
        total_size = 0

        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                relative_path = file_path.name
                file_paths[relative_path] = relative_path
                checksums[relative_path] = self._calculate_file_checksum(file_path)
                total_size += file_path.stat().st_size

        # Create metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            database_name=self.database_name,
            database_version=str(db_version),
            created_at=datetime.now(UTC),
            file_paths=file_paths,
            checksums=checksums,
            compression_algorithm=self.config.compression_algorithm,
            size_bytes=total_size,
            parent_backup_id=parent_backup_id,
            metadata={
                'original_db_path': str(self.database_path),
                'original_faiss_path': str(self.faiss_index_path),
                'faiss_included': self.config.include_faiss_index and self.faiss_index_path.exists(),
                'sqlite_pragmas': self._get_current_pragma_settings()
            }
        )

        return metadata

    def _generate_document_manifest(self, db_path: Path) -> Dict[str, Any]:
        """
        Generate comprehensive document/chunk manifest for fast diffing.

        Creates a detailed manifest containing all documents and their chunks with
        content hashes and FAISS IDs. This enables fast incremental backup diffing
        without requiring parent backup extraction.

        Parameters
        ----------
        db_path : Path
            Path to the SQLite database file

        Returns
        -------
        Dict[str, Any]
            Document manifest containing:
            - documents: Dict mapping doc_id to document info including:
              - content_hash: SHA-256 hash of document content
              - updated_at: Last modification timestamp
              - chunks: Dict mapping chunk_index to chunk info including:
                - content_hash: SHA-256 hash of chunk content
                - faiss_id: FAISS vector ID (may be None)
                - start_pos, end_pos: Character positions in document
        """
        manifest = {'documents': {}}

        try:
            with sqlite3.connect(db_path) as conn:
                # Get all documents with their metadata
                doc_cursor = conn.execute("""
                    SELECT id, content_hash, updated_at
                    FROM documents
                    ORDER BY id
                """)

                for doc_id, content_hash, updated_at in doc_cursor:
                    manifest['documents'][doc_id] = {
                        'content_hash': content_hash,
                        'updated_at': updated_at,
                        'chunks': {}
                    }

                    # Get all chunks for this document
                    chunk_cursor = conn.execute("""
                        SELECT chunk_index, content_hash, faiss_id, start_pos, end_pos
                        FROM chunks
                        WHERE document_id = ?
                        ORDER BY chunk_index
                    """, (doc_id,))

                    for chunk_index, chunk_hash, faiss_id, start_pos, end_pos in chunk_cursor:
                        manifest['documents'][doc_id]['chunks'][chunk_index] = {
                            'content_hash': chunk_hash,
                            'faiss_id': faiss_id,
                            'start_pos': start_pos,
                            'end_pos': end_pos
                        }

        except Exception as e:
            logger.error(f"Failed to generate document manifest: {e}")
            raise

        return manifest

    def _create_backup_archive(self, metadata: BackupMetadata, temp_dir: Path) -> Path:
        """Create compressed backup archive from temporary directory."""

        # Generate backup filename
        timestamp = metadata.created_at.strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{self.database_name}_backup_{timestamp}_{metadata.backup_id[:8]}.lvdb-backup"
        backup_path = self.config.backup_location / backup_filename

        # Calculate manifest checksum (without the checksum fields themselves)
        temp_manifest_data = metadata.to_dict()
        temp_manifest_data['archive_checksum'] = None  # Will be calculated after archive creation
        temp_manifest_data['manifest_checksum'] = None  # Will be calculated now

        # Calculate manifest checksum from the normalized JSON
        manifest_json = json.dumps(temp_manifest_data, sort_keys=True, separators=(',', ':'))
        manifest_checksum = hashlib.sha256(manifest_json.encode('utf-8')).hexdigest()

        # Update metadata with manifest checksum (but NOT archive_checksum to avoid circularity)
        metadata.manifest_checksum = manifest_checksum

        # Write final manifest to temp directory (without archive_checksum)
        manifest_path = temp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            # Create manifest data without archive_checksum to avoid circular dependency
            manifest_data_for_archive = metadata.to_dict()
            manifest_data_for_archive['archive_checksum'] = None
            json.dump(manifest_data_for_archive, f, indent=2)

        write_mode = {
            CompressionAlgorithm.GZIP: "w:gz",
            CompressionAlgorithm.LZMA: "w:xz",
            CompressionAlgorithm.BZIP: "w:bz2",
            CompressionAlgorithm.NONE: "w"
        }.get(self.config.compression_algorithm, "w")



        with tarfile.open(backup_path, write_mode) as tar:
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    tar.add(file_path, arcname=file_path.name)

        # Calculate archive checksum for integrity verification
        archive_checksum = self._calculate_file_checksum(backup_path)
        metadata.archive_checksum = archive_checksum

        # Write sidecar checksum file for verification
        sidecar_path = backup_path.with_suffix(backup_path.suffix + '.sha256')
        with open(sidecar_path, 'w') as f:
            f.write(f"{archive_checksum}  {backup_path.name}\n")

        logger.info(f"Created backup archive: {backup_path}")
        logger.info(f"Created sidecar checksum: {sidecar_path}")
        logger.debug(f"Archive checksum: {archive_checksum}")
        return backup_path

    def create_backup(
            self,
            backup_type: BackupType = BackupType.FULL,
            parent_backup_id: Optional[str] = None,
            backup_id: Optional[str] = None
    ) -> str:
        """
        Create a new backup of the database.

        Parameters
        ----------
        backup_type : BackupType
            Type of backup to create
        parent_backup_id : str, optional
            ID of parent backup (required for incremental backups)
        backup_id : str, optional
            Custom backup ID. If None, generates a UUID.

        Returns
        -------
        str
            Unique backup ID

        Raises
        ------
        FileNotFoundError
            If database file doesn't exist
        ValueError
            If incremental backup requested without parent ID
        """

        if not self.database_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.database_path}")

        if backup_type == BackupType.INCREMENTAL and not parent_backup_id:
            raise ValueError("Incremental backup requires parent_backup_id")

        if backup_id is None:
            backup_id = str(uuid.uuid4())

        logger.info(f"Creating {backup_type.value} backup with ID: {backup_id}")

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            try:
                # Backup SQLite database using SQLite's backup API
                self._backup_sqlite_database(temp_dir)

                # Backup FAISS index if configured and exists
                if self.config.include_faiss_index and self.faiss_index_path.exists():
                    self._backup_faiss_index(temp_dir)

                # Create manifest
                metadata = self._create_backup_manifest(
                    backup_id, backup_type, temp_dir, parent_backup_id
                )

                # Add document IDs to metadata for optimization (helps detect deletions in future incremental backups)
                try:
                    with sqlite3.connect(self.database_path) as conn:
                        cursor = conn.execute("SELECT id FROM documents")
                        document_ids = [row[0] for row in cursor.fetchall()]
                        metadata.metadata['document_ids'] = document_ids
                        logger.debug(f"Added {len(document_ids)} document IDs to backup metadata")
                except Exception as e:
                    logger.warning(f"Could not add document IDs to metadata: {e}")

                # Add comprehensive document manifest for fast diffing without extraction
                try:
                    metadata.metadata['document_manifest'] = self._generate_document_manifest(self.database_path)
                    logger.debug("Added document manifest to backup metadata")
                except Exception as e:
                    logger.warning(f"Could not add document manifest to metadata: {e}")

                # Create compressed archive
                backup_path = self._create_backup_archive(metadata, temp_dir)

                # Record backup in database
                self._record_backup_in_database(metadata, backup_path)

                # Verify integrity if configured
                if self.config.verify_integrity:
                    self._verify_backup_integrity(backup_path)

                logger.info(f"Backup completed successfully: {backup_id}")
                return backup_id

            except Exception as e:
                logger.error(f"Backup failed: {e}")
                raise

    def _backup_sqlite_database(self, temp_dir: Path) -> None:
        """Backup SQLite database using SQLite's backup API with optimization pragmas."""
        backup_db_path = temp_dir / f"{self.database_name}.sqlite"

        # NOTE: On Windows, sqlite3 context managers don't properly release file handles,
        # causing PermissionError when cleaning up temp directories. We use explicit
        # close() calls and gc.collect() to ensure handles are released.
        source_conn = None
        backup_conn = None
        try:
            source_conn = sqlite3.connect(self.database_path)
            backup_conn = sqlite3.connect(backup_db_path)

            # Store original pragmas from source
            original_pragmas = {}
            pragma_queries = [
                "PRAGMA synchronous", "PRAGMA journal_mode", "PRAGMA cache_size",
                "PRAGMA wal_autocheckpoint", "PRAGMA mmap_size"
            ]

            for pragma_query in pragma_queries:
                try:
                    result = source_conn.execute(pragma_query).fetchone()
                    if result:
                        key = pragma_query.split()[-1]  # Extract pragma name
                        original_pragmas[key] = result[0]
                except sqlite3.Error:
                    pass  # Ignore if pragma not supported

            # Before backup, ensure WAL is checkpointed
            try:
                source_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.debug("WAL checkpointed before backup")
            except sqlite3.Error as e:
                logger.debug(f"WAL checkpoint failed: {e}")

            # Apply backup-optimized pragmas temporarily
            backup_pragmas = {
                "synchronous": "OFF",  # Faster backup (temporary)
                "cache_size": -131072,  # 128MB cache for backup
            }

            for key, value in backup_pragmas.items():
                try:
                    source_conn.execute(f"PRAGMA {key} = {value}")
                except sqlite3.Error:
                    pass  # Best effort

            # Perform the backup
            source_conn.backup(backup_conn)

            logger.debug(f"SQLite database backed up to: {backup_db_path}")

            # Restore original pragmas
            for key, value in original_pragmas.items():
                try:
                    if isinstance(value, str):
                        source_conn.execute(f"PRAGMA {key} = '{value}'")
                    else:
                        source_conn.execute(f"PRAGMA {key} = {value}")
                except sqlite3.Error:
                    pass  # Best effort restoration
        finally:
            # Explicitly close connections to release file handles on Windows
            if backup_conn is not None:
                backup_conn.close()
            if source_conn is not None:
                source_conn.close()
            # Force garbage collection to release any lingering file handles
            gc.collect()

    def _get_current_pragma_settings(self) -> Dict[str, Any]:
        """Get current pragma settings from the database for backup metadata."""
        pragma_settings = {}

        try:
            # Use context manager to ensure connection is always closed
            with sqlite3.connect(self.database_path) as conn:
                # Query common pragma settings
                pragma_queries = [
                    ("synchronous", "PRAGMA synchronous"),
                    ("journal_mode", "PRAGMA journal_mode"),
                    ("cache_size", "PRAGMA cache_size"),
                    ("wal_autocheckpoint", "PRAGMA wal_autocheckpoint"),
                    ("mmap_size", "PRAGMA mmap_size"),
                    ("temp_store", "PRAGMA temp_store"),
                    ("foreign_keys", "PRAGMA foreign_keys"),
                    ("busy_timeout", "PRAGMA busy_timeout")
                ]

                for pragma_name, pragma_query in pragma_queries:
                    try:
                        result = conn.execute(pragma_query).fetchone()
                        if result:
                            pragma_settings[pragma_name] = result[0]
                    except sqlite3.Error:
                        pass  # Skip unsupported pragmas

                # Also try to get profile and overrides from config table if it exists
                try:
                    config_result = conn.execute(
                        "SELECT key, value FROM config WHERE key IN ('sqlite_profile', 'sqlite_pragma_overrides')"
                    ).fetchall()

                    for key, value in config_result:
                        if key == 'sqlite_profile':
                            pragma_settings['profile'] = value
                        elif key == 'sqlite_pragma_overrides':
                            try:
                                pragma_settings['overrides'] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                pass
                except sqlite3.Error:
                    pass  # Config table may not exist

        except Exception as e:
            logger.debug(f"Failed to get pragma settings: {e}")

        return pragma_settings

    def _backup_faiss_index(self, temp_dir: Path) -> None:
        """Backup FAISS index file."""
        if self.faiss_index_path.exists():
            backup_faiss_path = temp_dir / f"{self.database_name}.faiss"
            shutil.copy2(self.faiss_index_path, backup_faiss_path)
            logger.debug(f"FAISS index backed up to: {backup_faiss_path}")

    def _record_backup_in_database(self, metadata: BackupMetadata, backup_path: Path) -> None:
        """Record backup metadata in the database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT INTO backup_log
                    (id, backup_type, created_at, database_version, file_path,
                     checksum, parent_backup_id, metadata, size_bytes, compression_algorithm)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.backup_id,
                    metadata.backup_type.value,
                    metadata.created_at.isoformat(),
                    metadata.database_version,
                    str(backup_path),
                    metadata.archive_checksum or self._calculate_file_checksum(backup_path),
                    metadata.parent_backup_id,
                    json.dumps(metadata.metadata),
                    metadata.size_bytes,
                    metadata.compression_algorithm.value
                ))
        except sqlite3.OperationalError as e:
            # backup_log table might not exist in older databases
            logger.warning(f"Could not record backup in database: {e}")

    def _verify_archive_checksums(self, tar: tarfile.TarFile, metadata: BackupMetadata) -> None:
        """
        Verify checksums of files in the archive without extracting them.

        Parameters
        ----------
        tar : tarfile.TarFile
            Open tar file to verify
        metadata : BackupMetadata
            Backup metadata containing expected checksums

        Raises
        ------
        ValueError
            If any file checksum doesn't match expected value
        """
        for member in tar.getmembers():
            if not member.isfile() or member.name == "manifest.json":
                continue  # Skip non-files and manifest

            expected_checksum = metadata.checksums.get(member.name)
            if expected_checksum is None:
                raise ValueError(f"No expected checksum found for file: {member.name}")

            # Stream the file and compute hash without extracting
            file_data = tar.extractfile(member)
            if file_data is None:
                raise ValueError(f"Could not read file from archive: {member.name}")

            sha256_hash = hashlib.sha256()
            while True:
                chunk = file_data.read(4096)
                if not chunk:
                    break
                sha256_hash.update(chunk)

            actual_checksum = sha256_hash.hexdigest()
            if actual_checksum != expected_checksum:
                raise ValueError(
                    f"Checksum mismatch for {member.name}: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )

        logger.debug("Archive checksum verification passed")

    def _verify_backup_integrity(self, backup_path: Path) -> None:
        """Verify backup integrity by checking file structure and checksums."""
        logger.debug(f"Verifying backup integrity: {backup_path}")

        try:
            # 1. Verify archive checksum from sidecar file first
            sidecar_path = backup_path.with_suffix(backup_path.suffix + '.sha256')
            archive_checksum_from_sidecar = None

            if sidecar_path.exists():
                with open(sidecar_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split('  ', 1)
                        if len(parts) == 2:
                            archive_checksum_from_sidecar = parts[0]
                            expected_filename = parts[1]
                            if expected_filename != backup_path.name:
                                logger.warning("Sidecar filename mismatch: expected "
                                               f"{expected_filename}, got {backup_path.name}")
                        else:
                            archive_checksum_from_sidecar = parts[0]

                if archive_checksum_from_sidecar:
                    actual_archive_checksum = self._calculate_file_checksum(backup_path)
                    if actual_archive_checksum != archive_checksum_from_sidecar:
                        raise ValueError(
                            f"Archive checksum mismatch (from sidecar): expected {archive_checksum_from_sidecar}, "
                            f"got {actual_archive_checksum}"
                        )
                    logger.debug("Archive checksum verification passed (from sidecar)")
            else:
                logger.debug(f"No sidecar checksum file found at {sidecar_path}, "
                             "skipping archive checksum verification")

            with tarfile.open(backup_path, "r:*") as tar:
                # Check that manifest exists
                manifest_info = tar.getmember("manifest.json")
                manifest_file = tar.extractfile(manifest_info)
                if manifest_file is None:
                    raise ValueError("Could not extract manifest file from backup")
                manifest_data = json.load(manifest_file)

                metadata = BackupMetadata.from_dict(manifest_data)

                # 2. Verify manifest integrity (if available)
                if metadata.manifest_checksum:
                    # Reconstruct manifest data without checksum fields for verification
                    manifest_for_verification = manifest_data.copy()
                    manifest_for_verification['archive_checksum'] = None
                    manifest_for_verification['manifest_checksum'] = None

                    # Calculate checksum of normalized manifest
                    manifest_json = json.dumps(manifest_for_verification, sort_keys=True, separators=(',', ':'))
                    actual_manifest_checksum = hashlib.sha256(manifest_json.encode('utf-8')).hexdigest()

                    if actual_manifest_checksum != metadata.manifest_checksum:
                        raise ValueError(
                            f"Manifest integrity check failed: expected {metadata.manifest_checksum}, "
                            f"got {actual_manifest_checksum}"
                        )
                    logger.debug("Manifest integrity verification passed")

                # Verify all expected files are present
                tar_members = {member.name for member in tar.getmembers() if member.isfile()}
                expected_files = set(metadata.file_paths.keys()) | {"manifest.json"}

                if tar_members != expected_files:
                    missing = expected_files - tar_members
                    extra = tar_members - expected_files
                    raise ValueError(f"Backup integrity check failed. Missing: {missing}, Extra: {extra}")

                # Verify content checksums of all archived files
                self._verify_archive_checksums(tar, metadata)

                logger.debug("Backup integrity verification passed")

        except Exception as e:
            logger.error(f"Backup integrity verification failed: {e}")
            raise

    def list_backups(self, backup_type: Optional[BackupType] = None) -> List[BackupMetadata]:
        """
        List available backups.

        Parameters
        ----------
        backup_type : BackupType, optional
            Filter by backup type. If None, returns all backups.

        Returns
        -------
        List[BackupMetadata]
            List of backup metadata objects
        """
        backups = []

        try:
            with sqlite3.connect(self.database_path) as conn:
                query = "SELECT * FROM backup_log"
                params = []

                if backup_type:
                    query += " WHERE backup_type = ?"
                    params.append(backup_type.value)

                query += " ORDER BY created_at DESC"

                cursor = conn.execute(query, params)
                for row in cursor.fetchall():
                    backup_metadata = self._backup_row_to_metadata(row)
                    backups.append(backup_metadata)

        except sqlite3.OperationalError:
            # backup_log table might not exist
            logger.debug("backup_log table not found, checking filesystem")
            backups = self._list_backups_from_filesystem(backup_type)

        return backups

    def _backup_row_to_metadata(self, row: tuple) -> BackupMetadata:
        """Convert database row to BackupMetadata object."""
        (backup_id, backup_type, created_at, database_version, file_path,
         checksum, parent_backup_id, metadata_json, size_bytes, compression_algorithm) = row

        return BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType(backup_type),
            database_name=self.database_name,
            database_version=database_version,
            created_at=parse_iso8601(created_at),
            file_paths={"backup": file_path},
            checksums={"backup": checksum},
            compression_algorithm=CompressionAlgorithm(compression_algorithm or "gzip"),
            size_bytes=size_bytes or 0,
            parent_backup_id=parent_backup_id,
            metadata=json.loads(metadata_json) if metadata_json else {}
        )

    def _list_backups_from_filesystem(self, backup_type: Optional[BackupType] = None) -> List[BackupMetadata]:
        """List backups by scanning filesystem (fallback method)."""
        backups = []

        for backup_file in self.config.backup_location.glob("*.lvdb-backup"):
            try:
                with tarfile.open(backup_file, "r:*") as tar:
                    manifest_file = tar.extractfile("manifest.json")
                    if manifest_file is None:
                        continue  # Skip backups without manifest
                    manifest_data = json.load(manifest_file)
                    metadata = BackupMetadata.from_dict(manifest_data)

                    if backup_type is None or metadata.backup_type == backup_type:
                        backups.append(metadata)

            except Exception as e:
                logger.warning(f"Could not read backup metadata from {backup_file}: {e}")

        return sorted(backups, key=lambda x: x.created_at, reverse=True)

    def restore_backup(
            self,
            backup_id: str,
            restore_location: Optional[Union[str, Path]] = None,
            overwrite_existing: bool = False
    ) -> Path:
        """
        Restore database from backup.

        Parameters
        ----------
        backup_id : str
            ID of backup to restore
        restore_location : Union[str, Path], optional
            Directory to restore to. If None, restores to original location.
        overwrite_existing : bool
            Whether to overwrite existing files

        Returns
        -------
        Path
            Path to restored database directory

        Raises
        ------
        FileNotFoundError
            If backup file not found
        ValueError
            If restore would overwrite existing files without permission
        """

        logger.info(f"Restoring backup: {backup_id}")

        # Find backup metadata
        backup_metadata = self._find_backup_metadata(backup_id)
        if not backup_metadata:
            raise FileNotFoundError(f"Backup not found: {backup_id}")

        # Find backup file
        backup_file = self._find_backup_file(backup_id)
        if not backup_file:
            raise FileNotFoundError(f"Backup file not found for ID: {backup_id}")

        # Determine restore location
        if restore_location is None:
            restore_location = self.database_path.parent
        else:
            restore_location = Path(restore_location)

        restore_location.mkdir(parents=True, exist_ok=True)

        # Check for existing files
        restored_db_path = restore_location / f"{backup_metadata.database_name}.sqlite"
        restored_faiss_path = restore_location / f"{backup_metadata.database_name}.faiss"

        if not overwrite_existing:
            if restored_db_path.exists() or restored_faiss_path.exists():
                raise ValueError(
                    "Files already exist at restore location. Use overwrite_existing=True to overwrite."
                )

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            try:
                # Extract backup
                self._extract_backup_archive(backup_file, temp_dir)

                # Verify integrity
                self._verify_extracted_backup(backup_metadata, temp_dir)

                # Restore files
                self._restore_database_files(backup_metadata, temp_dir, restore_location)

                logger.info(f"Backup restored successfully to: {restore_location}")
                return restore_location

            except Exception as e:
                logger.error(f"Backup restore failed: {e}")
                raise

    def _find_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Find backup metadata by ID."""
        backups = self.list_backups()
        for backup in backups:
            if backup.backup_id == backup_id:
                return backup
        return None

    def _find_backup_file_by_manifest(self, backup_id: str) -> Optional[Path]:
        """
        Find backup file by exact backup ID match from manifest.

        This method safely parses the manifest.json from each backup file
        to find an exact match, avoiding substring collision issues.

        Parameters
        ----------
        backup_id : str
            Exact backup ID to search for

        Returns
        -------
        Optional[Path]
            Path to backup file if found, None otherwise
        """
        for backup_file in self.config.backup_location.glob("*.lvdb-backup"):
            try:
                with tarfile.open(backup_file, "r:*") as tar:
                    # Try to extract and parse manifest
                    try:
                        manifest_info = tar.getmember("manifest.json")
                        manifest_file = tar.extractfile(manifest_info)
                        if manifest_file is None:
                            continue  # Skip if manifest can't be read

                        manifest_data = json.load(manifest_file)
                        file_backup_id = manifest_data.get('backup_id')

                        if file_backup_id == backup_id:
                            return backup_file

                    except (KeyError, json.JSONDecodeError):
                        # Manifest missing or corrupted, skip this file
                        continue

            except Exception as e:
                # Archive corrupted or unreadable, log warning and continue
                logger.warning(f"Could not read backup file {backup_file}: {e}")
                continue

        return None

    def _find_backup_file(self, backup_id: str) -> Optional[Path]:
        """
        Find backup file by ID using manifest-based exact matching only.

        This method provides secure, predictable backup file resolution
        by relying exclusively on manifest parsing rather than potentially
        unsafe filename matching.
        """
        # Use only manifest-based exact matching for security and predictability
        return self._find_backup_file_by_manifest(backup_id)

    def _is_within_directory(self, directory: Path, target: Path) -> bool:
        """
        Check if target path is within the given directory.

        Parameters
        ----------
        directory : Path
            Base directory path
        target : Path
            Target path to check

        Returns
        -------
        bool
            True if target is within directory, False otherwise
        """
        try:
            directory = directory.resolve()
            target = target.resolve()
            target.relative_to(directory)
            return True
        except Exception:
            return False

    # Maximum allowed path length to prevent DoS via excessively long paths
    MAX_PATH_LENGTH = 4096

    def _safe_extract(self, tar: tarfile.TarFile, path: Path) -> None:
        """
        Safely extract tar archive, preventing path traversal attacks.

        Security features:
        - Rejects symlinks and hard links
        - Rejects absolute paths (Unix and Windows formats)
        - Rejects path traversal attempts (..)
        - Rejects null bytes and control characters
        - Enforces maximum path length
        - Verifies extracted paths stay within destination
        - Rejects device files and special file types

        Parameters
        ----------
        tar : tarfile.TarFile
            Tar file to extract
        path : Path
            Destination path for extraction

        Raises
        ------
        ValueError
            If archive contains unsafe paths or file types
        """
        for member in tar.getmembers():
            member_name = member.name

            # Reject null bytes which could be used for path manipulation
            if '\x00' in member_name:
                raise ValueError(f"Null byte detected in path: {repr(member_name)}")

            # Reject control characters (ASCII 0-31 except tab, newline, carriage return)
            for char in member_name:
                if ord(char) < 32 and char not in '\t\n\r':
                    raise ValueError(f"Control character detected in path: {repr(member_name)}")

            # Reject excessively long paths (DoS prevention)
            if len(member_name) > self.MAX_PATH_LENGTH:
                raise ValueError(
                    f"Path exceeds maximum length ({self.MAX_PATH_LENGTH} chars): {member_name[:100]}..."
                )

            member_path = path / member_name

            # Reject symlinks and hard links to prevent link-based attacks
            if member.islnk() or member.issym():
                raise ValueError(f"Refusing to extract archives with (sym)links: {member_name}")

            # Reject absolute paths - both Unix (/path) and Windows (C:\path, \\server\share)
            # Using Path.is_absolute() handles both platforms correctly
            if Path(member_name).is_absolute():
                raise ValueError(f"Absolute path detected in tar: {member_name}")

            # Also check for Windows drive letters that Path might not catch in all cases
            # e.g., "C:" at the start of the path
            if len(member_name) >= 2 and member_name[1] == ':' and member_name[0].isalpha():
                raise ValueError(f"Windows absolute path detected in tar: {member_name}")

            # Check for UNC paths (\\server\share or //server/share)
            if member_name.startswith('\\\\') or member_name.startswith('//'):
                raise ValueError(f"UNC path detected in tar: {member_name}")

            # Reject path traversal attempts using .. in any path component
            if ".." in Path(member_name).parts:
                raise ValueError(f"Path traversal attempt detected: {member_name}")

            # Verify extracted path stays within destination directory
            if not self._is_within_directory(path, member_path):
                raise ValueError(f"Path traversal detected: {member_name}")

            # Reject device files and other special file types
            if member.ischr() or member.isblk() or member.isfifo():
                raise ValueError(f"Refusing to extract special file type: {member_name}")

        # If all validations pass, extract the archive
        tar.extractall(path=path)

    def _extract_backup_archive(self, backup_file: Path, temp_dir: Path) -> None:
        """Extract backup archive to temporary directory."""
        logger.debug(f"Extracting backup archive: {backup_file}")

        with tarfile.open(backup_file, "r:*") as tar:
            self._safe_extract(tar, temp_dir)

    def _verify_extracted_backup(self, expected_metadata: BackupMetadata, temp_dir: Path) -> None:
        """Verify extracted backup matches expected metadata."""

        # Load manifest from extracted files
        manifest_path = temp_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Backup manifest not found in archive")

        with open(manifest_path) as f:
            extracted_metadata = BackupMetadata.from_dict(json.load(f))

        # Verify backup ID matches
        if extracted_metadata.backup_id != expected_metadata.backup_id:
            raise ValueError(
                f"Backup ID mismatch: expected {expected_metadata.backup_id}, got {extracted_metadata.backup_id}")

        # Verify checksums of extracted files
        for filename, expected_checksum in extracted_metadata.checksums.items():
            if filename == "manifest.json":  # Skip manifest checksum verification
                continue

            file_path = temp_dir / filename
            if not file_path.exists():
                raise ValueError(f"Expected file not found in backup: {filename}")

            actual_checksum = self._calculate_file_checksum(file_path)
            if actual_checksum != expected_checksum:
                raise ValueError(
                    f"Checksum mismatch for {filename}: expected {expected_checksum}, got {actual_checksum}")

        logger.debug("Backup integrity verification passed")

    def _restore_database_files(self, metadata: BackupMetadata, temp_dir: Path, restore_location: Path) -> None:
        """Restore database files from temporary directory to final location."""

        # Restore SQLite database
        source_db = temp_dir / f"{metadata.database_name}.sqlite"
        if source_db.exists():
            target_db = restore_location / f"{metadata.database_name}.sqlite"
            shutil.copy2(source_db, target_db)
            logger.debug(f"Restored database: {target_db}")

        # Restore FAISS index if present
        source_faiss = temp_dir / f"{metadata.database_name}.faiss"
        if source_faiss.exists():
            target_faiss = restore_location / f"{metadata.database_name}.faiss"
            shutil.copy2(source_faiss, target_faiss)
            logger.debug(f"Restored FAISS index: {target_faiss}")

    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.

        Parameters
        ----------
        backup_id : str
            ID of backup to delete

        Returns
        -------
        bool
            True if backup was deleted successfully
        """

        logger.info(f"Deleting backup: {backup_id}")

        # Find and delete backup file
        backup_file = self._find_backup_file(backup_id)
        if backup_file and backup_file.exists():
            backup_file.unlink()
            logger.debug(f"Deleted backup file: {backup_file}")

            # Also delete sidecar checksum file if it exists
            sidecar_path = backup_file.with_suffix(backup_file.suffix + '.sha256')
            if sidecar_path.exists():
                sidecar_path.unlink()
                logger.debug(f"Deleted sidecar checksum file: {sidecar_path}")

        # Remove from database log
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute("DELETE FROM backup_log WHERE id = ?", (backup_id,))
                if cursor.rowcount > 0:
                    logger.debug("Removed backup from database log")
        except sqlite3.OperationalError:
            logger.debug("backup_log table not available")

        return True

    def cleanup_old_backups(self, retention_days: Optional[int] = None) -> int:
        """
        Clean up old backups based on retention policy.

        Parameters
        ----------
        retention_days : int, optional
            Number of days to retain backups. If None, uses config value.

        Returns
        -------
        int
            Number of backups deleted
        """

        if retention_days is None:
            retention_days = self.config.retention_days

        if retention_days <= 0:
            return 0  # No cleanup if retention is 0 or negative

        cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)
        logger.info(f"Cleaning up backups older than {cutoff_date}")

        backups = self.list_backups()
        deleted_count = 0

        for backup in backups:
            if backup.created_at < cutoff_date:
                try:
                    self.delete_backup(backup.backup_id)
                    deleted_count += 1
                    logger.debug(f"Deleted old backup: {backup.backup_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete backup {backup.backup_id}: {e}")

        logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count

    def verify_backup_streaming(self, backup_id: str, verify_archive_members: bool = True) -> bool:
        """
        Verify backup integrity using streaming without full extraction.

        Parameters
        ----------
        backup_id : str
            ID of backup to verify
        verify_archive_members : bool, default True
            Whether to verify individual archive member checksums.
            If False, only verifies archive and manifest checksums.

        Returns
        -------
        bool
            True if backup is valid

        Notes
        -----
        This method provides efficient verification by streaming archive contents
        rather than extracting to disk. Useful for large backups or when disk
        space is limited.
        """
        logger.info(f"Streaming verification of backup: {backup_id}")

        backup_file = self._find_backup_file(backup_id)
        if not backup_file:
            logger.error(f"Backup file not found: {backup_id}")
            return False

        try:
            # Verify archive checksum from sidecar file first
            sidecar_path = backup_file.with_suffix(backup_file.suffix + '.sha256')
            archive_checksum_from_sidecar = None

            if sidecar_path.exists():
                with open(sidecar_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split('  ', 1)
                        if len(parts) == 2:
                            archive_checksum_from_sidecar = parts[0]
                            expected_filename = parts[1]
                            if expected_filename != backup_file.name:
                                logger.warning(f"Sidecar filename mismatch: expected "
                                               f"{expected_filename}, got {backup_file.name}")
                        else:
                            archive_checksum_from_sidecar = parts[0]

                if archive_checksum_from_sidecar:
                    actual_archive_checksum = self._calculate_file_checksum(backup_file)
                    if actual_archive_checksum != archive_checksum_from_sidecar:
                        raise ValueError(
                            f"Archive checksum mismatch (from sidecar): expected {archive_checksum_from_sidecar}, "
                            f"got {actual_archive_checksum}"
                        )
                    logger.debug("Archive checksum verification passed (from sidecar)")
            else:
                logger.debug(f"No sidecar checksum file found at {sidecar_path}, "
                             "skipping archive checksum verification")

            with tarfile.open(backup_file, "r:*") as tar:
                # Extract and parse manifest
                manifest_info = tar.getmember("manifest.json")
                manifest_file = tar.extractfile(manifest_info)
                if manifest_file is None:
                    raise ValueError("Could not extract manifest file from backup")
                manifest_data = json.load(manifest_file)
                metadata = BackupMetadata.from_dict(manifest_data)

                # Verify manifest integrity (if available)
                if metadata.manifest_checksum:
                    manifest_for_verification = manifest_data.copy()
                    manifest_for_verification['archive_checksum'] = None
                    manifest_for_verification['manifest_checksum'] = None

                    manifest_json = json.dumps(manifest_for_verification, sort_keys=True, separators=(',', ':'))
                    actual_manifest_checksum = hashlib.sha256(manifest_json.encode('utf-8')).hexdigest()

                    if actual_manifest_checksum != metadata.manifest_checksum:
                        raise ValueError(
                            f"Manifest integrity check failed: expected {metadata.manifest_checksum}, "
                            f"got {actual_manifest_checksum}"
                        )
                    logger.debug("Manifest integrity verification passed")

                # Verify file structure
                tar_members = {member.name for member in tar.getmembers() if member.isfile()}
                expected_files = set(metadata.file_paths.keys()) | {"manifest.json"}

                if tar_members != expected_files:
                    missing = expected_files - tar_members
                    extra = tar_members - expected_files
                    raise ValueError(f"Backup structure check failed. Missing: {missing}, Extra: {extra}")

                # Optionally verify individual archive member checksums via streaming
                if verify_archive_members:
                    self._verify_archive_checksums(tar, metadata)
                    logger.debug("Archive member checksum verification passed")

            logger.info(f"Streaming backup verification passed: {backup_id}")
            return True

        except Exception as e:
            logger.error(f"Streaming backup verification failed: {e}")
            return False

    def verify_backup(self, backup_id: str) -> bool:
        """
        Verify backup integrity without restoring.

        Parameters
        ----------
        backup_id : str
            ID of backup to verify

        Returns
        -------
        bool
            True if backup is valid
        """

        logger.info(f"Verifying backup: {backup_id}")

        backup_file = self._find_backup_file(backup_id)
        if not backup_file:
            logger.error(f"Backup file not found: {backup_id}")
            return False

        try:
            self._verify_backup_integrity(backup_file)
            logger.info(f"Backup verification passed: {backup_id}")
            return True
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

    def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """
        Get detailed information about a backup.

        Parameters
        ----------
        backup_id : str
            ID of backup to get info for

        Returns
        -------
        BackupMetadata or None
            Backup metadata if found
        """
        return self._find_backup_metadata(backup_id)


class IncrementalBackupManager:
    """
    Specialized manager for incremental backups using WAL tracking.

    Implements incremental backup functionality by tracking changes in SQLite's
    Write-Ahead Log (WAL) and maintaining FAISS index deltas.

    Parameters
    ----------
    backup_manager : BackupManager
        Parent backup manager instance
    """

    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        self.database_path = backup_manager.database_path
        self.faiss_index_path = backup_manager.faiss_index_path
        self.config = backup_manager.config
        self.version_manager = backup_manager.version_manager

    def create_incremental_backup(
            self,
            parent_backup_id: str,
            backup_id: Optional[str] = None
    ) -> str:
        """
        Create an incremental backup based on changes since parent backup.

        Parameters
        ----------
        parent_backup_id : str
            ID of the parent (full or incremental) backup
        backup_id : str, optional
            Custom backup ID. If None, generates a UUID.

        Returns
        -------
        str
            Unique backup ID for the incremental backup

        Raises
        ------
        ValueError
            If parent backup not found or WAL mode not enabled
        FileNotFoundError
            If database files don't exist
        """

        # Verify parent backup exists
        parent_backup = self.backup_manager._find_backup_metadata(parent_backup_id)
        if not parent_backup:
            raise ValueError(f"Parent backup not found: {parent_backup_id}")

        if backup_id is None:
            backup_id = str(uuid.uuid4())

        logger.info(f"Creating incremental backup {backup_id} based on parent {parent_backup_id}")

        # Enable WAL mode if not already enabled
        self._ensure_wal_mode()

        # Get changes since parent backup
        changes = self._get_changes_since_backup(parent_backup)

        if not changes['has_changes']:
            logger.info("No changes detected since parent backup")
            # Still create backup but mark as empty incremental
            changes['changed_documents'] = []
            changes['deleted_documents'] = []
            changes['faiss_changes'] = []

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            try:
                # Create incremental database with only changes
                self._create_incremental_database(changes, temp_dir)

                # Create incremental FAISS index if there are vector changes
                if changes['faiss_changes']:
                    self._create_incremental_faiss_index(changes, temp_dir)

                # Create change manifest
                self._create_change_manifest(changes, temp_dir, parent_backup_id)

                # Create backup metadata
                metadata = self.backup_manager._create_backup_manifest(
                    backup_id, BackupType.INCREMENTAL, temp_dir, parent_backup_id
                )

                # Add incremental-specific metadata
                metadata.metadata.update({
                    'incremental_type': 'wal_based',
                    'parent_backup_id': parent_backup_id,
                    'changes_count': len(changes['changed_documents']),
                    'deletions_count': len(changes['deleted_documents']),
                    'faiss_changes_count': len(changes['faiss_changes'])
                })

                # Add current document IDs to metadata for future deletion detection
                try:
                    with sqlite3.connect(self.database_path) as conn:
                        cursor = conn.execute("SELECT id FROM documents")
                        document_ids = [row[0] for row in cursor.fetchall()]
                        metadata.metadata['document_ids'] = document_ids
                        logger.debug(f"Added {len(document_ids)} document IDs to incremental backup metadata")
                except Exception as e:
                    logger.warning(f"Could not add document IDs to incremental metadata: {e}")

                # Add comprehensive document manifest for fast diffing without extraction
                try:
                    metadata.metadata['document_manifest'] = self.backup_manager._generate_document_manifest(
                        self.database_path
                    )
                    logger.debug("Added document manifest to incremental backup metadata")
                except Exception as e:
                    logger.warning(f"Could not add document manifest to incremental metadata: {e}")

                # Create compressed archive
                backup_path = self.backup_manager._create_backup_archive(metadata, temp_dir)

                # Record backup in database
                self.backup_manager._record_backup_in_database(metadata, backup_path)

                # Update last backup timestamp
                self._update_last_backup_timestamp()

                # Small delay to ensure file locks are released on Windows
                time.sleep(0.1)

                logger.info(f"Incremental backup completed: {backup_id}")
                return backup_id

            except Exception as e:
                logger.error(f"Incremental backup failed: {e}")
                raise

    def _ensure_wal_mode(self) -> None:
        """Ensure database is using WAL mode for incremental backups."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("PRAGMA journal_mode")
            current_mode = cursor.fetchone()[0]

            if current_mode.lower() != 'wal':
                logger.info("Enabling WAL mode for incremental backups")
                conn.execute("PRAGMA journal_mode=WAL")
                conn.commit()

                # Perform initial checkpoint to ensure WAL file is created
                conn.execute("PRAGMA wal_checkpoint(FULL)")
                logger.debug("WAL mode enabled and initial checkpoint performed")

    def _get_changes_since_backup(self, parent_backup: BackupMetadata) -> Dict[str, Any]:
        """
        Get database changes since parent backup using manifest-based diffing.

        Uses document manifest from parent backup metadata for fast diffing without
        extraction. Parent backup must have document manifest.

        Parameters
        ----------
        parent_backup : BackupMetadata
            Parent backup metadata with document manifest

        Returns
        -------
        Dict[str, Any]
            Dictionary containing changed documents, deleted documents, and FAISS changes

        Raises
        ------
        ValueError
            If parent backup lacks document manifest
        """
        # Check if parent backup has document manifest
        parent_manifest = parent_backup.metadata.get('document_manifest') if parent_backup.metadata else None

        if not parent_manifest:
            raise ValueError(f"Parent backup {parent_backup.backup_id} lacks document manifest. "
                             "Only backups with document manifest are supported for incremental operations.")

        logger.debug("Using manifest-based diffing for incremental backup")

        changes = {
            'has_changes': False,
            'changed_documents': [],
            'deleted_documents': [],
            'faiss_changes': [],
            'parent_timestamp': parent_backup.created_at
        }

        # Generate current database manifest
        current_manifest = self.backup_manager._generate_document_manifest(self.database_path)

        current_doc_ids = set(current_manifest['documents'].keys())
        parent_doc_ids = set(parent_manifest['documents'].keys())

        # Detect deleted documents with FAISS IDs (no extraction needed!)
        deleted_doc_ids = parent_doc_ids - current_doc_ids
        if deleted_doc_ids:
            logger.info(f"Detected {len(deleted_doc_ids)} deleted documents using manifest")

            for doc_id in deleted_doc_ids:
                parent_doc = parent_manifest['documents'][doc_id]

                # Add deleted document
                changes['deleted_documents'].append({
                    'id': doc_id,
                    'action': 'delete'
                })

                # Add FAISS IDs for deleted chunks directly from manifest
                for _, chunk_info in parent_doc['chunks'].items():
                    if chunk_info.get('faiss_id') is not None:
                        changes['faiss_changes'].append({
                            'document_id': doc_id,
                            'faiss_id': chunk_info['faiss_id'],
                            'action': 'delete'
                        })

                changes['has_changes'] = True

        # Detect modified documents by comparing with parent backup timestamp
        # This preserves the original timestamp-based logic for modified documents
        parent_timestamp = parent_backup.created_at

        with sqlite3.connect(self.database_path) as conn:
            # Get documents modified since parent backup
            cursor = conn.execute("""
                SELECT id, content, content_hash, updated_at
                FROM documents
                WHERE updated_at > ?
                ORDER BY updated_at
            """, (parent_timestamp.isoformat(),))

            for row in cursor.fetchall():
                doc_id, content, content_hash, updated_at = row
                _changes: list[dict] = changes["changed_documents"]
                _changes.append({
                    'id': doc_id,
                    'content': content,
                    'content_hash': content_hash,
                    'updated_at': updated_at
                })
                changes['has_changes'] = True

            # Get chunks for changed documents
            if changes['changed_documents']:
                doc_ids = [doc['id'] for doc in changes['changed_documents']]
                placeholders = ','.join(['?'] * len(doc_ids))

                cursor = conn.execute(f"""
                    SELECT document_id, chunk_index, content, content_hash,
                           start_pos, end_pos, start_line, start_col, end_line, end_col,
                           tokens, faiss_id
                    FROM chunks
                    WHERE document_id IN ({placeholders})
                    ORDER BY document_id, chunk_index
                """, doc_ids)

                chunks_by_doc = defaultdict(list)
                for row in cursor.fetchall():
                    doc_id = row[0]
                    chunk_data = {
                        'chunk_index': row[1],
                        'content': row[2],
                        'content_hash': row[3],
                        'start_pos': row[4],
                        'end_pos': row[5],
                        'start_line': row[6],
                        'start_col': row[7],
                        'end_line': row[8],
                        'end_col': row[9],
                        'tokens': row[10],
                        'faiss_id': row[11]
                    }
                    chunks_by_doc[doc_id].append(chunk_data)

                # Attach chunks to documents and track FAISS changes
                for doc in changes['changed_documents']:
                    doc['chunks'] = chunks_by_doc.get(doc['id'], [])

                    # Add FAISS changes for modified chunks
                    for chunk in doc['chunks']:
                        if chunk.get('faiss_id') is not None:
                            changes['faiss_changes'].append({
                                'document_id': doc['id'],
                                'chunk_index': chunk['chunk_index'],
                                'faiss_id': chunk['faiss_id'],
                                'action': 'update'
                            })

        logger.info(f"Manifest-based diffing found {len(changes['changed_documents'])} changed documents, "
                    f"{len(changes['deleted_documents'])} deleted documents, "
                    f"{len(changes['faiss_changes'])} FAISS changes")

        return changes

    def _create_incremental_database(self, changes: Dict[str, Any], temp_dir: Path) -> None:
        """Create incremental SQLite database containing only changes."""

        inc_db_path = temp_dir / f"{self.backup_manager.database_name}.sqlite"

        # Create incremental database with same schema
        inc_conn = sqlite3.connect(inc_db_path)
        orig_conn = sqlite3.connect(self.database_path)

        try:
            # Copy schema from original database
            cursor = orig_conn.execute("""
                SELECT sql FROM sqlite_master
                WHERE type IN ('table', 'index', 'trigger') AND sql IS NOT NULL
                ORDER BY type DESC
            """)

            for row in cursor.fetchall():
                sql = row[0]
                try:
                    inc_conn.execute(sql)
                except sqlite3.OperationalError:
                    # Skip if already exists or is incompatible
                    pass

            # Insert changed documents
            for doc in changes['changed_documents']:
                inc_conn.execute("""
                    INSERT OR REPLACE INTO documents
                    (id, content, content_hash, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    doc['id'], doc['content'], doc['content_hash'],
                    doc.get('created_at', datetime.now(UTC).isoformat()),
                    doc['updated_at']
                ))

                # Insert chunks for this document
                for chunk in doc.get('chunks', []):
                    inc_conn.execute("""
                        INSERT OR REPLACE INTO chunks
                        (document_id, chunk_index, content, content_hash,
                         start_pos, end_pos, start_line, start_col, end_line, end_col,
                         tokens, faiss_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        doc['id'], chunk['chunk_index'], chunk['content'], chunk['content_hash'],
                        chunk['start_pos'], chunk['end_pos'], chunk['start_line'], chunk['start_col'],
                        chunk['end_line'], chunk['end_col'], chunk['tokens'], chunk['faiss_id']
                    ))

            # Record change metadata
            inc_conn.execute("""
                INSERT OR REPLACE INTO config (key, value)
                VALUES (?, ?)
            """, ('incremental_backup_type', 'wal_based'))

            inc_conn.execute("""
                INSERT OR REPLACE INTO config (key, value)
                VALUES (?, ?)
            """, ('parent_backup_timestamp', changes['parent_timestamp'].isoformat()))

            inc_conn.commit()

        finally:
            # Explicitly close connections to release file locks on Windows
            orig_conn.close()
            inc_conn.close()

        logger.debug(f"Created incremental database: {inc_db_path}")

    def _create_compatible_base_index(self, original_index, dimension: int):
        """Create a base FAISS index that matches the original index type and metric."""
        import faiss

        # Get the base index from IndexIDMap if wrapped
        if hasattr(original_index, 'index'):
            base_index = original_index.index
        else:
            base_index = original_index

        # Detect index type and metric
        index_type = str(type(base_index).__name__)

        # Handle different index types and metrics
        if 'IP' in index_type or 'InnerProduct' in index_type:
            # Inner product metric
            return faiss.IndexFlatIP(dimension)
        elif 'L2' in index_type:
            # L2 metric (default)
            return faiss.IndexFlatL2(dimension)
        elif 'HNSW' in index_type:
            # For HNSW, try to preserve the metric but use flat for incremental
            if 'IP' in index_type:
                return faiss.IndexFlatIP(dimension)
            else:
                return faiss.IndexFlatL2(dimension)
        else:
            # Default fallback to L2
            logger.warning(f"Unknown index type {index_type}, defaulting to IndexFlatL2")
            return faiss.IndexFlatL2(dimension)

    def _create_incremental_faiss_index(self, changes: Dict[str, Any], temp_dir: Path) -> None:
        """Create incremental FAISS index containing only changed vectors."""

        if not self.faiss_index_path.exists():
            logger.warning("Original FAISS index not found, skipping incremental FAISS backup")
            return

        try:
            # Load original FAISS index
            original_index = faiss.read_index(str(self.faiss_index_path))

            # Extract changed vectors
            faiss_ids = [change['faiss_id'] for change in changes['faiss_changes']]

            if not faiss_ids:
                return

            # Create new index with only changed vectors
            changed_vectors = []
            for faiss_id in faiss_ids:
                try:
                    # Get vector from original index
                    vector = original_index.reconstruct(int(faiss_id))
                    changed_vectors.append(vector)
                except Exception as e:
                    logger.warning(f"Could not extract vector {faiss_id}: {e}")

            if changed_vectors:
                # Create incremental FAISS index
                vectors_array = np.array(changed_vectors)

                # Create new index with same configuration as original
                base_index = self._create_compatible_base_index(original_index, vectors_array.shape[1])
                inc_index = faiss.IndexIDMap2(base_index)

                # Add vectors with their original IDs
                inc_index.add_with_ids(vectors_array, np.array(faiss_ids, dtype=np.int64))

                # Save incremental index
                inc_faiss_path = temp_dir / f"{self.backup_manager.database_name}.faiss"
                faiss.write_index(inc_index, str(inc_faiss_path))

                logger.debug(f"Created incremental FAISS index with {len(changed_vectors)} vectors")

        except Exception as e:
            logger.error(f"Failed to create incremental FAISS index: {e}")
            # Continue without FAISS incremental backup

    def _create_change_manifest(self, changes: Dict[str, Any], temp_dir: Path, parent_backup_id: str) -> None:
        """Create manifest describing the incremental changes."""

        manifest = {
            'type': 'incremental_changes',
            'parent_backup_id': parent_backup_id,
            'parent_timestamp': changes['parent_timestamp'].isoformat(),
            'created_at': datetime.now(UTC).isoformat(),
            'changes_summary': {
                'documents_changed': len(changes['changed_documents']),
                'documents_deleted': len(changes['deleted_documents']),
                'faiss_changes': len(changes['faiss_changes'])
            },
            'document_changes': [
                {
                    'id': doc['id'],
                    'content_hash': doc['content_hash'],
                    'updated_at': doc['updated_at'],
                    'chunks_count': len(doc.get('chunks', []))
                }
                for doc in changes['changed_documents']
            ],
            'deleted_documents': [
                {
                    'id': doc['id'],
                    'action': doc.get('action', 'delete')
                }
                for doc in changes['deleted_documents']
            ],
            'faiss_changes': changes['faiss_changes']
        }

        manifest_path = temp_dir / "incremental_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.debug(f"Created incremental change manifest: {manifest_path}")

    def _update_last_backup_timestamp(self):
        """Update the last backup timestamp in the database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO config (key, value)
                    VALUES (?, ?)
                """, ('last_backup_timestamp', datetime.now(UTC).isoformat()))
                conn.commit()
        except sqlite3.OperationalError:
            logger.debug("Could not update last backup timestamp")

    def restore_incremental_backup_chain(
            self,
            target_backup_id: str,
            restore_location: Union[str, Path]
    ) -> Path:
        """
        Restore database by applying a chain of incremental backups.

        Parameters
        ----------
        target_backup_id : str
            ID of the target backup (can be full or incremental)
        restore_location : Union[str, Path]
            Directory to restore to

        Returns
        -------
        Path
            Path to restored database directory
        """

        logger.info(f"Restoring incremental backup chain to {target_backup_id}")

        # Build backup chain from target back to full backup
        backup_chain = self._build_backup_chain(target_backup_id)

        restore_location = Path(restore_location)
        restore_location.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            try:
                # Start with the full backup (first in chain)
                full_backup = backup_chain[0]
                logger.info(f"Restoring full backup: {full_backup.backup_id}")

                self.backup_manager.restore_backup(
                    full_backup.backup_id,
                    temp_dir,
                    overwrite_existing=True
                )

                # Apply incremental backups in order
                for inc_backup in backup_chain[1:]:
                    logger.info(f"Applying incremental backup: {inc_backup.backup_id}")
                    self._apply_incremental_backup(inc_backup, temp_dir)

                # Move restored files to final location
                self._finalize_incremental_restore(temp_dir, restore_location)

                # Small delay to ensure file locks are released on Windows
                time.sleep(0.1)

                logger.info(f"Incremental restore completed: {restore_location}")
                return restore_location

            except Exception as e:
                logger.error(f"Incremental restore failed: {e}")
                raise

    def _build_backup_chain(self, target_backup_id: str) -> List[BackupMetadata]:
        """Build the chain of backups from full backup to target."""

        chain = []
        current_backup_id = target_backup_id

        while current_backup_id:
            backup = self.backup_manager._find_backup_metadata(current_backup_id)
            if not backup:
                raise ValueError(f"Backup not found in chain: {current_backup_id}")

            chain.append(backup)

            if backup.backup_type == BackupType.FULL:
                break

            current_backup_id = backup.parent_backup_id
            if current_backup_id is None:
                break

        if not chain or chain[-1].backup_type != BackupType.FULL:
            raise ValueError("Could not find full backup at root of chain")

        # Reverse to get chronological order (full backup first)
        chain.reverse()
        return chain

    def _apply_incremental_backup(self, inc_backup: BackupMetadata, working_dir: Path) -> None:
        """Apply an incremental backup to the working directory."""

        # Extract incremental backup
        backup_file = self.backup_manager._find_backup_file(inc_backup.backup_id)
        if not backup_file:
            raise FileNotFoundError(f"Incremental backup file not found: {inc_backup.backup_id}")

        with tempfile.TemporaryDirectory() as inc_temp_dir_str:
            inc_temp_dir = Path(inc_temp_dir_str)

            # Extract incremental backup
            self.backup_manager._extract_backup_archive(backup_file, inc_temp_dir)

            # Apply database changes
            self._apply_database_changes(inc_temp_dir, working_dir)

            # Apply FAISS changes if present
            faiss_file = inc_temp_dir / f"{self.backup_manager.database_name}.faiss"
            if faiss_file.exists():
                self._apply_faiss_changes(faiss_file, working_dir)

    def _apply_database_changes(self, inc_dir: Path, working_dir: Path) -> None:
        """Apply database changes from incremental backup."""

        working_db_path = working_dir / f"{self.backup_manager.database_name}.sqlite"
        working_conn = sqlite3.connect(working_db_path)

        try:
            # First, process deletions from the manifest
            manifest_path = inc_dir / "incremental_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)

                # Apply deletions first (before any insertions/updates)
                deleted_documents = manifest.get('deleted_documents', [])
                if deleted_documents:
                    deleted_ids = [doc['id'] for doc in deleted_documents]
                    logger.info(f"Applying {len(deleted_ids)} document deletions")

                    # Delete chunks first (foreign key constraint)
                    placeholders = ','.join(['?' for _ in deleted_ids])
                    working_conn.execute(
                        f"DELETE FROM chunks WHERE document_id IN ({placeholders})",
                        deleted_ids
                    )

                    # Then delete documents
                    cursor = working_conn.execute(
                        f"DELETE FROM documents WHERE id IN ({placeholders})",
                        deleted_ids
                    )
                    actual_deleted = cursor.rowcount
                    logger.debug(f"Deleted {actual_deleted} documents from restored database")

                    working_conn.commit()

            # Now apply insertions/updates from the incremental database
            inc_db_path = inc_dir / f"{self.backup_manager.database_name}.sqlite"
            if not inc_db_path.exists():
                return  # No database changes to apply

            inc_conn = sqlite3.connect(inc_db_path)

            try:
                # Get documents table schema to handle all columns dynamically
                cursor = inc_conn.execute("PRAGMA table_info(documents)")
                column_info = cursor.fetchall()
                column_names = [col[1] for col in column_info]  # col[1] is the column name
                placeholders = ', '.join(['?' for _ in column_names])
                column_names_str = ', '.join(column_names)

                # Copy changed documents with all columns
                cursor = inc_conn.execute(f"SELECT {column_names_str} FROM documents")
                for row in cursor.fetchall():
                    working_conn.execute(f"""
                        INSERT OR REPLACE INTO documents
                        ({column_names_str})
                        VALUES ({placeholders})
                    """, row)

                # Copy changed chunks with optimized bulk operations
                cursor = inc_conn.execute("SELECT * FROM chunks")
                all_chunk_rows = cursor.fetchall()

                if all_chunk_rows:
                    # Collect unique document IDs that need chunk replacement
                    affected_doc_ids = set()
                    for row in all_chunk_rows:
                        doc_id = row[1]  # document_id is second column
                        affected_doc_ids.add(doc_id)

                    # Bulk delete all chunks for affected documents
                    if affected_doc_ids:
                        placeholders = ','.join(['?' for _ in affected_doc_ids])
                        working_conn.execute(
                            f"DELETE FROM chunks WHERE document_id IN ({placeholders})",
                            list(affected_doc_ids)
                        )
                        logger.debug(f"Bulk deleted chunks for {len(affected_doc_ids)} documents")

                    # Bulk insert all new chunks
                    chunk_insert_data = []
                    for row in all_chunk_rows:
                        chunk_insert_data.append(row[1:])  # Skip the auto-increment ID

                    working_conn.executemany("""
                        INSERT INTO chunks
                        (document_id, chunk_index, content, content_hash,
                         start_pos, end_pos, start_line, start_col, end_line, end_col,
                         tokens, faiss_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, chunk_insert_data)
                    logger.debug(f"Bulk inserted {len(chunk_insert_data)} chunks")

                working_conn.commit()

            finally:
                # Explicitly close connections to release file locks on Windows
                inc_conn.close()

        finally:
            working_conn.close()

            # Small delay to ensure file locks are released
            time.sleep(0.1)

    def _apply_faiss_changes(self, inc_faiss_path: Path, working_dir: Path) -> None:
        """Apply FAISS index changes from incremental backup."""

        working_faiss_path = working_dir / f"{self.backup_manager.database_name}.faiss"

        if not working_faiss_path.exists():
            # No existing FAISS index, just copy the incremental one
            shutil.copy2(inc_faiss_path, working_faiss_path)
            return

        try:
            # Load both indexes
            working_index = faiss.read_index(str(working_faiss_path))
            inc_index = faiss.read_index(str(inc_faiss_path))

            # Extract vectors and IDs from incremental index using centralized utilities
            inc_ids = get_faiss_external_ids(inc_index)

            # Reconstruct vectors using external IDs (preferred method for IndexIDMap2)
            inc_vectors = []
            for fid in inc_ids:
                try:
                    # Try direct reconstruction with external ID first
                    vector = inc_index.reconstruct(int(fid))
                    inc_vectors.append(vector)
                except Exception as e:
                    logger.debug(f"Failed to reconstruct vector for external ID {fid}: {e}")
                    continue

            if not inc_vectors:
                logger.warning("No vectors could be reconstructed from incremental index")
                return

            inc_vectors = np.array(inc_vectors, dtype=np.float32)

            # Remove old vectors with same IDs and add new ones
            for inc_id in inc_ids:
                try:
                    working_index.remove_ids(np.array([inc_id], dtype=np.int64))
                except Exception:
                    pass  # ID not found, which is fine

            # Add updated vectors
            working_index.add_with_ids(inc_vectors, inc_ids)

            # Save updated index
            faiss.write_index(working_index, str(working_faiss_path))

        except Exception as e:
            logger.error(f"Failed to apply FAISS changes: {e}")
            # Fall back to copying incremental index
            shutil.copy2(inc_faiss_path, working_faiss_path)

    def _finalize_incremental_restore(self, working_dir: Path, final_location: Path) -> None:
        """Move restored files from working directory to final location."""

        # Move database file
        working_db = working_dir / f"{self.backup_manager.database_name}.sqlite"
        if working_db.exists():
            final_db = final_location / f"{self.backup_manager.database_name}.sqlite"
            shutil.move(str(working_db), str(final_db))

        # Move FAISS index if present
        working_faiss = working_dir / f"{self.backup_manager.database_name}.faiss"
        if working_faiss.exists():
            final_faiss = final_location / f"{self.backup_manager.database_name}.faiss"
            shutil.move(str(working_faiss), str(final_faiss))


class PointInTimeRecoveryManager:
    """
    Point-in-time recovery (PITR) manager for LocalVectorDB.

    Provides the ability to restore a database to any point in time within
    the backup retention window by combining full and incremental backups.

    Parameters
    ----------
    backup_manager : BackupManager
        Parent backup manager instance
    incremental_manager : IncrementalBackupManager
        Incremental backup manager instance
    """

    def __init__(self, backup_manager: BackupManager, incremental_manager: IncrementalBackupManager):
        self.backup_manager = backup_manager
        self.incremental_manager = incremental_manager
        self.database_path = backup_manager.database_path
        self.config = backup_manager.config

    def get_recovery_timeline(self) -> List[Dict[str, Any]]:
        """
        Get the available recovery timeline showing all recovery points.

        Returns
        -------
        List[Dict[str, Any]]
            List of recovery points with timestamps and backup information
        """

        # Get all backups sorted by creation time
        backups = self.backup_manager.list_backups()

        recovery_points = []
        for backup in backups:
            recovery_points.append({
                'timestamp': backup.created_at,
                'backup_id': backup.backup_id,
                'backup_type': backup.backup_type.value,
                'parent_backup_id': backup.parent_backup_id,
                'database_version': backup.database_version,
                'size_bytes': backup.size_bytes
            })

        return sorted(recovery_points, key=lambda x: x['timestamp'])

    def find_recovery_point(
            self,
            target_timestamp: datetime,
            tolerance_minutes: int = 60
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best recovery point for a target timestamp.

        Parameters
        ----------
        target_timestamp : datetime
            Target timestamp for recovery
        tolerance_minutes : int
            Maximum tolerance in minutes to find a recovery point

        Returns
        -------
        Dict[str, Any] or None
            Recovery point information if found
        """

        timeline = self.get_recovery_timeline()

        # Find the latest backup at or before the target timestamp
        best_recovery_point = None
        min_time_diff = timedelta(minutes=tolerance_minutes + 1)

        for point in timeline:
            point_time = point['timestamp']
            if point_time <= target_timestamp:
                time_diff = target_timestamp - point_time
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_recovery_point = point

        if best_recovery_point and min_time_diff.total_seconds() <= tolerance_minutes * 60:
            return best_recovery_point

        return None

    def restore_to_point_in_time(
            self,
            target_timestamp: datetime,
            restore_location: Union[str, Path],
            tolerance_minutes: int = 60,
            dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Restore database to a specific point in time.

        Parameters
        ----------
        target_timestamp : datetime
            Target timestamp for recovery
        restore_location : Union[str, Path]
            Directory to restore to
        tolerance_minutes : int
            Maximum tolerance in minutes to find a recovery point
        dry_run : bool
            If True, only validate the recovery without actually restoring

        Returns
        -------
        Dict[str, Any]
            Recovery operation results with status and details
        """

        logger.info(f"Initiating point-in-time recovery to {target_timestamp}")

        # Find the best recovery point
        recovery_point = self.find_recovery_point(target_timestamp, tolerance_minutes)
        if not recovery_point:
            return {
                'success': False,
                'error': f'No recovery point found within {tolerance_minutes} minutes of {target_timestamp}',
                'target_timestamp': target_timestamp,
                'available_timeline': self.get_recovery_timeline()
            }

        # Calculate time difference
        actual_timestamp = recovery_point['timestamp']
        time_diff = abs(target_timestamp - actual_timestamp)

        logger.info(f"Found recovery point: {recovery_point['backup_id']} at {actual_timestamp} "
                    f"(diff: {time_diff.total_seconds():.1f} seconds)")

        if dry_run:
            return {
                'success': True,
                'dry_run': True,
                'target_timestamp': target_timestamp,
                'actual_timestamp': actual_timestamp,
                'time_difference_seconds': time_diff.total_seconds(),
                'recovery_point': recovery_point,
                'restore_location': str(restore_location),
                'recovery_chain': self._get_recovery_chain(recovery_point['backup_id'])
            }

        try:
            # Perform the actual recovery
            if recovery_point['backup_type'] == 'full':
                # Simple full backup restore
                restore_path = self.backup_manager.restore_backup(
                    recovery_point['backup_id'],
                    restore_location,
                    overwrite_existing=True
                )
            else:
                # Incremental backup chain restore
                restore_path = self.incremental_manager.restore_incremental_backup_chain(
                    recovery_point['backup_id'],
                    restore_location
                )

            return {
                'success': True,
                'target_timestamp': target_timestamp,
                'actual_timestamp': actual_timestamp,
                'time_difference_seconds': time_diff.total_seconds(),
                'recovery_point': recovery_point,
                'restore_location': str(restore_path),
                'recovery_chain': self._get_recovery_chain(recovery_point['backup_id'])
            }

        except Exception as e:
            logger.error(f"Point-in-time recovery failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'target_timestamp': target_timestamp,
                'recovery_point': recovery_point
            }

    def _get_recovery_chain(self, backup_id: str) -> List[Dict[str, Any]]:
        """Get the chain of backups needed for recovery."""

        try:
            backup_chain = self.incremental_manager._build_backup_chain(backup_id)
            return [
                {
                    'backup_id': backup.backup_id,
                    'backup_type': backup.backup_type.value,
                    'created_at': backup.created_at,
                    'database_version': backup.database_version
                }
                for backup in backup_chain
            ]
        except Exception:
            # Single backup (probably full)
            backup = self.backup_manager._find_backup_metadata(backup_id)
            if backup:
                return [{
                    'backup_id': backup.backup_id,
                    'backup_type': backup.backup_type.value,
                    'created_at': backup.created_at,
                    'database_version': backup.database_version
                }]
            return []

    def validate_recovery_timeline(self) -> Dict[str, Any]:
        """
        Validate the recovery timeline for consistency and completeness.

        Returns
        -------
        Dict[str, Any]
            Validation results with any issues found
        """

        logger.info("Validating recovery timeline")

        timeline = self.get_recovery_timeline()
        issues = []
        warnings = []

        # Check for gaps in the timeline
        full_backups = [p for p in timeline if p['backup_type'] == 'full']
        incremental_backups = [p for p in timeline if p['backup_type'] == 'incremental']

        if not full_backups:
            issues.append("No full backups found - recovery not possible")

        # Validate incremental backup chains
        for inc_backup in incremental_backups:
            parent_id = inc_backup.get('parent_backup_id')
            if parent_id:
                parent_found = any(p['backup_id'] == parent_id for p in timeline)
                if not parent_found:
                    issues.append(f"Incremental backup {inc_backup['backup_id']} references "
                                  f"missing parent {parent_id}")

        # Check for orphaned incremental backups
        for inc_backup in incremental_backups:
            try:
                chain = self._get_recovery_chain(inc_backup['backup_id'])
                if not any(c['backup_type'] == 'full' for c in chain):
                    warnings.append(f"Incremental backup {inc_backup['backup_id']} "
                                    f"has no full backup in its chain")
            except Exception as e:
                issues.append(f"Failed to validate chain for {inc_backup['backup_id']}: {e}")

        # Check for timeline gaps
        if len(timeline) > 1:
            for i in range(1, len(timeline)):
                prev_backup = timeline[i - 1]
                curr_backup = timeline[i]

                time_gap = curr_backup['timestamp'] - prev_backup['timestamp']
                if time_gap.total_seconds() > 24 * 3600:  # More than 24 hours
                    warnings.append(f"Large time gap ({time_gap.days} days) between "
                                    f"{prev_backup['backup_id']} and {curr_backup['backup_id']}")

        # Calculate recovery window
        recovery_window = None
        if timeline:
            earliest = min(p['timestamp'] for p in timeline)
            latest = max(p['timestamp'] for p in timeline)
            recovery_window = {
                'earliest': earliest,
                'latest': latest,
                'duration_days': (latest - earliest).days
            }

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_backups': len(timeline),
            'full_backups': len(full_backups),
            'incremental_backups': len(incremental_backups),
            'recovery_window': recovery_window,
            'timeline': timeline
        }

    def cleanup_recovery_timeline(
            self,
            max_age_days: Optional[int] = None,
            keep_full_backups: int = 3,
            dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Clean up old backups while maintaining recovery timeline integrity.

        Parameters
        ----------
        max_age_days : int, optional
            Maximum age of backups to keep. If None, uses config retention_days.
        keep_full_backups : int
            Minimum number of full backups to keep regardless of age
        dry_run : bool
            If True, only simulate cleanup without actually deleting

        Returns
        -------
        Dict[str, Any]
            Cleanup operation results
        """

        if max_age_days is None:
            max_age_days = self.config.retention_days

        cutoff_date = datetime.now(UTC) - timedelta(days=max_age_days)
        logger.info(f"Cleaning up recovery timeline (max_age: {max_age_days} days, "
                    f"keep_full: {keep_full_backups})")

        timeline = self.get_recovery_timeline()
        full_backups = [p for p in timeline if p['backup_type'] == 'full']
        full_backups.sort(key=lambda x: x['timestamp'], reverse=True)  # Newest first

        # Determine which backups to delete
        backups_to_delete = []
        backups_to_keep = []

        # Always keep the most recent full backups
        recent_full_backups = full_backups[:keep_full_backups]
        recent_full_ids = {b['backup_id'] for b in recent_full_backups}

        for backup in timeline:
            if backup['timestamp'] < cutoff_date:
                if backup['backup_type'] == 'full' and backup['backup_id'] in recent_full_ids:
                    # Keep recent full backups regardless of age
                    backups_to_keep.append(backup)
                elif backup['backup_type'] == 'incremental':
                    # Check if incremental backup depends on a kept full backup
                    try:
                        chain = self._get_recovery_chain(backup['backup_id'])
                        full_in_chain = any(c['backup_id'] in recent_full_ids and
                                            c['backup_type'] == 'full' for c in chain)
                        if full_in_chain:
                            backups_to_keep.append(backup)
                        else:
                            backups_to_delete.append(backup)
                    except Exception:
                        # If we can't determine the chain, delete it to be safe
                        backups_to_delete.append(backup)
                else:
                    # Old full backup beyond the keep limit
                    backups_to_delete.append(backup)
            else:
                # Recent backup - keep it
                backups_to_keep.append(backup)

        deleted_count = 0
        deletion_errors = []

        if not dry_run:
            for backup in backups_to_delete:
                try:
                    self.backup_manager.delete_backup(backup['backup_id'])
                    deleted_count += 1
                    logger.debug(f"Deleted backup: {backup['backup_id']}")
                except Exception as e:
                    deletion_errors.append(f"Failed to delete {backup['backup_id']}: {e}")

        return {
            'dry_run': dry_run,
            'total_backups': len(timeline),
            'backups_to_delete': len(backups_to_delete),
            'backups_to_keep': len(backups_to_keep),
            'deleted_count': deleted_count,
            'deletion_errors': deletion_errors,
            'kept_backups': backups_to_keep,
            'would_delete' if dry_run else 'deleted': backups_to_delete
        }

    def get_recovery_recommendations(
            self,
            target_timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Get recommendations for recovering to a specific point in time.

        Parameters
        ----------
        target_timestamp : datetime
            Target timestamp for recovery

        Returns
        -------
        Dict[str, Any]
            Recovery recommendations and analysis
        """

        timeline = self.get_recovery_timeline()

        if not timeline:
            return {
                'feasible': False,
                'reason': 'No backups available',
                'recommendations': ['Create initial full backup']
            }

        # Find closest recovery points
        before_points = [p for p in timeline if p['timestamp'] <= target_timestamp]
        after_points = [p for p in timeline if p['timestamp'] > target_timestamp]

        recommendations = []

        if before_points:
            closest_before = max(before_points, key=lambda x: x['timestamp'])
            time_diff_before = target_timestamp - closest_before['timestamp']

            recommendations.append({
                'option': 'restore_to_closest_before',
                'backup_id': closest_before['backup_id'],
                'timestamp': closest_before['timestamp'],
                'time_difference_seconds': time_diff_before.total_seconds(),
                'data_loss': 'None (exact recovery point)',
                'complexity': 'Low' if closest_before['backup_type'] == 'full' else 'Medium'
            })

        if after_points:
            closest_after = min(after_points, key=lambda x: x['timestamp'])
            time_diff_after = closest_after['timestamp'] - target_timestamp

            recommendations.append({
                'option': 'restore_to_closest_after',
                'backup_id': closest_after['backup_id'],
                'timestamp': closest_after['timestamp'],
                'time_difference_seconds': time_diff_after.total_seconds(),
                'data_loss': f"May include {time_diff_after.total_seconds():.0f} seconds of extra data",
                'complexity': 'Low' if closest_after['backup_type'] == 'full' else 'Medium'
            })

        # Determine feasibility
        feasible = len(before_points) > 0
        earliest = min(timeline, key=lambda x: x['timestamp'])
        latest = max(timeline, key=lambda x: x['timestamp'])

        return {
            'feasible': feasible,
            'target_timestamp': target_timestamp,
            'recovery_window': {
                'earliest': earliest['timestamp'],
                'latest': latest['timestamp']
            },
            'recommendations': recommendations,
            'timeline_summary': {
                'total_points': len(timeline),
                'full_backups': len([p for p in timeline if p['backup_type'] == 'full']),
                'incremental_backups': len([p for p in timeline if p['backup_type'] == 'incremental'])
            }
        }
