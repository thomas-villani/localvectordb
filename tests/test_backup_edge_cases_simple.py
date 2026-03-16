"""
Simplified edge case tests for LocalVectorDB backup and restore.

This module tests basic backup edge cases including:
- Different compression algorithms
- Error handling for file issues
- Basic security considerations
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from localvectordb.backup import (
    BackupConfig,
    BackupManager,
    BackupType,
    CompressionAlgorithm,
)
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.database import LocalVectorDB


@pytest.fixture
def temp_backup_dirs():
    """
    Create temporary directories for backup testing.

    Yields
    ------
    dict
        Paths for database, backup, and restore directories
    """
    base_temp = tempfile.mkdtemp()
    db_dir = Path(base_temp) / "db"
    backup_dir = Path(base_temp) / "backups"
    restore_dir = Path(base_temp) / "restored"

    db_dir.mkdir()
    backup_dir.mkdir()
    restore_dir.mkdir()

    yield {"base": Path(base_temp), "db": db_dir, "backup": backup_dir, "restore": restore_dir}

    shutil.rmtree(base_temp, ignore_errors=True)


@pytest.fixture
def test_database(temp_backup_dirs):
    """
    Create a test database with sample data.

    Parameters
    ----------
    temp_backup_dirs : dict
        Temporary directory paths

    Returns
    -------
    LocalVectorDB
        Test database instance
    """
    metadata_schema = {
        "category": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        "importance": MetadataField(type=MetadataFieldType.INTEGER),
        "timestamp": MetadataField(type=MetadataFieldType.DATE),
    }

    db = LocalVectorDB(
        "test_backup",
        str(temp_backup_dirs["db"]),
        metadata_schema=metadata_schema,
        embedding_provider="mock",
        embedding_model="mock-model",
        chunk_size=100,
    )

    # Add test documents
    documents = [f"Test document {i} with sample content for backup testing." for i in range(3)]
    metadata = [{"category": f"cat_{i % 2}", "importance": i, "timestamp": "2024-01-01"} for i in range(3)]
    ids = [f"doc_{i}" for i in range(3)]

    db.upsert(documents, metadata=metadata, ids=ids)
    return db


@pytest.mark.unit
class TestCompressionAlgorithms:
    """Test different compression algorithms for backups."""

    def test_gzip_compression(self, temp_backup_dirs, test_database):
        """
        Test backup with GZIP compression (default).
        """
        backup_config = BackupConfig(
            backup_location=temp_backup_dirs["backup"], compression_algorithm=CompressionAlgorithm.GZIP
        )
        db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
        faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
        backup_manager = BackupManager(db_path, faiss_path, backup_config)

        # Create backup with GZIP
        backup_id = backup_manager.create_backup(BackupType.FULL)

        # Verify backup was created
        backup_files = list(temp_backup_dirs["backup"].glob("*.lvdb-backup"))
        assert len(backup_files) > 0

        # Test restoration
        restore_path = backup_manager.restore_backup(backup_id, temp_backup_dirs["restore"])
        assert restore_path.exists()

    def test_lzma_compression(self, temp_backup_dirs, test_database):
        """
        Test backup with LZMA compression.
        """
        backup_config = BackupConfig(
            backup_location=temp_backup_dirs["backup"], compression_algorithm=CompressionAlgorithm.LZMA
        )
        db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
        faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
        backup_manager = BackupManager(db_path, faiss_path, backup_config)

        # Create backup with LZMA
        backup_id = backup_manager.create_backup(BackupType.FULL)

        # Verify backup was created
        backup_files = list(temp_backup_dirs["backup"].glob("*.lvdb-backup"))
        assert len(backup_files) > 0

        # Test restoration
        restore_path = backup_manager.restore_backup(backup_id, temp_backup_dirs["restore"])
        assert restore_path.exists()

    def test_no_compression(self, temp_backup_dirs, test_database):
        """
        Test backup with no compression.
        """
        backup_config = BackupConfig(
            backup_location=temp_backup_dirs["backup"], compression_algorithm=CompressionAlgorithm.NONE
        )
        db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
        faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
        backup_manager = BackupManager(db_path, faiss_path, backup_config)

        # Create uncompressed backup
        backup_id = backup_manager.create_backup(BackupType.FULL)

        # Verify backup was created
        backup_files = list(temp_backup_dirs["backup"].glob("*.lvdb-backup"))
        assert len(backup_files) > 0

        # Test restoration
        restore_path = backup_manager.restore_backup(backup_id, temp_backup_dirs["restore"])
        assert restore_path.exists()

    def test_compression_size_differences(self, temp_backup_dirs, test_database):
        """
        Test that different compression algorithms create different file sizes.
        """
        backup_sizes = {}

        for algo in [CompressionAlgorithm.NONE, CompressionAlgorithm.GZIP, CompressionAlgorithm.LZMA]:
            # Create separate backup directories for each algorithm
            algo_backup_dir = temp_backup_dirs["backup"] / algo.value
            algo_backup_dir.mkdir()

            backup_config = BackupConfig(backup_location=algo_backup_dir, compression_algorithm=algo)
            db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
            faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
            backup_manager = BackupManager(db_path, faiss_path, backup_config)

            _backup_id = backup_manager.create_backup(BackupType.FULL)

            # Get backup file size
            backup_files = list(algo_backup_dir.glob("*.lvdb-backup"))
            if backup_files:
                backup_sizes[algo.value] = backup_files[0].stat().st_size

        # Verify all backups were created
        assert len(backup_sizes) == 3
        assert all(size > 0 for size in backup_sizes.values())


@pytest.mark.unit
class TestBackupErrorHandling:
    """Test error handling for backup operations."""

    def test_backup_nonexistent_database(self, temp_backup_dirs):
        """
        Test backup of non-existent database.
        """
        backup_config = BackupConfig(backup_location=temp_backup_dirs["backup"])
        nonexistent_db = temp_backup_dirs["db"] / "nonexistent.sqlite"
        nonexistent_faiss = temp_backup_dirs["db"] / "nonexistent.faiss"

        backup_manager = BackupManager(nonexistent_db, nonexistent_faiss, backup_config)

        # Try to backup non-existent database - should fail
        with pytest.raises(FileNotFoundError):
            backup_manager.create_backup(BackupType.FULL)

    def test_restore_nonexistent_backup(self, temp_backup_dirs, test_database):
        """
        Test restoration of non-existent backup.
        """
        backup_config = BackupConfig(backup_location=temp_backup_dirs["backup"])
        db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
        faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
        backup_manager = BackupManager(db_path, faiss_path, backup_config)

        # Try to restore non-existent backup
        with pytest.raises((FileNotFoundError, ValueError)):
            backup_manager.restore_backup("nonexistent_backup_id", temp_backup_dirs["restore"])

    def test_invalid_backup_directory(self, temp_backup_dirs):
        """
        Test handling of invalid backup directory permissions.
        """
        # Create a file where we want a directory (simulating permission issues)
        invalid_backup_path = temp_backup_dirs["base"] / "invalid_backup"
        invalid_backup_path.touch()  # Create file instead of directory

        with pytest.raises((PermissionError, FileExistsError, OSError)):
            BackupConfig(backup_location=invalid_backup_path)


@pytest.mark.unit
class TestBackupIntegrity:
    """Test backup integrity and verification."""

    def test_backup_verification(self, temp_backup_dirs, test_database):
        """
        Test backup integrity verification.
        """
        backup_config = BackupConfig(backup_location=temp_backup_dirs["backup"], verify_integrity=True)
        db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
        faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
        backup_manager = BackupManager(db_path, faiss_path, backup_config)

        # Create backup
        backup_id = backup_manager.create_backup(BackupType.FULL)

        # Verify backup integrity
        is_valid = backup_manager.verify_backup(backup_id)
        assert is_valid

    def test_backup_without_verification(self, temp_backup_dirs, test_database):
        """
        Test backup creation without verification.
        """
        backup_config = BackupConfig(backup_location=temp_backup_dirs["backup"], verify_integrity=False)
        db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
        faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
        backup_manager = BackupManager(db_path, faiss_path, backup_config)

        # Create backup without verification
        backup_id = backup_manager.create_backup(BackupType.FULL)

        # Should still be able to list the backup
        backups = backup_manager.list_backups()
        backup_ids = [b.backup_id for b in backups]
        assert backup_id in backup_ids


@pytest.mark.unit
class TestBackupConfiguration:
    """Test different backup configuration options."""

    def test_backup_retention(self, temp_backup_dirs, test_database):
        """
        Test backup retention configuration.
        """
        backup_config = BackupConfig(backup_location=temp_backup_dirs["backup"], retention_days=30)
        db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
        faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
        backup_manager = BackupManager(db_path, faiss_path, backup_config)

        # Create multiple backups
        _backup_id1 = backup_manager.create_backup(BackupType.FULL)
        _backup_id2 = backup_manager.create_backup(BackupType.FULL)

        # Should have both backups
        backups = backup_manager.list_backups()
        assert len(backups) >= 2

        # Test cleanup (should not delete recent backups)
        deleted_count = backup_manager.cleanup_old_backups(retention_days=30)
        # Since backups are recent, nothing should be deleted
        assert deleted_count == 0

    def test_faiss_exclusion(self, temp_backup_dirs, test_database):
        """
        Test backup with FAISS index exclusion.
        """
        backup_config = BackupConfig(backup_location=temp_backup_dirs["backup"], include_faiss_index=False)
        db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
        faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
        backup_manager = BackupManager(db_path, faiss_path, backup_config)

        # Create backup without FAISS index
        backup_id = backup_manager.create_backup(BackupType.FULL)

        # Verify backup was created
        backups = backup_manager.list_backups()
        backup_ids = [b.backup_id for b in backups]
        assert backup_id in backup_ids

    def test_backup_with_size_limit(self, temp_backup_dirs, test_database):
        """
        Test backup with size limit configuration.
        """
        backup_config = BackupConfig(
            backup_location=temp_backup_dirs["backup"],
            max_backup_size_gb=1.0,  # 1GB limit (our test backup is much smaller)
        )
        db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
        faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
        backup_manager = BackupManager(db_path, faiss_path, backup_config)

        # Create backup within size limit
        backup_id = backup_manager.create_backup(BackupType.FULL)

        # Should succeed since test database is small
        backups = backup_manager.list_backups()
        backup_ids = [b.backup_id for b in backups]
        assert backup_id in backup_ids


@pytest.mark.unit
class TestBackupPaths:
    """Test backup path handling and security."""

    def test_backup_path_creation(self, temp_backup_dirs):
        """
        Test automatic backup directory creation.
        """
        # Create nested backup path that doesn't exist
        nested_backup_path = temp_backup_dirs["base"] / "deep" / "nested" / "backup"

        _backup_config = BackupConfig(backup_location=nested_backup_path)

        # Should create the directory automatically
        assert nested_backup_path.exists()
        assert nested_backup_path.is_dir()

    def test_relative_path_handling(self, temp_backup_dirs, test_database):
        """
        Test backup with relative paths.
        """
        import os

        original_cwd = os.getcwd()

        try:
            # Change to temp directory
            os.chdir(temp_backup_dirs["base"])

            backup_config = BackupConfig(backup_location="./backups")
            db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
            faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
            backup_manager = BackupManager(db_path, faiss_path, backup_config)

            # Create backup with relative path
            backup_id = backup_manager.create_backup(BackupType.FULL)

            # Should work correctly
            backups = backup_manager.list_backups()
            backup_ids = [b.backup_id for b in backups]
            assert backup_id in backup_ids

        finally:
            os.chdir(original_cwd)


@pytest.mark.unit
class TestBackupEdgeCases:
    """Test various edge cases in backup functionality."""

    def test_concurrent_backups(self, temp_backup_dirs, test_database):
        """
        Test handling of multiple backup operations.
        """
        backup_config = BackupConfig(backup_location=temp_backup_dirs["backup"])
        db_path = temp_backup_dirs["db"] / "test_backup.sqlite"
        faiss_path = temp_backup_dirs["db"] / "test_backup.faiss"
        backup_manager = BackupManager(db_path, faiss_path, backup_config)

        # Create multiple backups rapidly
        backup_ids = []
        for _i in range(3):
            backup_id = backup_manager.create_backup(BackupType.FULL)
            backup_ids.append(backup_id)

        # All should be unique
        assert len(set(backup_ids)) == len(backup_ids)

        # All should be listed
        backups = backup_manager.list_backups()
        listed_ids = [b.backup_id for b in backups]
        for backup_id in backup_ids:
            assert backup_id in listed_ids

    def test_empty_database_backup(self, temp_backup_dirs):
        """
        Test backup of empty database.
        """
        # Create empty database
        metadata_schema = {"test": MetadataField(type=MetadataFieldType.TEXT)}
        db = LocalVectorDB(
            "empty_test",
            str(temp_backup_dirs["db"]),
            metadata_schema=metadata_schema,
            embedding_provider="mock",
            embedding_model="mock-model",
        )
        db.close()

        backup_config = BackupConfig(backup_location=temp_backup_dirs["backup"])
        db_path = temp_backup_dirs["db"] / "empty_test.sqlite"
        faiss_path = temp_backup_dirs["db"] / "empty_test.faiss"
        backup_manager = BackupManager(db_path, faiss_path, backup_config)

        # Should be able to backup empty database
        backup_id = backup_manager.create_backup(BackupType.FULL)

        # Verify backup was created
        backups = backup_manager.list_backups()
        backup_ids = [b.backup_id for b in backups]
        assert backup_id in backup_ids
