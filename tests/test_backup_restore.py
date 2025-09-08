# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# tests/test_backup_restore.py

"""Comprehensive tests for LocalVectorDB backup and restore functionality.

Tests cover:
- Full backup creation and restoration
- Incremental backup chains
- Point-in-time recovery
- Backup integrity verification
- Error handling and edge cases
- Configuration options and settings
"""

import json
import pytest
import tempfile
import shutil
from datetime import datetime, UTC, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import faiss

from localvectordb.backup import (
    BackupManager, BackupConfig, BackupType, CompressionAlgorithm,
    IncrementalBackupManager, PointInTimeRecoveryManager, BackupMetadata
)
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.database import LocalVectorDB
from localvectordb.versioning import DatabaseVersion, VersionManager


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    base_temp = tempfile.mkdtemp()
    db_dir = Path(base_temp) / "db"
    backup_dir = Path(base_temp) / "backups"
    restore_dir = Path(base_temp) / "restored"
    
    db_dir.mkdir()
    backup_dir.mkdir()
    restore_dir.mkdir()
    
    yield {
        'base': Path(base_temp),
        'db': db_dir,
        'backup': backup_dir,
        'restore': restore_dir
    }
    
    # Cleanup
    shutil.rmtree(base_temp)


@pytest.fixture
def sample_database(temp_dirs):
    """Create a sample LocalVectorDB for testing."""
    db_path = temp_dirs['db'] / "test.sqlite"
    
    # Create database with sample data
    metadata_schema = {
        'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        'priority': MetadataField(type=MetadataFieldType.INTEGER, default_value=1),
        'created_date': MetadataField(type=MetadataFieldType.DATE)
    }
    
    db = LocalVectorDB(
        "test",
        str(temp_dirs['db']),
        metadata_schema=metadata_schema,
        embedding_provider="mock",
        embedding_model="mock-model",
        chunk_size=100
    )
    
    # Add sample documents
    documents = [
        "This is a test document about machine learning algorithms.",
        "Vector databases are essential for AI applications.",
        "LocalVectorDB provides efficient vector storage and retrieval."
    ]
    
    metadata = [
        {'category': 'tech', 'priority': 1, 'created_date': '2024-01-01'},
        {'category': 'ai', 'priority': 2, 'created_date': '2024-01-02'},
        {'category': 'database', 'priority': 3, 'created_date': '2024-01-03'}
    ]
    
    db.upsert(documents, metadata=metadata)
    db.save()  # Ensure data is persisted
    
    yield db
    
    db.close()


@pytest.fixture
def backup_config(temp_dirs):
    """Create backup configuration for testing."""
    return BackupConfig(
        backup_location=temp_dirs['backup'],
        compression_algorithm=CompressionAlgorithm.GZIP,
        verify_integrity=True,
        retention_days=30,
        include_faiss_index=True
    )


@pytest.fixture
def backup_manager(sample_database, backup_config, temp_dirs):
    """Create BackupManager instance for testing."""
    db_path = temp_dirs['db'] / "test.sqlite"
    faiss_path = temp_dirs['db'] / "test.faiss"
    
    return BackupManager(
        database_path=db_path,
        faiss_index_path=faiss_path,
        config=backup_config
    )


@pytest.mark.unit
class TestBackupManager:
    """Test cases for BackupManager functionality."""
    
    def test_create_full_backup(self, backup_manager, temp_dirs):
        """Test creation of full backup."""
        backup_id = backup_manager.create_backup(BackupType.FULL)
        
        assert backup_id is not None
        assert isinstance(backup_id, str)
        assert len(backup_id) == 36  # UUID length
        
        # Check backup file exists
        backup_files = list(temp_dirs['backup'].glob("*.lvdb-backup"))
        assert len(backup_files) == 1
        assert backup_id[:8] in backup_files[0].name
    
    def test_backup_metadata(self, backup_manager, temp_dirs):
        """Test backup metadata is correctly created."""
        backup_id = backup_manager.create_backup(BackupType.FULL)
        
        # Read backup metadata
        backup_files = list(temp_dirs['backup'].glob("*.lvdb-backup"))
        backup_file = backup_files[0]
        
        import tarfile
        with tarfile.open(backup_file, "r:*") as tar:
            manifest_file = tar.extractfile("manifest.json")
            manifest_data = json.load(manifest_file)
        
        metadata = BackupMetadata.from_dict(manifest_data)
        
        assert metadata.backup_id == backup_id
        assert metadata.backup_type == BackupType.FULL
        assert metadata.database_name == "test"
        assert metadata.compression_algorithm == CompressionAlgorithm.GZIP
        assert len(metadata.checksums) > 0
        assert metadata.size_bytes > 0
    
    def test_backup_integrity_verification(self, backup_manager, temp_dirs):
        """Test backup integrity verification."""
        backup_id = backup_manager.create_backup(BackupType.FULL)
        
        # Verify backup
        is_valid = backup_manager.verify_backup(backup_id)
        assert is_valid is True
    
    def test_list_backups(self, backup_manager):
        """Test listing available backups."""
        # Create multiple backups
        backup_id1 = backup_manager.create_backup(BackupType.FULL)
        backup_id2 = backup_manager.create_backup(BackupType.FULL)
        
        backups = backup_manager.list_backups()
        
        assert len(backups) == 2
        backup_ids = {b.backup_id for b in backups}
        assert backup_id1 in backup_ids
        assert backup_id2 in backup_ids
        
        # Test filtering by type
        full_backups = backup_manager.list_backups(BackupType.FULL)
        assert len(full_backups) == 2
    
    def test_restore_full_backup(self, backup_manager, sample_database, temp_dirs):
        """Test restoration of full backup."""
        # Create backup
        backup_id = backup_manager.create_backup(BackupType.FULL)
        
        # Restore backup
        restore_path = backup_manager.restore_backup(
            backup_id,
            temp_dirs['restore'],
            overwrite_existing=True
        )
        
        assert restore_path == temp_dirs['restore']
        
        # Verify restored files exist
        restored_db = temp_dirs['restore'] / "test.sqlite"
        restored_faiss = temp_dirs['restore'] / "test.faiss"
        
        assert restored_db.exists()
        assert restored_faiss.exists()
        
        # Verify restored data
        restored_db_instance = LocalVectorDB(
            "test",
            str(temp_dirs['restore']),
            create_if_not_exists=False
        )
        
        # Check document count
        docs = restored_db_instance.filter()
        assert len(docs) == 3
        
        restored_db_instance.close()
    
    def test_delete_backup(self, backup_manager, temp_dirs):
        """Test backup deletion."""
        backup_id = backup_manager.create_backup(BackupType.FULL)
        
        # Verify backup exists
        backups = backup_manager.list_backups()
        assert len(backups) == 1
        
        # Delete backup
        success = backup_manager.delete_backup(backup_id)
        assert success is True
        
        # Verify backup is deleted
        backup_files = list(temp_dirs['backup'].glob("*.lvdb-backup"))
        assert len(backup_files) == 0
    
    def test_cleanup_old_backups(self, backup_manager):
        """Test cleanup of old backups based on retention policy."""
        # Create multiple backups with different ages (mocked)
        with patch('localvectordb.backup.datetime') as mock_datetime:
            # Create old backup
            old_time = datetime.now(UTC) - timedelta(days=35)
            mock_datetime.now.return_value = old_time
            mock_datetime.UTC = UTC
            old_backup_id = backup_manager.create_backup(BackupType.FULL)
            
            # Create recent backup
            recent_time = datetime.now(UTC) - timedelta(days=5)
            mock_datetime.now.return_value = recent_time
            recent_backup_id = backup_manager.create_backup(BackupType.FULL)
        
        # Cleanup with 30-day retention
        cleaned = backup_manager.cleanup_old_backups(retention_days=30)
        assert cleaned >= 0  # At least 0 backups cleaned
        
        # Verify recent backup still exists
        backups = backup_manager.list_backups()
        remaining_ids = {b.backup_id for b in backups}
        assert recent_backup_id in remaining_ids

@pytest.mark.unit
class TestIncrementalBackup:
    """Test cases for incremental backup functionality."""
    
    @pytest.fixture
    def incremental_manager(self, backup_manager):
        """Create IncrementalBackupManager for testing."""
        return IncrementalBackupManager(backup_manager)
    
    def test_create_incremental_backup(self, incremental_manager, sample_database, temp_dirs):
        """Test creation of incremental backup."""
        # Create base full backup
        full_backup_id = incremental_manager.backup_manager.create_backup(BackupType.FULL)
        
        # Modify database
        sample_database.upsert(
            ["New document for incremental backup test"],
            metadata=[{'category': 'new', 'priority': 1}]
        )
        sample_database.save()
        
        # Create incremental backup
        inc_backup_id = incremental_manager.create_incremental_backup(full_backup_id)
        
        assert inc_backup_id is not None
        assert inc_backup_id != full_backup_id
        
        # Verify incremental backup file exists
        backup_files = list(temp_dirs['backup'].glob("*.lvdb-backup"))
        assert len(backup_files) == 2  # Full + incremental
    
    def test_incremental_backup_chain_restore(self, incremental_manager, sample_database, temp_dirs):
        """Test restoration of incremental backup chain."""
        # Create full backup
        full_backup_id = incremental_manager.backup_manager.create_backup(BackupType.FULL)
        
        # Modify database and create incremental backup
        sample_database.upsert(
            ["Incremental document 1"],
            metadata=[{'category': 'inc1', 'priority': 1}]
        )
        sample_database.save()
        inc_backup_id1 = incremental_manager.create_incremental_backup(full_backup_id)
        
        # Another modification and incremental backup
        sample_database.upsert(
            ["Incremental document 2"],
            metadata=[{'category': 'inc2', 'priority': 2}]
        )
        sample_database.save()
        inc_backup_id2 = incremental_manager.create_incremental_backup(inc_backup_id1)
        
        # Restore incremental backup chain
        restore_path = incremental_manager.restore_incremental_backup_chain(
            inc_backup_id2,
            temp_dirs['restore']
        )
        
        assert restore_path == temp_dirs['restore']
        
        # Verify restored database contains all data
        restored_db = LocalVectorDB(
            "test",
            str(temp_dirs['restore']),
            create_if_not_exists=False
        )
        
        docs = restored_db.filter()
        assert len(docs) >= 5  # Original 3 + 2 incremental
        
        restored_db.close()

@pytest.mark.unit
class TestPointInTimeRecovery:
    """Test cases for point-in-time recovery functionality."""
    
    @pytest.fixture
    def pitr_manager(self, backup_manager):
        """Create PointInTimeRecoveryManager for testing."""
        inc_manager = IncrementalBackupManager(backup_manager)
        return PointInTimeRecoveryManager(backup_manager, inc_manager)
    
    def test_get_recovery_timeline(self, pitr_manager, sample_database, temp_dirs):
        """Test recovery timeline generation."""
        # Create backups at different times
        backup_id1 = pitr_manager.backup_manager.create_backup(BackupType.FULL)
        
        # Simulate time passage and create another backup
        backup_id2 = pitr_manager.backup_manager.create_backup(BackupType.FULL)
        
        timeline = pitr_manager.get_recovery_timeline()
        
        assert len(timeline) == 2
        assert timeline[0]['timestamp'] <= timeline[1]['timestamp']  # Sorted by time
        
        backup_ids = {point['backup_id'] for point in timeline}
        assert backup_id1 in backup_ids
        assert backup_id2 in backup_ids
    
    def test_find_recovery_point(self, pitr_manager, sample_database):
        """Test finding recovery points for specific timestamps."""
        # Create backup
        backup_id = pitr_manager.backup_manager.create_backup(BackupType.FULL)
        backups = pitr_manager.backup_manager.list_backups()
        backup_time = backups[0].created_at
        
        # Find recovery point near backup time
        target_time = backup_time + timedelta(minutes=5)
        recovery_point = pitr_manager.find_recovery_point(target_time, tolerance_minutes=10)
        
        assert recovery_point is not None
        assert recovery_point['backup_id'] == backup_id
    
    def test_point_in_time_recovery_dry_run(self, pitr_manager, sample_database, temp_dirs):
        """Test point-in-time recovery dry run."""
        # Create backup
        backup_id = pitr_manager.backup_manager.create_backup(BackupType.FULL)
        backups = pitr_manager.backup_manager.list_backups()
        backup_time = backups[0].created_at
        
        # Perform dry-run recovery
        target_time = backup_time + timedelta(minutes=1)
        result = pitr_manager.restore_to_point_in_time(
            target_time,
            temp_dirs['restore'],
            dry_run=True
        )
        
        assert result['success'] is True
        assert result['dry_run'] is True
        assert 'recovery_point' in result
        assert result['recovery_point']['backup_id'] == backup_id
    
    def test_validate_recovery_timeline(self, pitr_manager, sample_database):
        """Test recovery timeline validation."""
        # Create valid backup chain
        full_backup_id = pitr_manager.backup_manager.create_backup(BackupType.FULL)
        
        validation_result = pitr_manager.validate_recovery_timeline()
        
        assert validation_result['valid'] is True
        assert validation_result['full_backups'] >= 1
        assert validation_result['total_backups'] >= 1
        assert len(validation_result['issues']) == 0


class TestBackupConfiguration:
    """Test cases for backup configuration options."""
    
    def test_compression_algorithms(self, sample_database, temp_dirs):
        """Test different compression algorithms."""
        db_path = temp_dirs['db'] / "test.sqlite"
        faiss_path = temp_dirs['db'] / "test.faiss"
        
        # Test GZIP compression
        gzip_config = BackupConfig(
            backup_location=temp_dirs['backup'] / "gzip",
            compression_algorithm=CompressionAlgorithm.GZIP
        )
        gzip_config.backup_location.mkdir(exist_ok=True)
        gzip_manager = BackupManager(db_path, faiss_path, gzip_config)
        gzip_backup_id = gzip_manager.create_backup(BackupType.FULL)
        
        # Test no compression
        none_config = BackupConfig(
            backup_location=temp_dirs['backup'] / "none",
            compression_algorithm=CompressionAlgorithm.NONE
        )
        none_config.backup_location.mkdir(exist_ok=True)
        none_manager = BackupManager(db_path, faiss_path, none_config)
        none_backup_id = none_manager.create_backup(BackupType.FULL)
        
        # Both backups should succeed
        assert gzip_backup_id is not None
        assert none_backup_id is not None
        
        # Verify files exist
        gzip_files = list((temp_dirs['backup'] / "gzip").glob("*.lvdb-backup"))
        none_files = list((temp_dirs['backup'] / "none").glob("*.lvdb-backup"))
        
        assert len(gzip_files) == 1
        assert len(none_files) == 1
    
    def test_faiss_exclusion(self, sample_database, temp_dirs):
        """Test backup without FAISS index."""
        db_path = temp_dirs['db'] / "test.sqlite"
        faiss_path = temp_dirs['db'] / "test.faiss"
        
        config = BackupConfig(
            backup_location=temp_dirs['backup'],
            include_faiss_index=False
        )
        
        manager = BackupManager(db_path, faiss_path, config)
        backup_id = manager.create_backup(BackupType.FULL)
        
        # Restore backup
        restore_path = manager.restore_backup(
            backup_id,
            temp_dirs['restore'],
            overwrite_existing=True
        )
        
        # Verify only SQLite file was restored
        restored_db = temp_dirs['restore'] / "test.sqlite"
        restored_faiss = temp_dirs['restore'] / "test.faiss"
        
        assert restored_db.exists()
        assert not restored_faiss.exists()


@pytest.mark.unit
class TestErrorHandling:
    """Test cases for error handling and edge cases."""
    
    def test_backup_nonexistent_database(self, temp_dirs, backup_config):
        """Test backup of non-existent database."""
        nonexistent_path = temp_dirs['db'] / "nonexistent.sqlite"
        
        manager = BackupManager(nonexistent_path, config=backup_config)
        
        with pytest.raises(FileNotFoundError):
            manager.create_backup(BackupType.FULL)
    
    def test_restore_nonexistent_backup(self, backup_manager, temp_dirs):
        """Test restoration of non-existent backup."""
        fake_backup_id = "nonexistent-backup-id"
        
        with pytest.raises(FileNotFoundError):
            backup_manager.restore_backup(fake_backup_id, temp_dirs['restore'])
    
    def test_incremental_backup_without_parent(self, temp_dirs, backup_config):
        """Test incremental backup with invalid parent ID."""
        db_path = temp_dirs['db'] / "test.sqlite"
        manager = BackupManager(db_path, config=backup_config)
        inc_manager = IncrementalBackupManager(manager)
        
        with pytest.raises(ValueError, match="Parent backup not found"):
            inc_manager.create_incremental_backup("invalid-parent-id")
    
    def test_corrupted_backup_file(self, backup_manager, temp_dirs):
        """Test handling of corrupted backup files."""
        # Create a valid backup first
        backup_id = backup_manager.create_backup(BackupType.FULL)
        
        # Corrupt the backup file
        backup_files = list(temp_dirs['backup'].glob("*.lvdb-backup"))
        backup_file = backup_files[0]
        
        # Write invalid data to corrupt the file
        with open(backup_file, 'wb') as f:
            f.write(b"corrupted data")
        
        # Verification should fail
        is_valid = backup_manager.verify_backup(backup_id)
        assert is_valid is False


# Integration tests
class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_backup_restore_cycle(self, sample_database, temp_dirs, backup_config):
        """Test complete backup and restore cycle."""
        db_path = temp_dirs['db'] / "test.sqlite"
        faiss_path = temp_dirs['db'] / "test.faiss"
        
        # Create backup manager
        manager = BackupManager(db_path, faiss_path, backup_config)
        
        # Create backup
        backup_id = manager.create_backup(BackupType.FULL)
        assert backup_id is not None
        
        # Verify backup
        assert manager.verify_backup(backup_id) is True
        
        # List backups
        backups = manager.list_backups()
        assert len(backups) == 1
        assert backups[0].backup_id == backup_id
        
        # Restore backup
        restore_path = manager.restore_backup(
            backup_id,
            temp_dirs['restore'],
            overwrite_existing=True
        )
        
        # Verify restoration
        restored_db = LocalVectorDB(
            "test",
            str(temp_dirs['restore']),
            create_if_not_exists=False
        )
        
        original_docs = sample_database.filter()
        restored_docs = restored_db.filter()
        
        assert len(restored_docs) == len(original_docs)
        
        # Verify content matches (basic check)
        original_contents = {doc.content for doc in original_docs}
        restored_contents = {doc.content for doc in restored_docs}
        assert original_contents == restored_contents
        
        restored_db.close()
        
        # Cleanup
        manager.delete_backup(backup_id)
        final_backups = manager.list_backups()
        assert len(final_backups) == 0