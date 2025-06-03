# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# tests/test_keymanager.py
"""
Tests for localvectordb_server.keymanager module.
"""

import pytest
import sqlite3
import os
import tempfile
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from localvectordb_server.keymanager import KeyManager, KeyRecord, get_key_manager


@pytest.fixture
def sample_key_record():
    """Create a sample KeyRecord for testing."""
    return KeyRecord(
        id="key_20240101_abc123",
        key_hash="$2b$12$hashed_key_value",
        description="Test API Key",
        created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        expires_at=datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC),
        last_used=datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC),
        active=True,
        created_by="test_user"
    )


@pytest.fixture
def expired_key_record():
    """Create an expired KeyRecord for testing."""
    return KeyRecord(
        id="key_20230101_xyz789",
        key_hash="$2b$12$expired_key_hash",
        description="Expired Test Key",
        created_at=datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC),
        expires_at=datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC),
        last_used=datetime(2023, 6, 1, 10, 0, 0, tzinfo=UTC),
        active=True,
        created_by="test_user"
    )


@pytest.fixture
def mock_key_manager(temp_dir):
    """Create a KeyManager with mocked bcrypt operations."""
    db_path = temp_dir / "test_keys.db"

    with patch('localvectordb_server.keymanager.bcrypt') as mock_bcrypt:
        # Mock bcrypt operations for consistent testing
        mock_bcrypt.gensalt.return_value = b'$2b$12$mock_salt'
        mock_bcrypt.hashpw.return_value = b'$2b$12$mock_hashed_password'
        mock_bcrypt.checkpw.return_value = True

        manager = KeyManager(str(db_path))
        yield manager, mock_bcrypt


class TestKeyRecord:
    """Test KeyRecord data class."""

    def test_key_record_creation(self, sample_key_record):
        """Test KeyRecord creation with all fields."""
        assert sample_key_record.id == "key_20240101_abc123"
        assert sample_key_record.description == "Test API Key"
        assert sample_key_record.active is True
        assert sample_key_record.created_by == "test_user"

    def test_key_record_is_expired_false(self, sample_key_record):
        """Test is_expired property for non-expired key."""
        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 6, 1, tzinfo=UTC)
            assert sample_key_record.is_expired is False

    def test_key_record_is_expired_true(self, expired_key_record):
        """Test is_expired property for expired key."""
        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 6, 1, tzinfo=UTC)
            assert expired_key_record.is_expired is True

    def test_key_record_is_expired_no_expiry(self):
        """Test is_expired property for key with no expiration."""
        key = KeyRecord(
            id="key_never_expires",
            key_hash="hash",
            expires_at=None
        )
        assert key.is_expired is False

    def test_days_until_expiry(self, sample_key_record):
        """Test days_until_expiry calculation."""
        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 12, 25, tzinfo=UTC)
            days = sample_key_record.days_until_expiry
            assert days == 6  # Dec 31 - Dec 25 = 6 days

    def test_days_until_expiry_expired(self, expired_key_record):
        """Test days_until_expiry for expired key."""
        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 6, 1, tzinfo=UTC)
            days = expired_key_record.days_until_expiry
            assert days == 0  # Should be 0 for expired keys

    def test_days_until_expiry_no_expiry(self):
        """Test days_until_expiry for key with no expiration."""
        key = KeyRecord(
            id="key_never_expires",
            key_hash="hash",
            expires_at=None
        )
        assert key.days_until_expiry is None

    def test_to_dict(self, sample_key_record):
        """Test to_dict serialization."""
        result = sample_key_record.to_dict()

        expected_keys = {
            'id', 'description', 'created_at', 'expires_at',
            'last_used', 'active', 'created_by', 'is_expired',
            'days_until_expiry'
        }
        assert set(result.keys()) == expected_keys
        assert result['id'] == "key_20240101_abc123"
        assert result['description'] == "Test API Key"
        assert result['active'] is True

    def test_from_db_row(self):
        """Test from_db_row class method."""
        # Create a mock SQLite row
        mock_row = {
            'id': 'test_key_id',
            'key_hash': 'test_hash',
            'description': 'Test Description',
            'created_at': '2024-01-01T12:00:00+00:00',
            'expires_at': '2024-12-31T23:59:59+00:00',
            'last_used': '2024-06-01T10:00:00+00:00',
            'active': 1,  # SQLite stores boolean as integer
            'created_by': 'test_user'
        }

        key_record = KeyRecord.from_db_row(mock_row)

        assert key_record.id == 'test_key_id'
        assert key_record.key_hash == 'test_hash'
        assert key_record.description == 'Test Description'
        assert key_record.active is True
        assert key_record.created_by == 'test_user'
        assert isinstance(key_record.created_at, datetime)
        assert isinstance(key_record.expires_at, datetime)

    def test_from_db_row_with_none_values(self):
        """Test from_db_row with None values."""
        mock_row = {
            'id': 'test_key_id',
            'key_hash': 'test_hash',
            'description': None,
            'created_at': None,
            'expires_at': None,
            'last_used': None,
            'active': 1,
            'created_by': None
        }

        key_record = KeyRecord.from_db_row(mock_row)

        assert key_record.id == 'test_key_id'
        assert key_record.description is None
        assert key_record.created_at is None
        assert key_record.expires_at is None
        assert key_record.last_used is None


class TestKeyManagerInitialization:
    """Test KeyManager initialization."""

    def test_init_creates_database(self, temp_dir):
        """Test that KeyManager creates database on initialization."""
        db_path = temp_dir / "test_keys.db"

        with patch('localvectordb_server.keymanager.bcrypt'):
            manager = KeyManager(str(db_path))

        assert db_path.exists()

        # Verify database structure
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row['name'] for row in cursor.fetchall()}

        expected_tables = {'api_keys', 'schema_version'}
        assert expected_tables.issubset(tables)

    def test_init_creates_parent_directories(self, temp_dir):
        """Test that KeyManager creates parent directories."""
        nested_path = temp_dir / "nested" / "path" / "keys.db"

        with patch('localvectordb_server.keymanager.bcrypt'):
            manager = KeyManager(str(nested_path))

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_init_sets_schema_version(self, temp_dir):
        """Test that KeyManager sets the schema version."""
        db_path = temp_dir / "test_keys.db"

        with patch('localvectordb_server.keymanager.bcrypt'):
            manager = KeyManager(str(db_path))

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT version FROM schema_version")
            version = cursor.fetchone()[0]

        assert version == KeyManager.SCHEMA_VERSION


class TestKeyManagerCreation:
    """Test KeyManager key creation functionality."""

    def test_create_key_basic(self, mock_key_manager):
        """Test basic key creation."""
        manager, mock_bcrypt = mock_key_manager

        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

            key_record = manager.create_key(description="Test Key")

        assert key_record.description == "Test Key"
        assert key_record.active is True
        assert key_record.plain_key is not None
        assert key_record.plain_key.startswith("lvdb_")
        assert len(key_record.plain_key) == len("lvdb_") + KeyManager.KEY_LENGTH

    def test_create_key_with_expiration(self, mock_key_manager):
        """Test key creation with expiration."""
        manager, mock_bcrypt = mock_key_manager

        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = now

            key_record = manager.create_key(
                description="Expiring Key",
                expires_days=30,
                created_by="test_user"
            )

        assert key_record.expires_at is not None
        expected_expiry = now + timedelta(days=30)
        assert key_record.expires_at == expected_expiry
        assert key_record.created_by == "test_user"

    def test_create_key_generates_unique_ids(self, mock_key_manager):
        """Test that multiple keys get unique IDs."""
        manager, mock_bcrypt = mock_key_manager

        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

            key1 = manager.create_key(description="Key 1")
            key2 = manager.create_key(description="Key 2")

        assert key1.id != key2.id
        assert key1.plain_key != key2.plain_key

    def test_create_key_hashes_password(self, mock_key_manager):
        """Test that key creation hashes the password."""
        manager, mock_bcrypt = mock_key_manager

        key_record = manager.create_key(description="Test Key")

        # Verify bcrypt was called
        mock_bcrypt.gensalt.assert_called_once()
        mock_bcrypt.hashpw.assert_called_once()

        # Verify key was stored in database
        retrieved_key = manager.get_key(key_record.id)
        assert retrieved_key is not None
        assert retrieved_key.description == "Test Key"


class TestKeyManagerValidation:
    """Test KeyManager key validation functionality."""

    def test_validate_key_success(self, mock_key_manager):
        """Test successful key validation."""
        manager, mock_bcrypt = mock_key_manager

        # Create a key first
        key_record = manager.create_key(description="Test Key")
        plain_key = key_record.plain_key

        # Mock bcrypt to return True for validation
        mock_bcrypt.checkpw.return_value = True

        # Validate the key
        is_valid = manager.validate_key(plain_key)

        assert is_valid is True
        mock_bcrypt.checkpw.assert_called()

    def test_validate_key_invalid_prefix(self, mock_key_manager):
        """Test validation fails for key with wrong prefix."""
        manager, mock_bcrypt = mock_key_manager

        is_valid = manager.validate_key("wrong_prefix_key123")

        assert is_valid is False

    def test_validate_key_empty_string(self, mock_key_manager):
        """Test validation fails for empty string."""
        manager, mock_bcrypt = mock_key_manager

        is_valid = manager.validate_key("")

        assert is_valid is False

    def test_validate_key_none(self, mock_key_manager):
        """Test validation fails for None."""
        manager, mock_bcrypt = mock_key_manager

        is_valid = manager.validate_key(None)

        assert is_valid is False

    def test_validate_key_wrong_key(self, mock_key_manager):
        """Test validation fails for wrong key."""
        manager, mock_bcrypt = mock_key_manager

        # Create a key
        manager.create_key(description="Test Key")

        # Mock bcrypt to return False for wrong key
        mock_bcrypt.checkpw.return_value = False

        is_valid = manager.validate_key("lvdb_wrongkey123")

        assert is_valid is False

    def test_validate_key_expired(self, mock_key_manager):
        """Test validation fails for expired key."""
        manager, mock_bcrypt = mock_key_manager

        # Create a key that expires in the past
        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            past_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = past_time

            key_record = manager.create_key(
                description="Expired Key",
                expires_days=1
            )
            plain_key = key_record.plain_key

            # Mock bcrypt to return True (key matches)
            mock_bcrypt.checkpw.return_value = True

        is_valid = manager.validate_key(plain_key)

        assert is_valid is False

    def test_validate_key_updates_last_used(self, mock_key_manager):
        """Test that validation updates last_used timestamp."""
        manager, mock_bcrypt = mock_key_manager

        # Create a key
        key_record = manager.create_key(description="Test Key")
        plain_key = key_record.plain_key

        # Mock bcrypt and datetime
        mock_bcrypt.checkpw.return_value = True

        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            validation_time = datetime(2024, 6, 1, 15, 30, 0, tzinfo=UTC)
            mock_datetime.now.return_value = validation_time

            manager.validate_key(plain_key)

        # Check that last_used was updated
        updated_key = manager.get_key(key_record.id)
        assert updated_key.last_used is not None

    def test_validate_key_no_update_last_used(self, mock_key_manager):
        """Test validation without updating last_used when flag is False."""
        manager, mock_bcrypt = mock_key_manager

        # Create a key
        key_record = manager.create_key(description="Test Key")
        plain_key = key_record.plain_key
        original_last_used = key_record.last_used

        mock_bcrypt.checkpw.return_value = True

        # Validate without updating last_used
        manager.validate_key(plain_key, update_last_used=False)

        # Check that last_used was not updated
        updated_key = manager.get_key(key_record.id)
        assert updated_key.last_used == original_last_used


class TestKeyManagerRetrieval:
    """Test KeyManager key retrieval functionality."""

    def test_get_key_exists(self, mock_key_manager):
        """Test getting an existing key."""
        manager, mock_bcrypt = mock_key_manager

        # Create a key
        key_record = manager.create_key(description="Test Key")

        # Retrieve the key
        retrieved = manager.get_key(key_record.id)

        assert retrieved is not None
        assert retrieved.id == key_record.id
        assert retrieved.description == "Test Key"

    def test_get_key_not_exists(self, mock_key_manager):
        """Test getting a non-existent key."""
        manager, mock_bcrypt = mock_key_manager

        retrieved = manager.get_key("nonexistent_key")

        assert retrieved is None

    def test_list_keys_empty(self, mock_key_manager):
        """Test listing keys when none exist."""
        manager, mock_bcrypt = mock_key_manager

        keys = manager.list_keys()

        assert keys == []

    def test_list_keys_multiple(self, mock_key_manager):
        """Test listing multiple keys."""
        manager, mock_bcrypt = mock_key_manager

        # Create multiple keys
        key1 = manager.create_key(description="Key 1")
        key2 = manager.create_key(description="Key 2")

        keys = manager.list_keys()

        assert len(keys) == 2
        key_ids = {key.id for key in keys}
        assert key1.id in key_ids
        assert key2.id in key_ids

    def test_list_keys_active_only(self, mock_key_manager):
        """Test listing only active keys."""
        manager, mock_bcrypt = mock_key_manager

        # Create keys
        active_key = manager.create_key(description="Active Key")
        revoked_key = manager.create_key(description="Revoked Key")

        # Revoke one key
        manager.revoke_key(revoked_key.id)

        # List only active keys
        active_keys = manager.list_keys(active_only=True)

        assert len(active_keys) == 1
        assert active_keys[0].id == active_key.id

    def test_list_keys_exclude_expired(self, mock_key_manager):
        """Test listing keys excluding expired ones."""
        manager, mock_bcrypt = mock_key_manager

        # Create a key that will be expired
        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            past_time = datetime(2023, 1, 1, tzinfo=UTC)
            mock_datetime.now.return_value = past_time

            expired_key = manager.create_key(
                description="Expired Key",
                expires_days=1
            )

            # Create a non-expired key
            valid_key = manager.create_key(description="Valid Key")

            # Fast forward time
            future_time = datetime(2024, 1, 1, tzinfo=UTC)
            mock_datetime.now.return_value = future_time

            # List keys excluding expired
            keys = manager.list_keys(include_expired=False)

        # Should only get the valid key
        assert len(keys) == 1
        assert keys[0].id == valid_key.id


class TestKeyManagerRevocation:
    """Test KeyManager key revocation functionality."""

    def test_revoke_key_success(self, mock_key_manager):
        """Test successful key revocation."""
        manager, mock_bcrypt = mock_key_manager

        # Create a key
        key_record = manager.create_key(description="Test Key")

        # Revoke the key
        success = manager.revoke_key(key_record.id)

        assert success is True

        # Verify key is revoked
        retrieved = manager.get_key(key_record.id)
        assert retrieved.active is False

    def test_revoke_key_not_exists(self, mock_key_manager):
        """Test revoking a non-existent key."""
        manager, mock_bcrypt = mock_key_manager

        success = manager.revoke_key("nonexistent_key")

        assert success is False

    def test_rotate_key_success(self, mock_key_manager):
        """Test successful key rotation."""
        manager, mock_bcrypt = mock_key_manager

        # Create original key
        original_key = manager.create_key(
            description="Original Key",
            expires_days=30,
            created_by="test_user"
        )
        original_id = original_key.id

        # Rotate the key
        new_key = manager.rotate_key(original_id)

        assert new_key is not None
        assert new_key.id != original_id
        assert "Rotated from" in new_key.description
        assert new_key.created_by == "test_user"

        # Verify original key is revoked
        original_retrieved = manager.get_key(original_id)
        assert original_retrieved.active is False

    def test_rotate_key_not_exists(self, mock_key_manager):
        """Test rotating a non-existent key."""
        manager, mock_bcrypt = mock_key_manager

        result = manager.rotate_key("nonexistent_key")

        assert result is None


class TestKeyManagerExpiration:
    """Test KeyManager expiration and cleanup functionality."""

    def test_prune_expired_soft_delete(self, mock_key_manager):
        """Test pruning expired keys with soft delete."""
        manager, mock_bcrypt = mock_key_manager

        # Create expired and valid keys
        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            past_time = datetime(2023, 1, 1, tzinfo=UTC)
            mock_datetime.now.return_value = past_time

            expired_key = manager.create_key(
                description="Expired Key",
                expires_days=1
            )
            valid_key = manager.create_key(description="Valid Key")

            # Fast forward time
            future_time = datetime(2024, 1, 1, tzinfo=UTC)
            mock_datetime.now.return_value = future_time

            # Prune expired keys
            count = manager.prune_expired(soft_delete=True)

        assert count == 1

        # Verify expired key is deactivated, not deleted
        retrieved = manager.get_key(expired_key.id)
        assert retrieved is not None
        assert retrieved.active is False

        # Verify valid key is still active
        valid_retrieved = manager.get_key(valid_key.id)
        assert valid_retrieved.active is True

    def test_prune_expired_hard_delete(self, mock_key_manager):
        """Test pruning expired keys with hard delete."""
        manager, mock_bcrypt = mock_key_manager

        # Create expired and valid keys
        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            past_time = datetime(2023, 1, 1, tzinfo=UTC)
            mock_datetime.now.return_value = past_time

            expired_key = manager.create_key(
                description="Expired Key",
                expires_days=1
            )
            valid_key = manager.create_key(description="Valid Key")

            # Fast forward time
            future_time = datetime(2024, 1, 1, tzinfo=UTC)
            mock_datetime.now.return_value = future_time

            # Prune expired keys with hard delete
            count = manager.prune_expired(soft_delete=False)

        assert count == 1

        # Verify expired key is deleted
        retrieved = manager.get_key(expired_key.id)
        assert retrieved is None

        # Verify valid key still exists
        valid_retrieved = manager.get_key(valid_key.id)
        assert valid_retrieved is not None

    def test_prune_expired_no_expired_keys(self, mock_key_manager):
        """Test pruning when no keys are expired."""
        manager, mock_bcrypt = mock_key_manager

        # Create valid key
        manager.create_key(description="Valid Key")

        count = manager.prune_expired()

        assert count == 0


class TestKeyManagerStats:
    """Test KeyManager statistics functionality."""

    def test_get_stats_empty(self, mock_key_manager):
        """Test getting stats when no keys exist."""
        manager, mock_bcrypt = mock_key_manager

        stats = manager.get_stats()

        assert stats['total_keys'] == 0
        assert stats['active_keys'] == 0
        assert stats['expired_keys'] == 0
        assert stats['expiring_soon'] == 0
        assert stats['recently_used'] == 0

    def test_get_stats_with_keys(self, mock_key_manager):
        """Test getting stats with various key types."""
        manager, mock_bcrypt = mock_key_manager

        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            now = datetime(2024, 6, 1, tzinfo=UTC)
            mock_datetime.now.return_value = now

            # Create different types of keys
            active_key = manager.create_key(description="Active Key")

            # Expired key
            expired_key = manager.create_key(
                description="Expired Key",
                expires_days=-10  # Already expired
            )

            # Expiring soon key
            expiring_key = manager.create_key(
                description="Expiring Soon",
                expires_days=3  # Expires in 3 days
            )

            # Recently used key
            recent_key = manager.create_key(description="Recent Key")

            # Update last_used for recent key
            with manager._get_connection() as conn:
                recent_time = now - timedelta(hours=12)
                conn.execute(
                    "UPDATE api_keys SET last_used = ? WHERE id = ?",
                    (recent_time.isoformat(), recent_key.id)
                )
                conn.commit()

            # Revoke one key
            manager.revoke_key(active_key.id)

            stats = manager.get_stats()

        assert stats['total_keys'] == 4
        assert stats['active_keys'] == 3  # expired, expiring, recent (active is revoked)
        assert stats['expired_keys'] == 1
        assert stats['expiring_soon'] == 1
        assert stats['recently_used'] == 1


class TestKeyManagerUtilities:
    """Test KeyManager utility functions."""

    def test_generate_key_id_format(self, mock_key_manager):
        """Test key ID generation format."""
        manager, mock_bcrypt = mock_key_manager

        with patch('localvectordb_server.keymanager.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 6, 15, tzinfo=UTC)

            key_id = manager._generate_key_id()

        assert key_id.startswith("key_20240615_")
        assert len(key_id) == len("key_20240615_") + 6  # 6 random chars

    def test_generate_api_key_format(self, mock_key_manager):
        """Test API key generation format."""
        manager, mock_bcrypt = mock_key_manager

        api_key = manager._generate_api_key()

        assert api_key.startswith("lvdb_")
        assert len(api_key) == len("lvdb_") + KeyManager.KEY_LENGTH

    def test_hash_and_verify_key(self, mock_key_manager):
        """Test key hashing and verification with real bcrypt."""
        manager, mock_bcrypt = mock_key_manager

        # Use real bcrypt for this test
        import bcrypt

        test_key = "test_key_12345"
        hashed = bcrypt.hashpw(test_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Test verification
        assert bcrypt.checkpw(test_key.encode('utf-8'), hashed.encode('utf-8'))
        assert not bcrypt.checkpw("wrong_key".encode('utf-8'), hashed.encode('utf-8'))


class TestGetKeyManagerFunction:
    """Test the get_key_manager utility function."""

    def test_get_key_manager_default_path(self, temp_dir):
        """Test get_key_manager with default path."""
        with patch('localvectordb_server.keymanager.KeyManager') as mock_km_class:
            mock_km_instance = Mock()
            mock_km_class.return_value = mock_km_instance

            result = get_key_manager()

            mock_km_class.assert_called_once_with("./.lvdb/api_keys.db")
            assert result == mock_km_instance

    def test_get_key_manager_custom_path(self, temp_dir):
        """Test get_key_manager with custom path."""
        custom_path = str(temp_dir / "custom_keys.db")

        with patch('localvectordb_server.keymanager.KeyManager') as mock_km_class:
            mock_km_instance = Mock()
            mock_km_class.return_value = mock_km_instance

            result = get_key_manager(custom_path)

            mock_km_class.assert_called_once_with(custom_path)
            assert result == mock_km_instance


class TestKeyManagerErrorHandling:
    """Test KeyManager error handling."""

    def test_validate_key_bcrypt_error(self, mock_key_manager):
        """Test validation handles bcrypt errors gracefully."""
        manager, mock_bcrypt = mock_key_manager

        # Create a key first
        key_record = manager.create_key(description="Test Key")

        # Mock bcrypt to raise an exception
        mock_bcrypt.checkpw.side_effect = Exception("Bcrypt error")

        # Should return False instead of raising
        is_valid = manager.validate_key(key_record.plain_key)

        assert is_valid is False

    def test_database_connection_error_handling(self, temp_dir):
        """Test handling of database connection errors."""
        # Create a file where the database should be (to cause permission error)
        db_path = temp_dir / "readonly_keys.db"
        db_path.touch()
        db_path.chmod(0o444)  # Read-only

        with patch('localvectordb_server.keymanager.bcrypt'):
            # This should raise an exception during database operations
            with pytest.raises(Exception):
                manager = KeyManager(str(db_path))
                # Try to create a key, which should fail
                manager.create_key(description="Test")

    def test_invalid_database_path(self):
        """Test initialization with invalid database path."""
        # Test with a path that can't be created (on most systems)
        invalid_path = "/root/cannot_create/keys.db"

        with patch('localvectordb_server.keymanager.bcrypt'):
            # Should raise an exception or handle gracefully
            try:
                KeyManager(invalid_path)
            except (PermissionError, OSError):
                # Expected on systems where /root is not writable
                pass