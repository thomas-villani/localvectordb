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
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock, patch

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
    # from localvectordb_server.keymanager import bcrypt as bcrypt_mod
    with patch('localvectordb_server.keymanager.bcrypt') as mock_bcrypt:
        # Mock bcrypt operations for consistent testing
        mock_bcrypt.gensalt.return_value = b'$2b$12$mock_salt'
        mock_bcrypt.hashpw.return_value = b'$2b$12$mock_hashed_password'
        mock_bcrypt.checkpw.return_value = True

        manager = KeyManager(str(db_path))
        yield manager, mock_bcrypt

@pytest.fixture
def real_key_manager(temp_dir):
    """Create a KeyManager that uses real bcrypt (no mocking)."""
    db_path = temp_dir / "test_keys.db"
    manager = KeyManager(str(db_path))
    yield manager


@pytest.mark.unit
class TestKeyRecord:
    """Test KeyRecord data class."""

    def test_key_record_creation(self, sample_key_record):
        """Test KeyRecord creation with all fields."""
        assert sample_key_record.id == "key_20240101_abc123"
        assert sample_key_record.description == "Test API Key"
        assert sample_key_record.active is True
        assert sample_key_record.created_by == "test_user"

    def test_key_record_is_expired_logic(self):
        """Test is_expired logic with known dates."""
        # Test with future expiry (not expired)
        future_key = KeyRecord(
            id="test",
            key_hash="hash",
            expires_at=datetime.now(UTC) + timedelta(days=30)
        )
        assert future_key.is_expired is False

        # Test with past expiry (expired)
        past_key = KeyRecord(
            id="test",
            key_hash="hash",
            expires_at=datetime.now(UTC) - timedelta(days=1)
        )
        assert past_key.is_expired is True

        # Test with no expiry
        no_expiry_key = KeyRecord(
            id="test",
            key_hash="hash",
            expires_at=None
        )
        assert no_expiry_key.is_expired is False

    def test_days_until_expiry(self):
        """Test is_expired logic with known dates."""
        # Test with future expiry (not expired)
        future_key = KeyRecord(
            id="test",
            key_hash="hash",
            expires_at=datetime.now(UTC) + timedelta(days=30)
        )
        assert future_key.days_until_expiry == 30


    def test_days_until_expiry_expired(self):
        """Test days_until_expiry for expired key."""
        future_key = KeyRecord(
            id="test",
            key_hash="hash",
            expires_at=datetime.now(UTC) - timedelta(days=30)
        )
        # Should be zero for expired keys
        assert future_key.days_until_expiry == 0

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


@pytest.mark.unit
@pytest.mark.database
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


@pytest.mark.unit
@pytest.mark.database
class TestKeyManagerWithRealBcrypt:
    """Test KeyManager with real bcrypt operations."""

    def test_create_and_validate_key_success(self, real_key_manager):
        """Test creating a key and then successfully validating it."""
        manager = real_key_manager

        # Create a key
        key_record = manager.create_key(description="Test Key")
        plain_key = key_record.plain_key

        # The plain key should be returned
        assert plain_key is not None
        assert plain_key.startswith("lvdb_")

        # Validate the same key should succeed
        is_valid = manager.validate_key(plain_key)
        assert is_valid is True

    def test_validate_wrong_key_fails(self, real_key_manager):
        """Test that validating a wrong key fails."""
        manager = real_key_manager

        # Create a key
        manager.create_key(description="Test Key")

        # Try to validate a different key
        wrong_key = "lvdb_wrongkeyhere123456789012345678"
        is_valid = manager.validate_key(wrong_key)
        assert is_valid is False

    def test_validate_key_updates_last_used(self, real_key_manager):
        """Test that successful validation updates last_used."""
        manager = real_key_manager

        # Create a key
        key_record = manager.create_key(description="Test Key")
        assert key_record.last_used is None
        # Validate the key
        manager.validate_key(key_record.plain_key, update_last_used=True)
        original_last_used = manager.get_key(key_record.id).last_used
        # print(original_last_used)
        # Wait a tiny bit to ensure timestamp difference
        import time
        time.sleep(0.01)

        # Check that last_used was updated
        manager.validate_key(key_record.plain_key, update_last_used=True)
        updated_key = manager.get_key(key_record.id)
        # print(updated_key.last_used)

        assert updated_key.last_used != original_last_used
        assert updated_key.last_used > original_last_used

    def test_validate_key_no_update_when_disabled(self, real_key_manager):
        """Test validation without updating last_used when flag is False."""
        manager = real_key_manager

        # Create a key
        key_record = manager.create_key(description="Test Key")
        original_last_used = key_record.last_used

        # Validate without updating last_used
        is_valid = manager.validate_key(key_record.plain_key, update_last_used=False)
        assert is_valid is True

        # Check that last_used was not updated
        updated_key = manager.get_key(key_record.id)
        assert updated_key.last_used == original_last_used

    def test_hash_verify_cycle(self, real_key_manager):
        """Test that the hash/verify cycle works correctly."""
        manager = real_key_manager

        # Generate a key
        test_key = manager._generate_api_key()

        # Hash it
        key_hash = manager._hash_key(test_key)

        # Verify it works
        assert manager._verify_key(test_key, key_hash) is True

        # Verify wrong key fails
        wrong_key = manager._generate_api_key()  # Different key
        assert manager._verify_key(wrong_key, key_hash) is False

    def test_multiple_keys_unique_hashes(self, real_key_manager):
        """Test that multiple keys get unique hashes."""
        manager = real_key_manager

        # Create multiple keys with same content but they should have different hashes
        key1 = manager.create_key(description="Same Description")
        key2 = manager.create_key(description="Same Description")

        # Keys should be different
        assert key1.plain_key != key2.plain_key
        assert key1.key_hash != key2.key_hash
        assert key1.id != key2.id

        # Both should validate correctly
        assert manager.validate_key(key1.plain_key) is True
        assert manager.validate_key(key2.plain_key) is True


@pytest.mark.unit
@pytest.mark.database
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
        expired_key = manager.create_key(
            description="Expired Key",
            expires_days=-10
        )
        # Create a non-expired key
        valid_key = manager.create_key(description="Valid Key")

        # List keys excluding expired
        keys = manager.list_keys(include_expired=False)

        # Should only get the valid key
        assert len(keys) == 1
        assert keys[0].id == valid_key.id


@pytest.mark.unit
@pytest.mark.database
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


@pytest.mark.unit
@pytest.mark.database
class TestKeyManagerExpiration:
    """Test KeyManager expiration and cleanup functionality."""

    def test_prune_expired_soft_delete(self, mock_key_manager):
        """Test pruning expired keys with soft delete."""
        manager, mock_bcrypt = mock_key_manager

        # Create expired and valid keys
        expired_key = manager.create_key(
            description="Expired Key",
            expires_days=-10
        )
        valid_key = manager.create_key(description="Valid Key")

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

        expired_key = manager.create_key(
            description="Expired Key",
            expires_days=-10
        )
        valid_key = manager.create_key(description="Valid Key")

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


@pytest.mark.unit
@pytest.mark.database
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
            recent_time = datetime.now(UTC) - timedelta(hours=12)
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


@pytest.mark.unit
@pytest.mark.database
class TestKeyManagerUtilities:
    """Test KeyManager utility functions."""

    def test_generate_key_id_format(self, mock_key_manager):
        """Test key ID generation format."""
        manager, mock_bcrypt = mock_key_manager

        key_id = manager._generate_key_id()
        expected_key_id = datetime.now().strftime("key_%Y%m%d_")

        assert key_id.startswith(expected_key_id)
        assert len(key_id) == len(expected_key_id) + 6  # 6 random chars

    def test_generate_api_key_format(self, mock_key_manager):
        """Test API key generation format."""
        manager, mock_bcrypt = mock_key_manager

        api_key = manager._generate_api_key()

        assert api_key.startswith("lvdb_")
        assert len(api_key) == len("lvdb_") + KeyManager.KEY_LENGTH

    def test_hash_and_verify_key(self, mock_key_manager):
        """Test key hashing and verification with real bcrypt."""
        # Use real bcrypt for this test
        import bcrypt

        test_key = "test_key_12345"
        hashed = bcrypt.hashpw(test_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Test verification
        assert bcrypt.checkpw(test_key.encode('utf-8'), hashed.encode('utf-8'))
        assert not bcrypt.checkpw("wrong_key".encode('utf-8'), hashed.encode('utf-8'))



@pytest.mark.unit
@pytest.mark.database
class TestKeyManagerErrorHandling:
    """Test KeyManager error handling."""


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