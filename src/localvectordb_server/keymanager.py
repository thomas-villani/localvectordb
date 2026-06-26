"""
API Key Management System for LocalVectorDB Server v1.0

This module provides secure API key management using SQLite storage with proper
hashing, expiration, and audit trails. Keys are stored separately from the main
configuration to improve security.

Features:
    - Secure key hashing with bcrypt
    - Key expiration and rotation
    - Audit trail and usage tracking
    - CLI integration for key management
    - Backward compatibility with config-based keys

Main Components:
    - KeyManager: Core key management functionality
    - KeyRecord: Data class for key metadata
    - CLI integration through auth commands

Security Features:
    - Keys are hashed with bcrypt before storage
    - Generated keys use cryptographically secure random data
    - Support for key expiration and automatic cleanup
    - Audit logging of key usage
    - Soft deletion for revoked keys

Examples:

    Creating and using the key manager::

        from localvectordb_server.key_manager import KeyManager

        # Initialize with database path
        key_mgr = KeyManager("/path/to/keys.db")

        # Create a new API key
        key_record = key_mgr.create_key(
            description="CI/CD Pipeline",
            expires_days=90
        )
        print(f"Generated key: {key_record.plain_key}")
        print(f"Key ID: {key_record.id}")

        # Validate a key
        is_valid = key_mgr.validate_key("lvdb_abcd1234...")

        # List active keys
        keys = key_mgr.list_keys(active_only=True)

        # Revoke a key
        key_mgr.revoke_key("key_12345")

    CLI usage::

        # Create a new key
        $ lvdb auth create-key --description "My App" --expires-days 30

        # List all keys
        $ lvdb auth list-keys

        # Revoke a key
        $ lvdb auth revoke-key key_12345

        # Clean up expired keys
        $ lvdb auth prune-expired
"""

import hashlib
import logging
import secrets
import sqlite3
import string
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from sqlite3 import Connection
from typing import Any, Dict, Generator, List, Optional

import bcrypt

from localvectordb.utils import parse_iso8601

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """API key permission levels"""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@dataclass
class KeyRecord:
    """Represents an API key record with metadata"""

    id: str
    key_hash: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    active: bool = True
    created_by: Optional[str] = None
    permission_level: PermissionLevel = PermissionLevel.READ_WRITE

    # This field is only populated during key creation
    plain_key: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if the key is expired"""
        if not self.expires_at:
            return False
        return datetime.now(UTC) > self.expires_at

    @property
    def days_until_expiry(self) -> Optional[int]:
        """Get days until expiry, None if no expiration"""
        if not self.expires_at:
            return None
        delta = self.expires_at - datetime.now(UTC)
        return max(0, delta.days)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "active": self.active,
            "created_by": self.created_by,
            "permission_level": self.permission_level.value,
            "is_expired": self.is_expired,
            "days_until_expiry": self.days_until_expiry,
        }

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "KeyRecord":
        """Create KeyRecord from database row"""
        # Handle permission_level with backward compatibility
        permission_level = PermissionLevel.READ_WRITE  # Default for backward compatibility
        if "permission_level" in row.keys() and row["permission_level"]:
            try:
                permission_level = PermissionLevel(row["permission_level"])
            except ValueError:
                # If invalid permission level in DB, default to read_write
                permission_level = PermissionLevel.READ_WRITE

        return cls(
            id=row["id"],
            key_hash=row["key_hash"],
            description=row["description"],
            created_at=parse_iso8601(row["created_at"]) if row["created_at"] else None,
            expires_at=parse_iso8601(row["expires_at"]) if row["expires_at"] else None,
            last_used=parse_iso8601(row["last_used"]) if row["last_used"] else None,
            active=bool(row["active"]),
            created_by=row["created_by"],
            permission_level=permission_level,
        )


class KeyManager:
    """
    Manages API keys with SQLite storage and bcrypt hashing

    This class handles the complete lifecycle of API keys including creation,
    validation, rotation, and cleanup. Keys are stored in a SQLite database
    with proper security measures.
    """

    # Database schema version for migrations
    SCHEMA_VERSION = 3

    # Key generation settings
    KEY_PREFIX = "lvdb_"
    KEY_LENGTH = 32  # Characters after prefix
    KEY_CHARSET = string.ascii_letters + string.digits

    def __init__(self, db_path: str):
        """
        Initialize KeyManager with database path

        Parameters
        ----------
        db_path : str
            Path to SQLite database file for storing keys
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the SQLite database with required schema and handle migrations"""
        with self._get_connection() as conn:
            # Create keys table with latest schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL,
                    key_fingerprint TEXT,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP,
                    last_used TIMESTAMP,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_by TEXT,
                    permission_level TEXT NOT NULL DEFAULT 'read_write'
                )
            """)

            # Create indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_active
                ON api_keys(active)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_expires
                ON api_keys(expires_at)
            """)

            # Create fingerprint index for fast lookup
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_fingerprint
                ON api_keys(key_fingerprint)
            """)

            # Only create permission index after ensuring column exists
            try:
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_keys_permission
                    ON api_keys(permission_level)
                """)
            except Exception as e:
                # Column might not exist yet if this is a migration
                logger.debug(f"Could not create permission index: {e}")
                pass

            # Create schema version table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            # Check current schema version and migrate if needed
            cursor = conn.execute("SELECT version FROM schema_version")
            version_row = cursor.fetchone()

            if not version_row:
                # New database - set current version
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (self.SCHEMA_VERSION,))
            else:
                current_version = version_row["version"]
                if current_version < self.SCHEMA_VERSION:
                    self._migrate_database(conn, current_version)
                    # Update schema version
                    conn.execute("UPDATE schema_version SET version = ?", (self.SCHEMA_VERSION,))

            conn.commit()

    def _migrate_database(self, conn: Connection, from_version: int) -> None:
        """Migrate database schema from one version to another"""
        logger.info(f"Migrating key database from version {from_version} to {self.SCHEMA_VERSION}")

        if from_version < 2:
            # Migration from version 1 to 2: Add permission_level column
            try:
                # Check if column already exists (in case of interrupted migration)
                cursor = conn.execute("PRAGMA table_info(api_keys)")
                columns = [row[1] for row in cursor.fetchall()]

                if "permission_level" not in columns:
                    logger.info("Adding permission_level column to api_keys table")
                    conn.execute("""
                        ALTER TABLE api_keys
                        ADD COLUMN permission_level TEXT NOT NULL DEFAULT 'read_write'
                    """)
                    # Create index on the new column
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_api_keys_permission
                        ON api_keys(permission_level)
                    """)
                    logger.info("Successfully added permission_level column and index")
                else:
                    logger.info("permission_level column already exists")

            except Exception as e:
                logger.error(f"Error during database migration: {e}")
                raise

        if from_version < 3:
            # Migration from version 2 to 3: Add key_fingerprint column for fast lookup
            try:
                cursor = conn.execute("PRAGMA table_info(api_keys)")
                columns = [row[1] for row in cursor.fetchall()]

                if "key_fingerprint" not in columns:
                    logger.info("Adding key_fingerprint column to api_keys table")
                    conn.execute("""
                        ALTER TABLE api_keys
                        ADD COLUMN key_fingerprint TEXT
                    """)
                    # Create index on the fingerprint column for fast lookup
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_api_keys_fingerprint
                        ON api_keys(key_fingerprint)
                    """)

                    # Generate fingerprints for existing keys - we cannot retroactively
                    # generate them from hashes, so existing keys will have NULL fingerprints
                    # and will fall back to the slower validation path
                    logger.info("Successfully added key_fingerprint column and index")
                    logger.info("Note: Existing keys will use slower validation until rotated")
                else:
                    logger.info("key_fingerprint column already exists")

            except Exception as e:
                logger.error(f"Error during database migration to v3: {e}")
                raise

    @contextmanager
    def _get_connection(self) -> Generator[Connection, Any, None]:
        """Get database connection with proper settings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _generate_key_id() -> str:
        """Generate a unique key ID"""
        # Use timestamp + random suffix for readability
        timestamp = datetime.now(UTC).strftime("%Y%m%d")
        random_suffix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(6))
        return f"key_{timestamp}_{random_suffix}"

    def _generate_api_key(self) -> str:
        """Generate a cryptographically secure API key"""
        random_part = "".join(secrets.choice(self.KEY_CHARSET) for _ in range(self.KEY_LENGTH))
        return f"{self.KEY_PREFIX}{random_part}"

    @staticmethod
    def _generate_fingerprint(key: str) -> str:
        """Generate a SHA-256 fingerprint of the API key for fast lookup

        Parameters
        ----------
        key : str
            The API key to fingerprint

        Returns
        -------
        str
            SHA-256 hash of the key (hex digest)
        """
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _hash_key(self, key: str) -> str:
        """Hash an API key using bcrypt"""
        # Use bcrypt with a reasonable cost factor
        salt = bcrypt.gensalt(rounds=14)
        return str(bcrypt.hashpw(key.encode("utf-8"), salt).decode("utf-8"))

    def _verify_key(self, key: str, key_hash: str) -> bool:
        """Verify an API key against its hash"""
        try:
            return bool(bcrypt.checkpw(key.encode("utf-8"), key_hash.encode("utf-8")))
        except Exception as e:
            logger.warning(f"Error verifying key: {e}")
            return False

    def create_key(
        self,
        description: Optional[str] = None,
        expires_days: Optional[int] = None,
        created_by: Optional[str] = None,
        permission_level: PermissionLevel = PermissionLevel.READ_WRITE,
    ) -> KeyRecord:
        """
        Create a new API key

        Parameters
        ----------
        description : str, optional
            Human-readable description of the key's purpose
        expires_days : int, optional
            Number of days until key expires (None = never expires)
        created_by : str, optional
            Identifier of who created the key
        permission_level : PermissionLevel, optional
            Permission level for the key (defaults to READ_WRITE)

        Returns
        -------
        KeyRecord
            The created key record with the plain key included
        """
        # Generate key and metadata
        key_id = self._generate_key_id()
        plain_key = self._generate_api_key()
        key_hash = self._hash_key(plain_key)
        key_fingerprint = self._generate_fingerprint(plain_key)
        created_at = datetime.now(UTC)
        expires_at = None
        if expires_days is not None:
            expires_at = created_at + timedelta(days=expires_days)

        # Store in database
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (id, key_hash, key_fingerprint,
                description, created_at, expires_at, created_by, active, permission_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    key_id,
                    key_hash,
                    key_fingerprint,
                    description,
                    created_at.isoformat(),
                    expires_at.isoformat() if expires_at else None,
                    created_by,
                    True,
                    permission_level.value,
                ),
            )
            conn.commit()

        logger.info(f"Created API key {key_id} with description: {description}, permission: {permission_level.value}")

        # Return record with plain key (only time it's available)
        return KeyRecord(
            id=key_id,
            key_hash=key_hash,
            description=description,
            created_at=created_at,
            expires_at=expires_at,
            created_by=created_by,
            permission_level=permission_level,
            active=True,
            plain_key=plain_key,  # Only included during creation
        )

    def validate_key(self, key: str, update_last_used: bool = True, prune_expired: bool = False) -> bool:
        """
        Validate an API key

        Parameters
        ----------
        key : str
            The API key to validate
        update_last_used : bool, default True
            Whether to update the last_used timestamp

        Returns
        -------
        bool
            True if key is valid and active, False otherwise
        """
        if not key or not key.startswith(self.KEY_PREFIX):
            return False

        # Generate fingerprint for fast lookup
        key_fingerprint = self._generate_fingerprint(key)

        with self._get_connection() as conn:
            # First try fast lookup by fingerprint
            cursor = conn.execute(
                """
                SELECT id, key_hash, expires_at FROM api_keys
                WHERE active = TRUE AND key_fingerprint = ?
            """,
                (key_fingerprint,),
            )

            candidates = cursor.fetchall()

            # If no fingerprint matches, fall back to checking all active keys
            # This handles legacy keys without fingerprints
            if not candidates:
                cursor = conn.execute("""
                    SELECT id, key_hash, expires_at FROM api_keys
                    WHERE active = TRUE AND key_fingerprint IS NULL
                """)
                candidates = cursor.fetchall()

            for row in candidates:
                # Check if key matches using bcrypt
                if self._verify_key(key, row["key_hash"]):
                    # Check expiration
                    if row["expires_at"]:
                        expires_at = parse_iso8601(row["expires_at"])
                        if datetime.now(UTC) > expires_at:
                            logger.info(f"Key {row['id']} is expired")
                            if prune_expired:
                                logger.info(f"Pruning expired key: {row['id']}")
                                conn.execute("DELETE FROM api_keys WHERE id = ?", (row["id"],))
                                conn.commit()
                            return False

                    # Update last used timestamp
                    if update_last_used:
                        conn.execute(
                            """
                            UPDATE api_keys SET last_used = ? WHERE id = ?
                        """,
                            (datetime.now(UTC).isoformat(), row["id"]),
                        )
                        conn.commit()

                    logger.debug(f"Key {row['id']} validated successfully")
                    return True

        logger.debug("Key validation failed")
        return False

    def validate_key_with_permissions(
        self, key: str, update_last_used: bool = True, prune_expired: bool = False
    ) -> tuple[bool, Optional[PermissionLevel], Optional[str]]:
        """
        Validate an API key and return its permission level and key_id

        Parameters
        ----------
        key : str
            The API key to validate
        update_last_used : bool, default True
            Whether to update the last_used timestamp
        prune_expired : bool, default False
            Whether to automatically prune expired keys

        Returns
        -------
        tuple[bool, Optional[PermissionLevel], Optional[str]]
            (is_valid, permission_level, key_id) - permission_level and key_id are None if key is invalid
        """
        if not key or not key.startswith(self.KEY_PREFIX):
            return False, None, None

        # Generate fingerprint for fast lookup
        key_fingerprint = self._generate_fingerprint(key)

        with self._get_connection() as conn:
            # First try fast lookup by fingerprint
            cursor = conn.execute(
                """
                SELECT id, key_hash, expires_at, permission_level FROM api_keys
                WHERE active = TRUE AND key_fingerprint = ?
            """,
                (key_fingerprint,),
            )

            candidates = cursor.fetchall()

            # If no fingerprint matches, fall back to checking all active keys
            # This handles legacy keys without fingerprints
            if not candidates:
                cursor = conn.execute("""
                    SELECT id, key_hash, expires_at, permission_level FROM api_keys
                    WHERE active = TRUE AND key_fingerprint IS NULL
                """)
                candidates = cursor.fetchall()

            for row in candidates:
                # Check if key matches using bcrypt
                if self._verify_key(key, row["key_hash"]):
                    # Check expiration
                    if row["expires_at"]:
                        expires_at = parse_iso8601(row["expires_at"])
                        if datetime.now(UTC) > expires_at:
                            logger.info(f"Key {row['id']} is expired")
                            if prune_expired:
                                logger.info(f"Pruning expired key: {row['id']}")
                                conn.execute("DELETE FROM api_keys WHERE id = ?", (row["id"],))
                                conn.commit()
                            return False, None, None

                    # Update last used timestamp
                    if update_last_used:
                        conn.execute(
                            """
                            UPDATE api_keys SET last_used = ? WHERE id = ?
                        """,
                            (datetime.now(UTC).isoformat(), row["id"]),
                        )
                        conn.commit()

                    # Get permission level
                    try:
                        permission_level = PermissionLevel(row["permission_level"])
                    except (ValueError, KeyError):
                        # If invalid permission level in DB, default to read_write for safety
                        permission_level = PermissionLevel.READ_WRITE

                    logger.debug(f"Key {row['id']} validated successfully with permission: {permission_level.value}")
                    return True, permission_level, row["id"]

        logger.debug("Key validation failed")
        return False, None, None

    def get_key(self, key_id: str) -> Optional[KeyRecord]:
        """
        Get a key record by ID

        Parameters
        ----------
        key_id : str
            The key ID to retrieve

        Returns
        -------
        KeyRecord or None
            The key record if found, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM api_keys WHERE id = ?
            """,
                (key_id,),
            )

            row = cursor.fetchone()
            if row:
                return KeyRecord.from_db_row(row)
            return None

    def list_keys(self, active_only: bool = False, include_expired: bool = True) -> List[KeyRecord]:
        """
        List API keys

        Parameters
        ----------
        active_only : bool, default False
            Only return active (non-revoked) keys
        include_expired : bool, default True
            Include expired keys in results

        Returns
        -------
        List[KeyRecord]
            List of key records
        """
        conditions = []
        params = []

        if active_only:
            conditions.append("active = TRUE")

        if not include_expired:
            conditions.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(datetime.now(UTC).isoformat())

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM api_keys {where_clause}
                ORDER BY created_at DESC
            """,
                params,
            )

            return [KeyRecord.from_db_row(row) for row in cursor.fetchall()]

    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke (deactivate) an API key

        Parameters
        ----------
        key_id : str
            The key ID to revoke

        Returns
        -------
        bool
            True if key was revoked, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE api_keys SET active = FALSE WHERE id = ?
            """,
                (key_id,),
            )
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Revoked API key {key_id}")
                return True
            else:
                logger.warning(f"Key {key_id} not found for revocation")
                return False

    def rotate_key(self, key_id: str) -> Optional[KeyRecord]:
        """
        Rotate an API key (create new key, deactivate old one)

        Parameters
        ----------
        key_id : str
            The key ID to rotate

        Returns
        -------
        KeyRecord or None
            The new key record if successful, None if original key not found
        """
        # Get original key
        original_key = self.get_key(key_id)
        if not original_key:
            return None

        # Create new key with same metadata
        new_key = self.create_key(
            description=f"Rotated from {key_id}: {original_key.description}",
            expires_days=original_key.days_until_expiry,
            created_by=original_key.created_by,
        )

        # Revoke original key
        self.revoke_key(key_id)

        logger.info(f"Rotated key {key_id} to {new_key.id}")
        return new_key

    def prune_expired(self, soft_delete: bool = True) -> int:
        """
        Clean up expired keys

        Parameters
        ----------
        soft_delete : bool, default True
            If True, mark as inactive instead of deleting

        Returns
        -------
        int
            Number of keys pruned
        """
        now = datetime.now(UTC).isoformat()

        with self._get_connection() as conn:
            if soft_delete:
                cursor = conn.execute(
                    """
                    UPDATE api_keys SET active = FALSE
                    WHERE expires_at IS NOT NULL AND expires_at <= ? AND active = TRUE
                """,
                    (now,),
                )
            else:
                cursor = conn.execute(
                    """
                    DELETE FROM api_keys
                    WHERE expires_at IS NOT NULL AND expires_at <= ?
                """,
                    (now,),
                )

            conn.commit()
            count = int(cursor.rowcount)

        if count > 0:
            action = "deactivated" if soft_delete else "deleted"
            logger.info(f"Pruned {count} expired keys ({action})")

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get key management statistics

        Returns
        -------
        Dict[str, Any]
            Statistics about keys in the system
        """
        with self._get_connection() as conn:
            stats = {}

            # Total keys
            cursor = conn.execute("SELECT COUNT(*) as count FROM api_keys")
            stats["total_keys"] = cursor.fetchone()["count"]

            # Active keys
            cursor = conn.execute("SELECT COUNT(*) as count FROM api_keys WHERE active = TRUE")
            stats["active_keys"] = cursor.fetchone()["count"]

            # Expired keys
            now = datetime.now(UTC).isoformat()
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count FROM api_keys
                WHERE expires_at IS NOT NULL AND expires_at <= ?
            """,
                (now,),
            )
            stats["expired_keys"] = cursor.fetchone()["count"]

            # Keys expiring soon (next 7 days)
            future = (datetime.now(UTC) + timedelta(days=7)).isoformat()
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count FROM api_keys
                WHERE expires_at IS NOT NULL AND expires_at <= ? AND expires_at > ? AND active = TRUE
            """,
                (future, now),
            )
            stats["expiring_soon"] = cursor.fetchone()["count"]

            # Recently used keys (last 24 hours)
            recent = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count FROM api_keys
                WHERE last_used IS NOT NULL AND last_used >= ? AND active = TRUE
            """,
                (recent,),
            )
            stats["recently_used"] = cursor.fetchone()["count"]

            return stats


def get_key_manager(key_db_path: Optional[str] = None) -> KeyManager:
    """
    Get a KeyManager instance using configuration

    Parameters
    ----------
    key_db_path : str, optional
        Path to the key database, defaults to "./.lvdb/api_keys.db"

    Returns
    -------
    KeyManager
        Configured KeyManager instance
    """
    # Default key database location
    key_db_path = key_db_path or "./.lvdb/api_keys.db"
    return KeyManager(str(key_db_path))
