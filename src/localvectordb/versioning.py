# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/versioning.py

"""Database versioning and migration tracking system for LocalVectorDB.

This module provides comprehensive database versioning capabilities using SQLite's
built-in PRAGMA user_version along with additional metadata tracking for migrations
and schema evolution.

Classes:
    DatabaseVersion: Manages version comparison and representation
    VersionManager: Handles version tracking and database migrations
"""

import logging
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Current LocalVectorDB schema version
CURRENT_SCHEMA_VERSION = "1.0.0"


class DatabaseVersion:
    """
    Represents a database version with semantic versioning support.

    Supports semantic versioning format (MAJOR.MINOR.PATCH) and provides
    comparison operations and conversion to/from SQLite's integer user_version.

    Parameters
    ----------
    version : str
        Version string in semantic versioning format (e.g., "1.0.0")

    Examples
    --------
    Basic usage::

        version = DatabaseVersion("1.2.3")
        print(version.major)  # 1
        print(version.minor)  # 2
        print(version.patch)  # 3
        print(version.to_sqlite_version())  # 1002003

    Version comparison::

        v1 = DatabaseVersion("1.0.0")
        v2 = DatabaseVersion("1.1.0")
        print(v1 < v2)  # True
        print(v2.is_compatible_with(v1))  # True (same major)
    """

    def __init__(self, version: str):
        self.original = version
        self._parse_version(version)

    def _parse_version(self, version: str) -> None:
        """Parse version string into components"""
        # Support semantic versioning: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
        pattern = (
            r"^(\d+)\.(\d+)\.(\d+)"
            r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
            r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
        )
        match = re.match(pattern, version)

        if not match:
            raise ValueError(f"Invalid version format: {version}. Expected semantic versioning (e.g., '1.0.0')")

        self.major = int(match.group(1))
        self.minor = int(match.group(2))
        self.patch = int(match.group(3))
        self.prerelease = match.group(4)
        self.build = match.group(5)

    def to_sqlite_version(self) -> int:
        """
        Convert version to SQLite PRAGMA user_version integer format.

        Uses format: MAJOR * 1,000,000 + MINOR * 1,000 + PATCH
        This allows for versions up to 999.999.999 which should be sufficient.

        Returns
        -------
        int
            Integer representation suitable for SQLite PRAGMA user_version
        """
        return self.major * 1_000_000 + self.minor * 1_000 + self.patch

    @classmethod
    def from_sqlite_version(cls, sqlite_version: int) -> "DatabaseVersion":
        """
        Create DatabaseVersion from SQLite PRAGMA user_version integer.

        Parameters
        ----------
        sqlite_version : int
            Integer from SQLite PRAGMA user_version

        Returns
        -------
        DatabaseVersion
            Parsed version object
        """
        if sqlite_version == 0:
            return cls("0.0.0")

        major = sqlite_version // 1_000_000
        minor = (sqlite_version % 1_000_000) // 1_000
        patch = sqlite_version % 1_000

        return cls(f"{major}.{minor}.{patch}")

    def is_compatible_with(self, other: "DatabaseVersion") -> bool:
        """
        Check if this version is compatible with another version.

        Versions are considered compatible if they have the same major version.
        This follows semantic versioning rules where major version changes
        indicate breaking changes.

        Parameters
        ----------
        other : DatabaseVersion
            Version to compare against

        Returns
        -------
        bool
            True if versions are compatible
        """
        return self.major == other.major

    def __str__(self) -> str:
        return self.original

    def __repr__(self) -> str:
        return f"DatabaseVersion('{self.original}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DatabaseVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, DatabaseVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, DatabaseVersion):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, DatabaseVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, DatabaseVersion):
            return NotImplemented
        return self == other or self > other


class VersionManager:
    """
    Manages database version tracking and migration metadata.

    Handles SQLite PRAGMA user_version, version metadata in the config table,
    and migration tracking through the migration_log table.

    Parameters
    ----------
    db_path : Union[str, Path]
        Path to the SQLite database file
    """

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)

    def get_database_version(self, conn: Optional[sqlite3.Connection] = None) -> DatabaseVersion:
        """
        Get the current database version.

        Reads from PRAGMA user_version and falls back to config table if needed.

        Parameters
        ----------
        conn : sqlite3.Connection, optional
            Database connection to use. If None, creates a new connection.

        Returns
        -------
        DatabaseVersion
            Current database version
        """
        should_close = conn is None
        if conn is None:
            conn = sqlite3.connect(self.db_path)

        try:
            # Try to get version from PRAGMA user_version first
            cursor = conn.execute("PRAGMA user_version")
            sqlite_version = cursor.fetchone()[0]

            if sqlite_version > 0:
                return DatabaseVersion.from_sqlite_version(sqlite_version)

            # Fall back to config table for legacy databases
            try:
                cursor = conn.execute("SELECT value FROM config WHERE key = ?", ("db_version",))
                row = cursor.fetchone()
                if row:
                    return DatabaseVersion(row[0])
            except sqlite3.OperationalError:
                # Config table doesn't exist or is inaccessible
                pass

            # Default version for unversioned databases
            return DatabaseVersion("0.0.0")

        finally:
            if should_close:
                conn.close()

    def set_database_version(self, version: DatabaseVersion, conn: Optional[sqlite3.Connection] = None) -> None:
        """
        Set the database version using both PRAGMA user_version and config table.

        Parameters
        ----------
        version : DatabaseVersion
            Version to set
        conn : sqlite3.Connection, optional
            Database connection to use. If None, creates a new connection.
        """
        should_close = conn is None
        if conn is None:
            conn = sqlite3.connect(self.db_path)

        try:
            with conn:
                # Set PRAGMA user_version (primary version tracking)
                conn.execute(f"PRAGMA user_version = {version.to_sqlite_version()}")

                # Also store in config table for additional metadata
                conn.execute(
                    """
                    INSERT OR REPLACE INTO config (key, value)
                    VALUES (?, ?)
                """,
                    ("db_version", str(version)),
                )

                # Store version metadata
                conn.execute(
                    """
                    INSERT OR REPLACE INTO config (key, value)
                    VALUES (?, ?)
                """,
                    ("version_updated_at", datetime.now(UTC).isoformat()),
                )

                logger.info(f"Database version updated to {version}")

        finally:
            if should_close:
                conn.close()

    def get_migration_history(self, conn: Optional[sqlite3.Connection] = None) -> List[Dict]:
        """
        Get the history of applied migrations.

        Parameters
        ----------
        conn : sqlite3.Connection, optional
            Database connection to use. If None, creates a new connection.

        Returns
        -------
        List[Dict]
            List of migration records with version, timestamp, and metadata
        """
        should_close = conn is None
        if conn is None:
            conn = sqlite3.connect(self.db_path)

        try:
            cursor = conn.execute("""
                SELECT version, applied_at, rollback_script, checksum
                FROM migration_log
                ORDER BY applied_at ASC
            """)

            migrations = []
            for row in cursor.fetchall():
                migrations.append(
                    {"version": row[0], "applied_at": row[1], "rollback_script": row[2], "checksum": row[3]}
                )

            return migrations

        except sqlite3.OperationalError:
            # migration_log table doesn't exist
            return []

        finally:
            if should_close:
                conn.close()

    def record_migration(
        self,
        version: str,
        rollback_script: Optional[str] = None,
        checksum: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """
        Record a completed migration in the migration log.

        Parameters
        ----------
        version : str
            Version that was migrated to
        rollback_script : str, optional
            SQL script for rolling back this migration
        checksum : str, optional
            Checksum of the migration for integrity verification
        conn : sqlite3.Connection, optional
            Database connection to use. If None, creates a new connection.
        """
        should_close = conn is None
        if conn is None:
            conn = sqlite3.connect(self.db_path)

        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO migration_log (version, applied_at, rollback_script, checksum)
                    VALUES (?, ?, ?, ?)
                """,
                    (version, datetime.now(UTC).isoformat(), rollback_script, checksum),
                )

                logger.info(f"Recorded migration to version {version}")

        finally:
            if should_close:
                conn.close()

    def needs_migration(
        self, target_version: Optional[DatabaseVersion] = None, conn: Optional[sqlite3.Connection] = None
    ) -> bool:
        """
        Check if database needs migration to target version.

        Parameters
        ----------
        target_version : DatabaseVersion, optional
            Target version to check against. Defaults to CURRENT_SCHEMA_VERSION.
        conn : sqlite3.Connection, optional
            Database connection to use. If None, creates a new connection.

        Returns
        -------
        bool
            True if migration is needed
        """
        if target_version is None:
            target_version = DatabaseVersion(CURRENT_SCHEMA_VERSION)

        current_version = self.get_database_version(conn)
        return current_version < target_version

    def initialize_version_tracking(self, conn: Optional[sqlite3.Connection] = None) -> None:
        """
        Initialize version tracking for a new database.

        Sets the database to the current schema version and records initial state.

        Parameters
        ----------
        conn : sqlite3.Connection, optional
            Database connection to use. If None, creates a new connection.
        """
        should_close = conn is None
        if conn is None:
            conn = sqlite3.connect(self.db_path)

        try:
            current_version = DatabaseVersion(CURRENT_SCHEMA_VERSION)
            self.set_database_version(current_version, conn)

            # Record initial state
            self.record_migration(
                str(current_version), rollback_script=None, checksum=None, conn=conn  # No rollback for initial version
            )

            logger.info(f"Initialized version tracking at {current_version}")

        finally:
            if should_close:
                conn.close()
