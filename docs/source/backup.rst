.. _backup:

==========================
Backup and Recovery System
==========================

LocalVectorDB provides a comprehensive backup and recovery system that ensures your vector databases are protected against data loss. The system supports full backups, incremental backups, and point-in-time recovery (PITR).

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The backup system is designed to:

- Create consistent snapshots of your databases (both SQLite metadata and FAISS indexes)
- Support incremental backups to minimize storage requirements
- Enable point-in-time recovery to any moment in your backup history
- Automatically manage backup retention and cleanup
- Verify backup integrity with checksums

Key Features
^^^^^^^^^^^^

- **Full Backups**: Complete database snapshots including all metadata and vectors
- **Incremental Backups**: Only backup changes since the last backup
- **Point-in-Time Recovery**: Restore to any specific moment in time
- **Compression**: Support for gzip and lzma compression
- **Integrity Verification**: SHA-256 checksums for all backup files
- **Automatic Cleanup**: Configurable retention policies

Quick Start
-----------

Creating a Backup
^^^^^^^^^^^^^^^^^

Create a full backup of a database:

.. code-block:: bash

    $ lvdb backup create mydatabase

Create an incremental backup:

.. code-block:: bash

    $ lvdb backup create mydatabase --type incremental

Create a backup with specific compression:

.. code-block:: bash

    $ lvdb backup create mydatabase --compression lzma --compression-level 9

Listing Backups
^^^^^^^^^^^^^^^

List all backups for a database:

.. code-block:: bash

    $ lvdb backup list --database mydatabase

List backups in JSON format:

.. code-block:: bash

    $ lvdb backup list --database mydatabase --json

Restoring from Backup
^^^^^^^^^^^^^^^^^^^^^

Restore a specific backup:

.. code-block:: bash

    $ lvdb backup restore backup-id-here

Restore to a different location:

.. code-block:: bash

    $ lvdb backup restore backup-id-here --to-location ./restored-db

Point-in-Time Recovery
^^^^^^^^^^^^^^^^^^^^^^^

Restore to a specific point in time:

.. code-block:: bash

    $ lvdb backup pitr "2024-01-15 14:30:00"

Restore to a timestamp with custom location:

.. code-block:: bash

    $ lvdb backup pitr "2024-01-15 14:30:00" --to-location ./pitr-restored

Backup Types
------------

Full Backups
^^^^^^^^^^^^

A full backup creates a complete snapshot of the database, including:

- All SQLite tables and metadata
- Complete FAISS index
- Database configuration
- Schema version information

Full backups are self-contained and can be restored independently.

**Example:**

.. code-block:: python

    from localvectordb.backup import BackupManager
    
    backup_manager = BackupManager(
        db_path="./my_database.db",
        backup_dir="./backups"
    )
    
    # Create a full backup
    backup_id = backup_manager.create_backup(BackupType.FULL)
    print(f"Created backup: {backup_id}")

Incremental Backups
^^^^^^^^^^^^^^^^^^^

Incremental backups only store changes since the last backup:

- Uses SQLite WAL (Write-Ahead Logging) to track changes
- Significantly smaller than full backups
- Requires the backup chain for restoration

**Example:**

.. code-block:: python

    # Create an incremental backup
    backup_id = backup_manager.create_backup(BackupType.INCREMENTAL)
    
    # List backup chain
    chain = backup_manager.get_backup_chain(backup_id)
    for backup in chain:
        print(f"{backup['type']}: {backup['timestamp']}")

Configuration
-------------

Backup settings can be configured in your LocalVectorDB configuration file.

TOML Configuration
^^^^^^^^^^^^^^^^^^

.. code-block:: toml

    [backup]
    enabled = true
    default_location = "./backups"
    retention_days = 30
    max_backups = 50
    compression_type = "gzip"
    compression_level = 6
    
    # Auto-backup settings
    auto_backup_enabled = true
    auto_backup_interval_hours = 24
    auto_backup_type = "incremental"
    
    # Performance settings
    backup_chunk_size = 1048576  # 1MB
    verify_backups = true

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

You can also configure backup settings using environment variables:

.. code-block:: bash

    export LVDB_BACKUP_ENABLED=true
    export LVDB_BACKUP_DEFAULT_LOCATION=/var/backups/lvdb
    export LVDB_BACKUP_RETENTION_DAYS=60
    export LVDB_BACKUP_COMPRESSION_TYPE=lzma
    export LVDB_BACKUP_AUTO_BACKUP_ENABLED=true

Configuration Options
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Backup Configuration Options
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``enabled``
     - ``true``
     - Enable/disable backup functionality
   * - ``default_location``
     - ``./backups``
     - Default directory for storing backups
   * - ``retention_days``
     - ``30``
     - Number of days to retain backups
   * - ``max_backups``
     - ``50``
     - Maximum number of backups per database
   * - ``compression_type``
     - ``gzip``
     - Compression algorithm (gzip, lzma, none)
   * - ``compression_level``
     - ``6``
     - Compression level (1-9)
   * - ``auto_backup_enabled``
     - ``false``
     - Enable automatic scheduled backups
   * - ``auto_backup_interval_hours``
     - ``24``
     - Hours between automatic backups
   * - ``auto_backup_type``
     - ``incremental``
     - Type of automatic backup (full, incremental)
   * - ``backup_chunk_size``
     - ``1048576``
     - Chunk size for streaming backups (bytes)
   * - ``verify_backups``
     - ``true``
     - Verify backup integrity after creation

Point-in-Time Recovery (PITR)
------------------------------

PITR allows you to restore your database to any specific moment in time within your backup history.

How PITR Works
^^^^^^^^^^^^^^

1. **Backup Chain**: PITR uses a chain of full and incremental backups
2. **WAL Replay**: Applies Write-Ahead Log entries up to the target time
3. **Consistent State**: Ensures database consistency at the recovery point

Using PITR
^^^^^^^^^^

**Command Line:**

.. code-block:: bash

    # Restore to a specific timestamp
    $ lvdb backup pitr "2024-01-15 14:30:00" --database mydatabase
    
    # Restore to a relative time
    $ lvdb backup pitr "1 hour ago" --database mydatabase
    
    # Restore with verification
    $ lvdb backup pitr "2024-01-15 14:30:00" --verify

**Python API:**

.. code-block:: python

    from localvectordb.backup import PointInTimeRecoveryManager
    from datetime import datetime, timedelta
    
    pitr_manager = PointInTimeRecoveryManager(
        backup_dir="./backups",
        db_name="mydatabase"
    )
    
    # Restore to specific time
    target_time = datetime.now() - timedelta(hours=2)
    restored_path = pitr_manager.restore_to_point_in_time(
        target_time=target_time,
        restore_path="./restored_db"
    )
    
    print(f"Database restored to: {restored_path}")

PITR Limitations
^^^^^^^^^^^^^^^^

- Requires a complete backup chain from a full backup to the target time
- Cannot recover beyond the latest backup
- WAL mode must be enabled for incremental backups

Backup Management
-----------------

Verifying Backups
^^^^^^^^^^^^^^^^^

Verify the integrity of a backup:

.. code-block:: bash

    $ lvdb backup verify backup-id-here

Verify all backups for a database:

.. code-block:: bash

    $ lvdb backup verify --database mydatabase --all

**Python API:**

.. code-block:: python

    # Verify a specific backup
    is_valid = backup_manager.verify_backup(backup_id)
    
    # Verify with detailed report
    report = backup_manager.verify_backup(backup_id, detailed=True)
    print(f"Checksum valid: {report['checksum_valid']}")
    print(f"Files intact: {report['files_intact']}")

Cleanup and Retention
^^^^^^^^^^^^^^^^^^^^^

Clean up old backups based on retention policy:

.. code-block:: bash

    $ lvdb backup cleanup --database mydatabase

Force cleanup of backups older than 7 days:

.. code-block:: bash

    $ lvdb backup cleanup --database mydatabase --older-than 7

Delete specific backup:

.. code-block:: bash

    $ lvdb backup delete backup-id-here

**Python API:**

.. code-block:: python

    # Clean up based on retention policy
    deleted = backup_manager.cleanup_old_backups(
        retention_days=30,
        keep_min_backups=5
    )
    print(f"Deleted {len(deleted)} old backups")
    
    # Delete specific backup
    backup_manager.delete_backup(backup_id)

Backup Format
-------------

LocalVectorDB uses a custom ``.lvdb-backup`` format that combines all database components into a single compressed archive.

Archive Structure
^^^^^^^^^^^^^^^^^

.. code-block:: none

    backup.lvdb-backup
    ├── manifest.json        # Backup metadata
    ├── database.db         # SQLite database
    ├── index.faiss        # FAISS index
    ├── checksum.sha256    # Integrity checksum
    └── wal/               # WAL files (incremental only)
        ├── wal-00001
        └── wal-00002

Manifest Format
^^^^^^^^^^^^^^^

.. code-block:: json

    {
        "backup_id": "20240115_143000_full_abc123",
        "database_name": "mydatabase",
        "backup_type": "full",
        "timestamp": "2024-01-15T14:30:00",
        "version": "1.0.0",
        "compression": {
            "type": "gzip",
            "level": 6
        },
        "database": {
            "version": "1.2.0",
            "document_count": 1000,
            "chunk_count": 5000,
            "index_size": 104857600
        },
        "chain": {
            "parent_backup": null,
            "chain_length": 1
        }
    }

Best Practices
--------------

Backup Strategy
^^^^^^^^^^^^^^^

1. **Regular Full Backups**: Schedule weekly full backups
2. **Daily Incrementals**: Use incremental backups for daily protection
3. **Retention Policy**: Keep at least 30 days of backups
4. **Off-site Storage**: Store backups in a different location
5. **Test Restores**: Regularly test your restore process

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

- Use compression for large databases to save storage
- Schedule backups during low-traffic periods
- Use incremental backups to minimize backup time
- Consider parallel backups for multiple databases

Security Considerations
^^^^^^^^^^^^^^^^^^^^^^^

- Encrypt sensitive backups before storage
- Restrict access to backup directories
- Audit backup and restore operations
- Validate checksums before restoration

Monitoring
^^^^^^^^^^

Monitor your backup system:

- Set up alerts for backup failures
- Track backup sizes and growth
- Monitor restoration times
- Log all backup operations

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Backup Fails with "Database is locked"**

The database is being actively written to. Solutions:

- Use WAL mode for concurrent access
- Schedule backups during maintenance windows
- Implement retry logic with exponential backoff

**Incremental Backup Chain Broken**

Missing backup in the chain. Solutions:

- Create a new full backup
- Use PITR to the last valid point
- Verify all backups regularly

**Restoration Fails with Version Mismatch**

Database version incompatibility. Solutions:

- Check migration requirements
- Use the migration system to upgrade
- Restore to a compatible version

Error Messages
^^^^^^^^^^^^^^

.. list-table:: Common Error Messages
   :header-rows: 1
   :widths: 30 70

   * - Error
     - Solution
   * - ``BackupNotFoundError``
     - Verify backup ID exists using ``lvdb backup list``
   * - ``IncrementalChainError``
     - Create a new full backup to start a fresh chain
   * - ``ChecksumMismatchError``
     - Backup may be corrupted; use a different backup
   * - ``InsufficientSpaceError``
     - Free up disk space or change backup location
   * - ``VersionMismatchError``
     - Use migration system before restoration

API Reference
-------------

BackupManager
^^^^^^^^^^^^^

.. code-block:: python

    class BackupManager:
        """Manages database backups."""
        
        def __init__(self, db_path: str, backup_dir: str = "./backups"):
            """Initialize backup manager."""
        
        def create_backup(self, backup_type: BackupType = BackupType.FULL,
                         compression: str = "gzip",
                         compression_level: int = 6) -> str:
            """Create a backup and return backup ID."""
        
        def restore_backup(self, backup_id: str, 
                          restore_path: Optional[str] = None) -> str:
            """Restore from backup."""
        
        def list_backups(self, database_name: Optional[str] = None) -> List[Dict]:
            """List available backups."""
        
        def verify_backup(self, backup_id: str) -> bool:
            """Verify backup integrity."""
        
        def cleanup_old_backups(self, retention_days: int = 30) -> List[str]:
            """Clean up old backups."""

PointInTimeRecoveryManager
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class PointInTimeRecoveryManager:
        """Manages point-in-time recovery."""
        
        def __init__(self, backup_dir: str, db_name: str):
            """Initialize PITR manager."""
        
        def restore_to_point_in_time(self, target_time: datetime,
                                    restore_path: str) -> str:
            """Restore database to specific point in time."""
        
        def get_available_recovery_points(self) -> List[datetime]:
            """Get list of available recovery points."""
        
        def validate_recovery_point(self, target_time: datetime) -> bool:
            """Check if recovery to target time is possible."""

See Also
--------

- :ref:`migrations` - Database migration system
- :ref:`configuration` - Configuration options
- :ref:`cli` - Command-line interface
- :ref:`api` - Python API reference