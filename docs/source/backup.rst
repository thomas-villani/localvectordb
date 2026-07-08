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

Create an incremental backup (requires the parent backup's ID — incremental
backups are stored as a delta against their parent):

.. code-block:: bash

    $ lvdb backup create mydatabase --type incremental --parent <parent-backup-id>

Create a backup with a specific compression algorithm:

.. code-block:: bash

    $ lvdb backup create mydatabase --compression lzma

Listing Backups
^^^^^^^^^^^^^^^

List all backups for a database:

.. code-block:: bash

    $ lvdb backup list --database mydatabase

List backups in JSON format:

.. code-block:: bash

    $ lvdb backup list --database mydatabase --format json

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

Restore to a specific point in time (``--to-location`` is required):

.. code-block:: bash

    $ lvdb backup pitr "2024-01-15 14:30:00" --to-location ./pitr-restored

Validate a recovery point without restoring, widening the search tolerance:

.. code-block:: bash

    $ lvdb backup pitr "2024-01-15 14:30:00" --to-location ./pitr-restored --tolerance 120 --dry-run

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

    from localvectordb.backup import (
        BackupManager,
        BackupConfig,
        BackupType,
        CompressionAlgorithm,
    )

    # Create backup configuration
    config = BackupConfig(
        backup_location="./backups",
        compression_algorithm=CompressionAlgorithm.GZIP,
    )

    backup_manager = BackupManager(
        database_path="./my_database.db",
        config=config
    )

    # Create a full backup
    backup_id = backup_manager.create_backup(BackupType.FULL)
    print(f"Created backup: {backup_id}")

Incremental Backups
^^^^^^^^^^^^^^^^^^^

Incremental backups only store changes since the last backup:

- Uses manifest-based diffing to compare document and chunk hashes
- Significantly smaller than full backups
- Requires the backup chain for restoration

**Example:**

.. code-block:: python

    # An incremental backup is taken against a parent backup. Create (or look up)
    # a full backup first, then pass its ID as ``parent_backup_id``.
    full_id = backup_manager.create_backup(BackupType.FULL)

    backup_id = backup_manager.create_backup(
        BackupType.INCREMENTAL,
        parent_backup_id=full_id,
    )
    print(f"Created incremental backup {backup_id} (parent {full_id})")

Restoring an Incremental Chain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BackupManager.restore_backup()`` restores a single archive. Restoring an
*incremental* backup, however, requires replaying the full chain (the root full
backup plus every incremental up to the target). That dedicated restore path
lives on :class:`IncrementalBackupManager`, which wraps a ``BackupManager``:

.. code-block:: python

    from localvectordb.backup import (
        BackupManager,
        BackupConfig,
        IncrementalBackupManager,
    )

    config = BackupConfig(backup_location="./backups")
    backup_manager = BackupManager("./my_database.db", config=config)

    # IncrementalBackupManager takes the BackupManager it should operate on.
    incremental_manager = IncrementalBackupManager(backup_manager)

    # restore_incremental_backup_chain() rebuilds the chain from the target
    # backup back to its root full backup, restores the full backup, then
    # applies each incremental in order. ``target_backup_id`` may be a full or
    # an incremental backup id. Returns the restore directory as a Path.
    restore_path = incremental_manager.restore_incremental_backup_chain(
        target_backup_id=backup_id,
        restore_location="./restored_db",
    )
    print(f"Restored incremental chain to: {restore_path}")

.. note::

   Incremental backups are *created* with
   ``BackupManager.create_backup(BackupType.INCREMENTAL, parent_backup_id=...)``
   (shown above); ``IncrementalBackupManager`` is only needed on the restore
   side, and internally by point-in-time recovery to replay chains.

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
2. **Manifest-Based Recovery**: Applies document and chunk changes from the backup chain up to the target time
3. **Consistent State**: Ensures database consistency at the recovery point

Using PITR
^^^^^^^^^^

**Command Line:**

.. code-block:: bash

    # Restore to a specific timestamp (timestamp must be
    # "YYYY-MM-DD HH:MM:SS" or ISO 8601; --to-location is required)
    $ lvdb backup pitr "2024-01-15 14:30:00" --to-location ./restored

    # Widen the search window and validate without restoring
    $ lvdb backup pitr "2024-01-15T14:30:00Z" --to-location ./restored --tolerance 120 --dry-run

**Python API:**

.. code-block:: python

    from localvectordb.backup import (
        BackupManager,
        BackupConfig,
        IncrementalBackupManager,
        PointInTimeRecoveryManager,
    )
    from datetime import datetime, timedelta

    # Create backup configuration
    config = BackupConfig(backup_location="./backups")

    # PITR is driven by PointInTimeRecoveryManager, not BackupManager.
    # Build the manager stack: a BackupManager, an IncrementalBackupManager
    # (needed to replay incremental chains), then the PITR manager itself.
    backup_manager = BackupManager(
        database_path="./mydatabase.db",
        config=config,
    )
    incremental_manager = IncrementalBackupManager(backup_manager)
    pitr_manager = PointInTimeRecoveryManager(backup_manager, incremental_manager)

    # Restore to a specific time using PITR. Returns a dict describing the
    # recovery operation (status, the recovery point used, and details).
    target_time = datetime.now() - timedelta(hours=2)
    result = pitr_manager.restore_to_point_in_time(
        target_time,
        restore_location="./restored_db",
        tolerance_minutes=60,
    )

    if result["success"]:
        print(f"Restored to {result['restore_location']}")
        print(f"Recovery point: {result['recovery_point']['backup_id']}")
    else:
        print(f"Recovery failed: {result['error']}")

Inspecting the Recovery Timeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PointInTimeRecoveryManager`` also exposes read-only helpers for inspecting the
available recovery points before committing to a restore. All of them operate on
the same manager instance constructed above.

.. code-block:: python

    # List every recovery point (full + incremental) in chronological order.
    # Each entry is a dict with timestamp, backup_id, backup_type,
    # parent_backup_id, database_version, and size_bytes.
    timeline = pitr_manager.get_recovery_timeline()
    for point in timeline:
        print(f"{point['timestamp']}  {point['backup_type']:<11}  {point['backup_id']}")

    # Find the best recovery point at or before a target time. Returns the
    # recovery-point dict, or None if nothing falls within tolerance_minutes.
    target_time = datetime.now() - timedelta(hours=2)
    point = pitr_manager.find_recovery_point(target_time, tolerance_minutes=60)
    if point is None:
        print("No recovery point within tolerance")

    # Validate the timeline for broken chains and gaps. Returns a dict with
    # 'valid', 'issues', 'warnings', backup counts, and 'recovery_window'.
    report = pitr_manager.validate_recovery_timeline()
    if not report["valid"]:
        for issue in report["issues"]:
            print(f"Timeline issue: {issue}")

    # Get human-oriented recovery options for a target time. Returns a dict
    # with 'feasible', 'recovery_window', a list of 'recommendations'
    # (closest-before / closest-after backups), and a 'timeline_summary'.
    recs = pitr_manager.get_recovery_recommendations(target_time)
    for option in recs["recommendations"]:
        print(f"{option['option']}: {option['backup_id']} ({option['data_loss']})")

Pruning the Timeline
^^^^^^^^^^^^^^^^^^^^

``cleanup_recovery_timeline()`` deletes old backups while preserving the
integrity of the recovery chain, keeping the most recent full backups (and any
incrementals that still depend on them) regardless of age.

.. code-block:: python

    # Preview what would be removed without deleting anything (dry_run=True).
    # If max_age_days is omitted, the configured retention_days is used.
    plan = pitr_manager.cleanup_recovery_timeline(
        max_age_days=30,
        keep_full_backups=3,
        dry_run=True,
    )
    print(f"Would delete {plan['backups_to_delete']} of {plan['total_backups']} backups")

    # Perform the cleanup for real. Returns a dict including 'deleted_count'
    # and any 'deletion_errors'.
    result = pitr_manager.cleanup_recovery_timeline(
        max_age_days=30,
        keep_full_backups=3,
    )
    print(f"Deleted {result['deleted_count']} backups")

PITR Limitations
^^^^^^^^^^^^^^^^

- Requires a complete backup chain from a full backup to the target time
- Cannot recover beyond the latest backup
- Backup chain integrity must be maintained for successful recovery

Backup Management
-----------------

Verifying Backups
^^^^^^^^^^^^^^^^^

Verify the integrity of a backup by ID:

.. code-block:: bash

    $ lvdb backup verify backup-id-here

To verify several backups, list them first and verify each ID:

.. code-block:: bash

    $ lvdb backup list --format json

**Python API:**

.. code-block:: python

    # Verify a specific backup. Returns True if the backup is intact.
    is_valid = backup_manager.verify_backup(backup_id)
    print(f"Backup valid: {is_valid}")

Streaming Verification
^^^^^^^^^^^^^^^^^^^^^^

For large backups or memory-constrained environments, LocalVectorDB provides streaming verification that can verify backup integrity without extracting the entire archive to disk. This is particularly useful for backups that exceed available disk space or memory.

Streaming verification is available through the Python API via
``verify_backup_streaming()``.

**Python API:**

.. code-block:: python

    # Basic streaming verification
    is_valid = backup_manager.verify_backup_streaming(backup_id)
    print(f"Backup valid: {is_valid}")

    # Stream verification with selective checking
    is_valid = backup_manager.verify_backup_streaming(
        backup_id,
        verify_archive_members=True  # Set to False for faster verification
    )

    # Streaming verification with progress monitoring
    def verify_large_backup_with_progress(backup_id):
        """Example of streaming verification with progress tracking."""
        print(f"Starting streaming verification of {backup_id}...")

        try:
            is_valid = backup_manager.verify_backup_streaming(
                backup_id,
                verify_archive_members=True
            )

            if is_valid:
                print("✅ Backup verification successful")
                return True
            else:
                print("❌ Backup verification failed")
                return False

        except Exception as e:
            print(f"❌ Verification error: {e}")
            return False

    # Verify multiple backups with streaming.
    # list_backups() returns a list of BackupMetadata objects (attribute access).
    for backup_info in backup_manager.list_backups():
        backup_id = backup_info.backup_id
        size_mb = backup_info.size_bytes / (1024 * 1024)

        # Use streaming verification for large backups
        if size_mb > 100:  # Backups larger than 100MB
            print(f"Streaming verification for large backup ({size_mb:.1f}MB): {backup_id}")
            is_valid = backup_manager.verify_backup_streaming(backup_id)
        else:
            print(f"Standard verification for backup ({size_mb:.1f}MB): {backup_id}")
            is_valid = backup_manager.verify_backup(backup_id)

        print(f"  Result: {'✅ Valid' if is_valid else '❌ Invalid'}")

Benefits of Streaming Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Efficiency**: Streaming verification processes backup archives in small chunks, using minimal memory regardless of backup size.

**Disk Space Savings**: No temporary extraction required - verification happens directly from the compressed archive.

**Faster for Large Backups**: Avoids the overhead of extracting large archives to disk before verification.

**Suitable for Production**: Can verify backups without impacting system resources significantly.

**Performance Comparison:**

.. code-block:: python

    import time
    from pathlib import Path

    def compare_verification_methods(backup_id):
        """Compare standard vs streaming verification performance."""

        # Get backup info
        backup_file = backup_manager._find_backup_file(backup_id)
        backup_size = Path(backup_file).stat().st_size / (1024 * 1024)  # MB

        print(f"Backup size: {backup_size:.1f} MB")

        # Standard verification
        print("\nStandard verification:")
        start_time = time.time()
        try:
            standard_valid = backup_manager.verify_backup(backup_id)
            standard_time = time.time() - start_time
            print(f"  Time: {standard_time:.2f}s")
            print(f"  Result: {'✅ Valid' if standard_valid else '❌ Invalid'}")
        except Exception as e:
            print(f"  Failed: {e}")
            standard_time = None

        # Streaming verification
        print("\nStreaming verification:")
        start_time = time.time()
        try:
            streaming_valid = backup_manager.verify_backup_streaming(backup_id)
            streaming_time = time.time() - start_time
            print(f"  Time: {streaming_time:.2f}s")
            print(f"  Result: {'✅ Valid' if streaming_valid else '❌ Invalid'}")

            # Performance comparison
            if standard_time and streaming_time:
                speedup = standard_time / streaming_time
                if speedup > 1:
                    print(f"  🚀 Streaming is {speedup:.1f}x faster")
                else:
                    print(f"  📊 Standard is {1/speedup:.1f}x faster")

        except Exception as e:
            print(f"  Failed: {e}")

    # Example usage
    recent_backups = backup_manager.list_backups()[:3]  # Test latest 3 backups
    for backup_info in recent_backups:
        compare_verification_methods(backup_info.backup_id)

Verification Options
~~~~~~~~~~~~~~~~~~~~

The streaming verification method provides several options for different use cases:

**Full Verification** (verify_archive_members=True):
- Verifies archive integrity and manifest checksum
- Checks individual file checksums within the archive
- Most thorough but takes longer for large backups

**Quick Verification** (verify_archive_members=False):
- Verifies only archive and manifest checksums
- Fastest option for regular health checks
- Suitable for automated monitoring

.. code-block:: python

    # Full verification (recommended for important backups)
    is_valid_full = backup_manager.verify_backup_streaming(
        backup_id,
        verify_archive_members=True
    )

    # Quick verification (suitable for regular health checks)
    is_valid_quick = backup_manager.verify_backup_streaming(
        backup_id,
        verify_archive_members=False
    )

    print(f"Quick check: {'✅' if is_valid_quick else '❌'}")
    print(f"Full check: {'✅' if is_valid_full else '❌'}")

Automated Backup Health Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use streaming verification in automated backup monitoring systems:

.. code-block:: python

    import schedule
    import logging
    from datetime import datetime, timedelta

    def automated_backup_health_check():
        """Daily backup health check using streaming verification."""

        logging.info("Starting automated backup health check")

        # Get backups from the last 7 days.
        # BackupMetadata.created_at is a datetime; derive a matching-tz cutoff.
        recent_backups = []

        for backup_info in backup_manager.list_backups():
            backup_date = backup_info.created_at
            cutoff_date = datetime.now(backup_date.tzinfo) - timedelta(days=7)
            if backup_date >= cutoff_date:
                recent_backups.append(backup_info)

        # Verify each recent backup
        failed_backups = []
        for backup_info in recent_backups:
            backup_id = backup_info.backup_id

            try:
                # Use quick streaming verification for daily checks
                is_valid = backup_manager.verify_backup_streaming(
                    backup_id,
                    verify_archive_members=False
                )

                if not is_valid:
                    failed_backups.append(backup_id)
                    logging.warning(f"Backup verification failed: {backup_id}")
                else:
                    logging.info(f"Backup verified successfully: {backup_id}")

            except Exception as e:
                failed_backups.append(backup_id)
                logging.error(f"Backup verification error for {backup_id}: {e}")

        # Report results
        if failed_backups:
            logging.error(f"Failed backup verifications: {failed_backups}")
            # Send alert notification here
        else:
            logging.info("All recent backups verified successfully")

        return len(failed_backups) == 0

    # Schedule daily health checks
    schedule.every().day.at("02:00").do(automated_backup_health_check)

    # Run the scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)

Best Practices for Streaming Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use for Large Backups**: Prefer streaming verification for backups over 100MB
2. **Regular Health Checks**: Use quick streaming verification (verify_archive_members=False) for daily monitoring
3. **Full Verification Periodically**: Run full streaming verification weekly or monthly
4. **Monitor Performance**: Track verification times to detect backup corruption or system issues
5. **Automate Verification**: Include streaming verification in automated backup monitoring workflows

.. code-block:: python

    # Example best practice workflow
    def backup_verification_workflow(backup_id):
        """Best practice backup verification workflow."""

        # Look up this backup's metadata (list_backups returns BackupMetadata objects)
        backup_info = next(
            (b for b in backup_manager.list_backups() if b.backup_id == backup_id),
            None,
        )
        if backup_info is None:
            raise ValueError(f"Backup not found: {backup_id}")
        size_mb = backup_info.size_bytes / (1024 * 1024)
        age_days = (datetime.now(backup_info.created_at.tzinfo) - backup_info.created_at).days

        # Choose verification method based on size and age
        if size_mb > 100:
            print(f"Large backup ({size_mb:.1f}MB) - using streaming verification")
            use_streaming = True
        else:
            print(f"Standard backup ({size_mb:.1f}MB) - using regular verification")
            use_streaming = False

        # Choose verification depth based on age
        if age_days <= 1:
            # Recent backups get full verification
            verify_members = True
            print("Recent backup - performing full verification")
        else:
            # Older backups get quick verification unless it's a weekly full check
            verify_members = (age_days % 7 == 0)  # Full check weekly
            print(f"Older backup - {'full' if verify_members else 'quick'} verification")

        # Perform verification
        if use_streaming:
            is_valid = backup_manager.verify_backup_streaming(backup_id, verify_members)
        else:
            is_valid = backup_manager.verify_backup(backup_id)

        return is_valid

Cleanup and Retention
^^^^^^^^^^^^^^^^^^^^^

Clean up backups older than the default retention period:

.. code-block:: bash

    $ lvdb backup cleanup

Clean up backups older than 7 days, keeping at least 5 full backups, and preview
the result first:

.. code-block:: bash

    $ lvdb backup cleanup --older-than 7 --keep-full 5 --dry-run

**Python API:**

.. code-block:: python

    # Clean up based on retention policy. Returns the number of backups deleted.
    # If retention_days is omitted, the configured value is used.
    deleted_count = backup_manager.cleanup_old_backups(retention_days=30)
    print(f"Deleted {deleted_count} old backups")

    # Delete a specific backup (returns True on success)
    backup_manager.delete_backup(backup_id)

Inspecting a Single Backup
^^^^^^^^^^^^^^^^^^^^^^^^^^

``get_backup_info()`` fetches the metadata for one backup by id, returning a
``BackupMetadata`` object (or ``None`` if no backup with that id exists). This is
a convenient alternative to scanning the full ``list_backups()`` result:

.. code-block:: python

    # Fetch a single backup's metadata by id (None if not found).
    info = backup_manager.get_backup_info(backup_id)
    if info is None:
        print(f"No backup found with id {backup_id}")
    else:
        print(f"Type:       {info.backup_type.value}")
        print(f"Created:    {info.created_at}")
        print(f"Size:       {info.size_bytes} bytes")
        print(f"Parent:     {info.parent_backup_id}")
        print(f"DB version: {info.database_version}")

Backup Format
-------------

LocalVectorDB uses a custom ``.lvdb-backup`` format that combines all database components into a single compressed archive.

Archive Structure
^^^^^^^^^^^^^^^^^

.. code-block:: none

    backup.lvdb-backup              # Main backup archive (gzip/lzma compressed)
    ├── manifest.json               # Backup metadata and integrity information
    ├── database.db                 # SQLite database snapshot
    ├── index.faiss                # FAISS vector index
    └── [other components]          # Additional database files

    backup.lvdb-backup.sha256       # Sidecar file with archive checksum

The backup system creates **two files** for each backup:

1. **Main Archive** (``.lvdb-backup``): Contains all database components in a compressed archive
2. **Sidecar File** (``.sha256``): Contains the SHA-256 checksum of the main archive for integrity verification

Manifest Format
^^^^^^^^^^^^^^^

The manifest.json file contains comprehensive metadata and integrity information:

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
        },
        "file_paths": {
            "database.db": "database.db",
            "index.faiss": "index.faiss"
        },
        "checksums": {
            "database.db": "a1b2c3d4e5f6...",
            "index.faiss": "f6e5d4c3b2a1..."
        },
        "archive_checksum": "9f8e7d6c5b4a3210...",
        "manifest_checksum": "1234567890abcdef..."
    }

Integrity Protection System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LocalVectorDB implements a comprehensive multi-layered integrity protection system to ensure backup reliability:

**1. Sidecar Checksum Files (.sha256)**

Every backup creates a sidecar file containing the SHA-256 checksum of the entire archive:

.. code-block:: none

    # Contents of backup.lvdb-backup.sha256
    9f8e7d6c5b4a3210fedcba0987654321  backup.lvdb-backup

This allows quick verification without opening the archive:

.. code-block:: bash

    # Verify archive integrity using system tools
    sha256sum -c backup.lvdb-backup.sha256

    # Or use LocalVectorDB verification
    lvdb backup verify backup-id-here

**2. Individual File Checksums**

Each file within the backup has its own SHA-256 checksum stored in the manifest:

.. code-block:: python

    # Access individual file checksums
    metadata = next(
        b for b in backup_manager.list_backups() if b.backup_id == backup_id
    )

    for filename, checksum in metadata.checksums.items():
        print(f"{filename}: {checksum}")

    # Example output:
    # database.db: a1b2c3d4e5f6789012345678901234567890abcdef
    # index.faiss: f6e5d4c3b2a1098765432109876543210fedcba

**3. Manifest Integrity Protection**

The manifest itself is protected with a checksum to detect tampering:

.. code-block:: python

    # The manifest_checksum protects against manifest tampering
    metadata = next(
        b for b in backup_manager.list_backups() if b.backup_id == backup_id
    )
    print(f"Manifest checksum: {metadata.manifest_checksum}")

    # Verification process:
    # 1. Extract manifest from archive
    # 2. Remove checksum fields to create normalized version
    # 3. Calculate SHA-256 of normalized manifest
    # 4. Compare with stored manifest_checksum

**4. Archive-Level Protection**

The archive_checksum provides end-to-end verification:

.. code-block:: python

    # Archive checksum protects the entire backup file
    metadata = next(
        b for b in backup_manager.list_backups() if b.backup_id == backup_id
    )
    print(f"Archive checksum: {metadata.archive_checksum}")

    # This checksum is calculated after the archive is created
    # and stored both in the manifest and the sidecar file

Verification Hierarchy
^^^^^^^^^^^^^^^^^^^^^^

The backup system uses a hierarchical verification approach:

**Level 1: Quick Archive Verification**
- Verify sidecar `.sha256` file against archive
- Fast integrity check without extracting archive
- Suitable for automated monitoring

**Level 2: Manifest Verification**
- Extract and verify manifest.json integrity
- Detect any tampering with backup metadata
- Moderate performance impact

**Level 3: Individual File Verification**
- Extract and verify each file's checksum
- Most thorough but slowest verification
- Used for complete integrity validation

.. code-block:: python

    # Example verification workflow
    def comprehensive_backup_verification(backup_id):
        """Perform all levels of backup verification."""

        # Level 1: Quick archive check
        try:
            backup_file = backup_manager._find_backup_file(backup_id)
            sidecar_path = backup_file.with_suffix(backup_file.suffix + '.sha256')

            if sidecar_path.exists():
                with open(sidecar_path, 'r') as f:
                    expected_checksum = f.readline().split()[0]

                actual_checksum = backup_manager._calculate_file_checksum(backup_file)

                if actual_checksum == expected_checksum:
                    print("✅ Level 1: Archive integrity verified")
                else:
                    print("❌ Level 1: Archive checksum mismatch")
                    return False
        except Exception as e:
            print(f"❌ Level 1: Archive verification failed: {e}")
            return False

        # Level 2 & 3: Full verification using built-in method
        try:
            is_valid = backup_manager.verify_backup(backup_id)
            if is_valid:
                print("✅ Level 2&3: Manifest and file integrity verified")
                return True
            else:
                print("❌ Level 2&3: Detailed verification failed")
                return False
        except Exception as e:
            print(f"❌ Level 2&3: Verification failed: {e}")
            return False

Integrity in Practice
^^^^^^^^^^^^^^^^^^^^^

**Backup Creation Process:**

1. Create temporary directory with database components
2. Calculate SHA-256 checksum for each file
3. Create manifest with file checksums
4. Calculate manifest checksum (without archive_checksum field)
5. Create compressed archive with all components
6. Calculate archive checksum of final archive
7. Write sidecar `.sha256` file with archive checksum
8. Update manifest with archive checksum

**Verification Process:**

1. **Quick Check**: Compare sidecar checksum with actual archive checksum
2. **Manifest Check**: Extract and verify manifest integrity
3. **File Check**: Verify individual file checksums within archive
4. **Streaming Verification**: All checks without full extraction (for large backups)

**Corruption Detection:**

The multi-layer approach detects various corruption scenarios:

- **Storage corruption**: Sidecar checksum catches archive-level corruption
- **Archive corruption**: Individual file checksums detect partial corruption
- **Tampering**: Manifest checksum detects metadata manipulation
- **Incomplete backups**: Missing files detected during verification

.. code-block:: python

    # Example: Detect specific corruption types
    def diagnose_backup_corruption(backup_id):
        """Diagnose the type of backup corruption."""

        try:
            backup_manager.verify_backup_streaming(backup_id, verify_archive_members=False)
            print("✅ Archive and manifest are intact")

            try:
                backup_manager.verify_backup_streaming(backup_id, verify_archive_members=True)
                print("✅ All individual files are intact")
            except Exception as e:
                print(f"❌ File corruption detected: {e}")

        except Exception as e:
            if "archive checksum mismatch" in str(e).lower():
                print("❌ Archive-level corruption (storage/transmission error)")
            elif "manifest integrity" in str(e).lower():
                print("❌ Manifest tampering detected")
            else:
                print(f"❌ Unknown corruption: {e}")

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

        def __init__(self, database_path: Union[str, Path],
                     faiss_index_path: Optional[Union[str, Path]] = None,
                     config: Optional[BackupConfig] = None):
            """Initialize backup manager."""

        def create_backup(self, backup_type: BackupType = BackupType.FULL,
                          parent_backup_id: Optional[str] = None,
                          backup_id: Optional[str] = None) -> str:
            """Create a backup and return its backup ID.

            Compression is taken from the BackupConfig passed to the manager.
            Incremental backups require ``parent_backup_id``.
            """

        def restore_backup(self, backup_id: str,
                           restore_location: Optional[Union[str, Path]] = None,
                           overwrite_existing: bool = False) -> Path:
            """Restore from backup; returns the restored database directory."""

        def list_backups(self,
                         backup_type: Optional[BackupType] = None
                         ) -> List[BackupMetadata]:
            """List available backups as BackupMetadata objects."""

        def verify_backup(self, backup_id: str) -> bool:
            """Verify backup integrity."""

        def verify_backup_streaming(self, backup_id: str,
                                    verify_archive_members: bool = True) -> bool:
            """Verify backup integrity without full extraction."""

        def cleanup_old_backups(self,
                               retention_days: Optional[int] = None) -> int:
            """Delete old backups; returns the number deleted."""

        def delete_backup(self, backup_id: str) -> bool:
            """Delete a specific backup."""

        def get_backup_info(self,
                            backup_id: str) -> Optional[BackupMetadata]:
            """Return one backup's metadata by id, or None if not found."""

IncrementalBackupManager
^^^^^^^^^^^^^^^^^^^^^^^^^

Incremental backups are *created* through
``BackupManager.create_backup(BackupType.INCREMENTAL, parent_backup_id=...)``.
``IncrementalBackupManager`` wraps a ``BackupManager`` and provides the
chain-aware restore path (and is used internally by PITR):

.. code-block:: python

    class IncrementalBackupManager:
        """Chain-aware incremental backup create/restore."""

        def __init__(self, backup_manager: BackupManager):
            """Wrap a BackupManager instance."""

        def create_incremental_backup(self, parent_backup_id: str,
                                      backup_id: Optional[str] = None) -> str:
            """Create an incremental backup against a parent; returns its id."""

        def restore_incremental_backup_chain(self, target_backup_id: str,
                                             restore_location: Union[str, Path]
                                             ) -> Path:
            """Restore the full+incremental chain up to target_backup_id.

            Returns the restore directory as a Path.
            """

PointInTimeRecoveryManager
^^^^^^^^^^^^^^^^^^^^^^^^^^

Point-in-time recovery is provided by ``PointInTimeRecoveryManager`` (not by
``BackupManager``). It is constructed from a ``BackupManager`` and an
``IncrementalBackupManager``:

.. code-block:: python

    class PointInTimeRecoveryManager:
        """Restore a database to any point within the backup window."""

        def __init__(self, backup_manager: BackupManager,
                     incremental_manager: IncrementalBackupManager):
            """Build the PITR manager from the backup + incremental managers."""

        def restore_to_point_in_time(self, target_timestamp: datetime,
                                     restore_location: Union[str, Path],
                                     tolerance_minutes: int = 60,
                                     dry_run: bool = False) -> Dict[str, Any]:
            """Restore the database to a specific point in time.

            Returns a dict describing the recovery operation (including a
            ``success`` flag and the recovery point used). Pass ``dry_run=True``
            to validate without restoring.
            """

        def get_recovery_timeline(self) -> List[Dict[str, Any]]:
            """List all recovery points in chronological order."""

        def find_recovery_point(self, target_timestamp: datetime,
                                tolerance_minutes: int = 60
                                ) -> Optional[Dict[str, Any]]:
            """Find the best recovery point at or before target_timestamp."""

        def validate_recovery_timeline(self) -> Dict[str, Any]:
            """Check the timeline for broken chains and gaps."""

        def cleanup_recovery_timeline(self, max_age_days: Optional[int] = None,
                                      keep_full_backups: int = 3,
                                      dry_run: bool = False) -> Dict[str, Any]:
            """Prune old backups while preserving chain integrity."""

        def get_recovery_recommendations(self, target_timestamp: datetime
                                         ) -> Dict[str, Any]:
            """Suggest recovery options (closest-before / closest-after)."""

See Also
--------

- :doc:`/migrations` - Database migration system
- :doc:`/cli` - Command-line interface
- :doc:`/installation` - Installation and configuration