# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/cli/_backup.py

"""Backup and restore CLI commands for LocalVectorDB.

Provides command-line interface for backup operations including:
- Creating full and incremental backups
- Listing available backups
- Restoring from backups
- Verifying backup integrity
- Cleaning up old backups
- Point-in-time recovery operations
"""

from datetime import datetime, timezone
from pathlib import Path

import click

from localvectordb.backup import (
    BackupConfig,
    BackupManager,
    BackupType,
    CompressionAlgorithm,
    IncrementalBackupManager,
    PointInTimeRecoveryManager,
)
from localvectordb_server.cli._utils import EXIT_CODE_ERROR, format_table, print_json_output


@click.group('backup')
@click.pass_context
def backup_group(ctx):
    """
    Backup and restore operations for databases.

    Provides comprehensive backup functionality including full backups,
    incremental backups, and point-in-time recovery capabilities.

    \b
    Examples:
        \b
        lvdb backup create mydb --type full
        lvdb backup list --database mydb
        lvdb backup restore backup-id --to-location ./restored
        lvdb backup verify backup-id
        lvdb backup cleanup --older-than 30
    """
    pass


@backup_group.command('create')
@click.argument('database_name')
@click.option('--type', '-t', 'backup_type',
              type=click.Choice(['full', 'incremental']),
              default='full',
              help='Type of backup to create')
@click.option('--parent', '-p',
              help='Parent backup ID for incremental backups')
@click.option('--location', '-l',
              type=click.Path(exists=False, file_okay=False),
              help='Backup storage location (default: ./backups)')
@click.option('--compression', '-c',
              type=click.Choice(['none', 'gzip', 'lzma']),
              default='gzip',
              help='Compression algorithm to use')
@click.option('--no-verify', is_flag=True,
              help='Skip backup integrity verification')
@click.option('--exclude-faiss', is_flag=True,
              help='Exclude FAISS index from backup')
@click.option('--json', 'output_json', is_flag=True,
              help='Output result in JSON format')
@click.pass_context
def create_backup(ctx, database_name, backup_type, parent, location,
                  compression, no_verify, exclude_faiss, output_json):
    """
    Create a backup of the specified database.

    Creates either a full backup (complete database snapshot) or an incremental
    backup (changes since parent backup). Full backups are self-contained while
    incremental backups require the parent backup chain for restoration.

    \b
    Examples:
        \b
        lvdb backup create mydb --type full
        lvdb backup create mydb --type incremental --parent backup-abc123
        lvdb backup create mydb --compression lzma --location /backups
    """

    db_folder = ctx.obj.get("db_folder")
    if not db_folder:
        click.secho("Database folder not specified", fg="red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    try:
        # Set up database paths
        db_path = Path(db_folder) / f"{database_name}.sqlite"
        faiss_path = Path(db_folder) / f"{database_name}.faiss"

        if not db_path.exists():
            click.secho(f"Database '{database_name}' not found in {db_folder}",
                       fg="red", err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        # Configure backup settings
        backup_location = Path(location) if location else Path("./backups")

        config = BackupConfig(
            backup_location=backup_location,
            compression_algorithm=CompressionAlgorithm(compression),
            verify_integrity=not no_verify,
            include_faiss_index=not exclude_faiss
        )

        # Create backup manager
        backup_manager = BackupManager(db_path, faiss_path, config)

        if backup_type == 'full':
            # Create full backup
            backup_id = backup_manager.create_backup(BackupType.FULL)

            result = {
                'success': True,
                'backup_id': backup_id,
                'backup_type': 'full',
                'database': database_name,
                'location': str(backup_location)
            }

        else:  # incremental
            if not parent:
                click.secho("Parent backup ID required for incremental backups",
                           fg="red", err=True)
                raise click.exceptions.Exit(EXIT_CODE_ERROR)

            # Create incremental backup
            inc_manager = IncrementalBackupManager(backup_manager)
            backup_id = inc_manager.create_incremental_backup(parent)

            result = {
                'success': True,
                'backup_id': backup_id,
                'backup_type': 'incremental',
                'parent_backup_id': parent,
                'database': database_name,
                'location': str(backup_location)
            }

        if output_json:
            print_json_output(result)
        else:
            click.secho("✓ Backup created successfully", fg="green")
            click.echo(f"  Backup ID: {backup_id}")
            click.echo(f"  Type: {backup_type}")
            click.echo(f"  Database: {database_name}")
            click.echo(f"  Location: {backup_location}")
            if backup_type == 'incremental':
                click.echo(f"  Parent: {parent}")

    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'database': database_name
        }

        if output_json:
            print_json_output(result)
        else:
            click.secho(f"✗ Backup failed: {e}", fg="red", err=True)

        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@backup_group.command('list')
@click.option('--database', '-d',
              help='Filter backups for specific database')
@click.option('--type', '-t',
              type=click.Choice(['full', 'incremental']),
              help='Filter by backup type')
@click.option('--limit', '-n', type=int,
              help='Limit number of backups shown')
@click.option('--json', 'output_json', is_flag=True,
              help='Output in JSON format')
@click.option('--location', '-l',
              type=click.Path(exists=True, file_okay=False),
              help='Backup storage location to scan (default: ./backups)')
@click.pass_context
def list_backups(ctx, database, type, limit, output_json, location):
    """
    List available backups.

    Shows all available backups with their metadata including creation time,
    type, size, and parent relationships for incremental backups.

    \b
    Examples:
        \b
        lvdb backup list
        lvdb backup list --database mydb --type full
        lvdb backup list --limit 10 --json
    """

    try:
        backup_location = Path(location) if location else Path("./backups")

        if not backup_location.exists():
            if output_json:
                print_json_output({'backups': []})
            else:
                click.echo("No backups directory found")
            return

        # Find all backup files
        backup_files = list(backup_location.glob("*.lvdb-backup"))

        if not backup_files:
            if output_json:
                print_json_output({'backups': []})
            else:
                click.echo("No backups found")
            return

        # For listing, we'll scan backup files directly since we might not have database access
        backups = []

        for backup_file in backup_files:
            try:
                import tarfile

                with tarfile.open(backup_file, "r:*") as tar:
                    # Read manifest
                    manifest_file = tar.extractfile("manifest.json")
                    if manifest_file:
                        import json
                        manifest_data = json.load(manifest_file)

                        # Filter by criteria
                        if database and manifest_data['database_name'] != database:
                            continue

                        if type and manifest_data['backup_type'] != type:
                            continue

                        backups.append({
                            'backup_id': manifest_data['backup_id'],
                            'database_name': manifest_data['database_name'],
                            'backup_type': manifest_data['backup_type'],
                            'created_at': manifest_data['created_at'],
                            'database_version': manifest_data['database_version'],
                            'size_bytes': manifest_data.get('size_bytes', 0),
                            'parent_backup_id': manifest_data.get('parent_backup_id'),
                            'compression': manifest_data.get('compression_algorithm', 'unknown'),
                            'file_path': str(backup_file)
                        })

            except Exception as e:
                click.echo(f"Warning: Could not read backup {backup_file}: {e}", err=True)
                continue

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created_at'], reverse=True)

        # Apply limit
        if limit:
            backups = backups[:limit]

        if output_json:
            print_json_output({'backups': backups, 'total_count': len(backups)})

        else:
            if not backups:
                click.echo("No backups match the specified criteria")
                return

            click.echo(f"Found {len(backups)} backup(s):\n")

            # Format as table
            headers = ['Backup ID', 'Database', 'Type', 'Created', 'Size', 'Parent']
            rows = []

            for backup in backups:
                size_mb = backup['size_bytes'] / (1024 * 1024) if backup['size_bytes'] else 0
                created_dt = datetime.fromisoformat(backup['created_at'].replace('Z', '+00:00'))
                created_str = created_dt.strftime('%Y-%m-%d %H:%M:%S')
                parent_short = backup['parent_backup_id'][:8] if backup['parent_backup_id'] else '-'

                rows.append([
                    backup['backup_id'][:8],
                    backup['database_name'],
                    backup['backup_type'],
                    created_str,
                    f"{size_mb:.1f}MB",
                    parent_short
                ])

            click.echo(format_table(headers, rows))

    except Exception as e:
        if output_json:
            print_json_output({'error': str(e), 'backups': []})
        else:
            click.secho(f"Error listing backups: {e}", fg="red", err=True)

        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@backup_group.command('restore')
@click.argument('backup_id')
@click.option('--to-location', '-t',
              type=click.Path(exists=False, file_okay=False),
              help='Directory to restore to (default: original location)')
@click.option('--overwrite', is_flag=True,
              help='Overwrite existing files without confirmation')
@click.option('--location', '-l',
              type=click.Path(exists=True, file_okay=False),
              help='Backup storage location (default: ./backups)')
@click.option('--json', 'output_json', is_flag=True,
              help='Output result in JSON format')
@click.pass_context
def restore_backup(ctx, backup_id, to_location, overwrite, location, output_json):
    """
    Restore a database from backup.

    Restores a database from either a full backup or by applying an incremental
    backup chain. For incremental backups, automatically finds and applies the
    complete backup chain starting from the base full backup.

    \b
    Examples:
        \b
        lvdb backup restore abc12345
        lvdb backup restore abc12345 --to-location ./restored
        lvdb backup restore abc12345 --overwrite
    """

    try:
        backup_location = Path(location) if location else Path("./backups")

        if not backup_location.exists():
            error = f"Backup location not found: {backup_location}"
            if output_json:
                print_json_output({'success': False, 'error': error})
            else:
                click.secho(f"✗ {error}", fg="red", err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        # Find backup file (support partial backup ID)
        backup_file = None
        for f in backup_location.glob("*.lvdb-backup"):
            if backup_id in f.name:
                backup_file = f
                break

        if not backup_file:
            error = f"Backup not found: {backup_id}"
            if output_json:
                print_json_output({'success': False, 'error': error})
            else:
                click.secho(f"✗ {error}", fg="red", err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        # Read backup metadata
        import json
        import tarfile

        with tarfile.open(backup_file, "r:*") as tar:
            manifest_file = tar.extractfile("manifest.json")
            manifest_data = json.load(manifest_file)

        database_name = manifest_data['database_name']
        backup_type = manifest_data['backup_type']

        # Determine restore location
        if not to_location:
            db_folder = ctx.obj.get("db_folder", ".")
            to_location = Path(db_folder)
        else:
            to_location = Path(to_location)

        to_location.mkdir(parents=True, exist_ok=True)

        # Create temporary backup manager for restoration
        temp_db_path = to_location / f"{database_name}.sqlite"
        temp_faiss_path = to_location / f"{database_name}.faiss"

        config = BackupConfig(backup_location=backup_location)
        backup_manager = BackupManager(temp_db_path, temp_faiss_path, config)

        if backup_type == 'full':
            # Simple full backup restore
            restored_path = backup_manager.restore_backup(
                manifest_data['backup_id'],
                to_location,
                overwrite_existing=overwrite
            )

        else:
            # Incremental backup restore
            inc_manager = IncrementalBackupManager(backup_manager)
            restored_path = inc_manager.restore_incremental_backup_chain(
                manifest_data['backup_id'],
                to_location
            )

        result = {
            'success': True,
            'backup_id': manifest_data['backup_id'],
            'database_name': database_name,
            'backup_type': backup_type,
            'restored_to': str(restored_path),
            'files_restored': [
                f"{database_name}.sqlite",
                f"{database_name}.faiss" if (restored_path / f"{database_name}.faiss").exists() else None
            ]
        }

        # Remove None values
        result['files_restored'] = [f for f in result['files_restored'] if f]

        if output_json:
            print_json_output(result)
        else:
            click.secho("✓ Restore completed successfully", fg="green")
            click.echo(f"  Backup ID: {manifest_data['backup_id'][:8]}")
            click.echo(f"  Database: {database_name}")
            click.echo(f"  Type: {backup_type}")
            click.echo(f"  Restored to: {restored_path}")
            for file in result['files_restored']:
                click.echo(f"  - {file}")

    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'backup_id': backup_id
        }

        if output_json:
            print_json_output(result)
        else:
            click.secho(f"✗ Restore failed: {e}", fg="red", err=True)

        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@backup_group.command('verify')
@click.argument('backup_id')
@click.option('--location', '-l',
              type=click.Path(exists=True, file_okay=False),
              help='Backup storage location (default: ./backups)')
@click.option('--json', 'output_json', is_flag=True,
              help='Output result in JSON format')
def verify_backup(backup_id, location, output_json):
    """
    Verify backup integrity.

    Verifies that a backup file is valid and can be restored by checking
    file structure, checksums, and metadata consistency.

    \b
    Examples:
        \b
        lvdb backup verify abc12345
        lvdb backup verify abc12345 --json
    """

    try:
        backup_location = Path(location) if location else Path("./backups")

        # Find backup file
        backup_file = None
        for f in backup_location.glob("*.lvdb-backup"):
            if backup_id in f.name:
                backup_file = f
                break

        if not backup_file:
            error = f"Backup not found: {backup_id}"
            if output_json:
                print_json_output({'success': False, 'error': error})
            else:
                click.secho(f"✗ {error}", fg="red", err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        # Verify backup using BackupManager
        import json
        import tarfile

        # Read manifest first
        with tarfile.open(backup_file, "r:*") as tar:
            manifest_file = tar.extractfile("manifest.json")
            manifest_data = json.load(manifest_file)

        database_name = manifest_data['database_name']

        # Create temporary backup manager for verification
        temp_db_path = Path("/tmp") / f"{database_name}.sqlite"
        config = BackupConfig(backup_location=backup_location)
        backup_manager = BackupManager(temp_db_path, config=config)

        # Perform verification
        is_valid = backup_manager.verify_backup(manifest_data['backup_id'])

        result = {
            'success': is_valid,
            'backup_id': manifest_data['backup_id'],
            'database_name': database_name,
            'backup_type': manifest_data['backup_type'],
            'created_at': manifest_data['created_at'],
            'file_path': str(backup_file),
            'verification_passed': is_valid
        }

        if output_json:
            print_json_output(result)
        else:
            if is_valid:
                click.secho("✓ Backup verification passed", fg="green")
            else:
                click.secho("✗ Backup verification failed", fg="red")

            click.echo(f"  Backup ID: {manifest_data['backup_id'][:8]}")
            click.echo(f"  Database: {database_name}")
            click.echo(f"  Type: {manifest_data['backup_type']}")
            click.echo(f"  Created: {manifest_data['created_at']}")

    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'backup_id': backup_id,
            'verification_passed': False
        }

        if output_json:
            print_json_output(result)
        else:
            click.secho(f"✗ Verification error: {e}", fg="red", err=True)

        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@backup_group.command('cleanup')
@click.option('--older-than', type=int, default=30,
              help='Delete backups older than N days (default: 30)')
@click.option('--keep-full', type=int, default=3,
              help='Minimum number of full backups to keep (default: 3)')
@click.option('--location', '-l',
              type=click.Path(exists=True, file_okay=False),
              help='Backup storage location (default: ./backups)')
@click.option('--dry-run', is_flag=True,
              help='Show what would be deleted without actually deleting')
@click.option('--json', 'output_json', is_flag=True,
              help='Output result in JSON format')
def cleanup_backups(older_than, keep_full, location, dry_run, output_json):
    """
    Clean up old backups based on retention policy.

    Removes old backups while maintaining backup chain integrity and keeping
    a minimum number of full backups for recovery purposes.

    \b
    Examples:
        \b
        lvdb backup cleanup --older-than 30
        lvdb backup cleanup --keep-full 5 --dry-run
        lvdb backup cleanup --older-than 7 --json
    """

    try:
        backup_location = Path(location) if location else Path("./backups")

        if not backup_location.exists():
            if output_json:
                print_json_output({'deleted_count': 0, 'message': 'No backup location found'})
            else:
                click.echo("No backup location found")
            return

        # Find backup files and create a dummy database for the PITR manager
        backup_files = list(backup_location.glob("*.lvdb-backup"))

        if not backup_files:
            if output_json:
                print_json_output({'deleted_count': 0, 'message': 'No backups found'})
            else:
                click.echo("No backups found")
            return

        # Create managers for cleanup (using first backup's database as reference)
        if backup_files:
            import json
            import tarfile

            # Read first backup to get database info
            with tarfile.open(backup_files[0], "r:*") as tar:
                manifest_file = tar.extractfile("manifest.json")
                manifest_data = json.load(manifest_file)

            database_name = manifest_data['database_name']

            # Create temporary paths for managers
            temp_db_path = Path("/tmp") / f"{database_name}.sqlite"
            config = BackupConfig(backup_location=backup_location, retention_days=older_than)

            backup_manager = BackupManager(temp_db_path, config=config)
            inc_manager = IncrementalBackupManager(backup_manager)
            pitr_manager = PointInTimeRecoveryManager(backup_manager, inc_manager)

            # Perform cleanup
            cleanup_result = pitr_manager.cleanup_recovery_timeline(
                max_age_days=older_than,
                keep_full_backups=keep_full,
                dry_run=dry_run
            )

            if output_json:
                print_json_output(cleanup_result)

            else:
                if dry_run:
                    click.secho(f"Dry run: Would delete {cleanup_result['backups_to_delete']} backup(s)",
                               fg="yellow")
                else:
                    click.secho(f"✓ Deleted {cleanup_result['deleted_count']} backup(s)", fg="green")

                click.echo(f"  Total backups: {cleanup_result['total_backups']}")
                click.echo(f"  Would delete: {cleanup_result['backups_to_delete']}")
                click.echo(f"  Keeping: {cleanup_result['backups_to_keep']}")

                if cleanup_result.get('deletion_errors'):
                    click.echo("\nErrors:")
                    for error in cleanup_result['deletion_errors']:
                        click.secho(f"  - {error}", fg="red")

    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'deleted_count': 0
        }

        if output_json:
            print_json_output(result)
        else:
            click.secho(f"✗ Cleanup failed: {e}", fg="red", err=True)

        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@backup_group.command('pitr')
@click.argument('timestamp')
@click.option('--to-location', '-t',
              type=click.Path(exists=False, file_okay=False),
              required=True,
              help='Directory to restore to')
@click.option('--tolerance', type=int, default=60,
              help='Tolerance in minutes for finding recovery point (default: 60)')
@click.option('--location', '-l',
              type=click.Path(exists=True, file_okay=False),
              help='Backup storage location (default: ./backups)')
@click.option('--dry-run', is_flag=True,
              help='Validate recovery without actually restoring')
@click.option('--json', 'output_json', is_flag=True,
              help='Output result in JSON format')
def point_in_time_recovery(timestamp, to_location, tolerance, location, dry_run, output_json):
    """
    Perform point-in-time recovery to a specific timestamp.

    Restores the database to the state it was in at the specified timestamp
    by finding the closest backup point and applying the backup chain.

    Timestamp format: YYYY-MM-DD HH:MM:SS or ISO format

    \b
    Examples:
        \b
        lvdb backup pitr "2024-01-15 14:30:00" --to-location ./restored
        lvdb backup pitr "2024-01-15T14:30:00Z" --tolerance 120 --dry-run
    """

    try:
        from datetime import datetime

        # Parse timestamp
        try:
            # Try ISO format first
            if 'T' in timestamp:
                target_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                # Try simple format
                target_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                target_dt = target_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            error = f"Invalid timestamp format: {timestamp}. Use 'YYYY-MM-DD HH:MM:SS' or ISO format"
            if output_json:
                print_json_output({'success': False, 'error': error})
            else:
                click.secho(f"✗ {error}", fg="red", err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        backup_location = Path(location) if location else Path("./backups")
        to_location = Path(to_location)

        # Create managers (need at least one backup to determine database name)
        backup_files = list(backup_location.glob("*.lvdb-backup"))

        if not backup_files:
            error = "No backups available for point-in-time recovery"
            if output_json:
                print_json_output({'success': False, 'error': error})
            else:
                click.secho(f"✗ {error}", fg="red", err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        # Get database name from first backup
        import json
        import tarfile

        with tarfile.open(backup_files[0], "r:*") as tar:
            manifest_file = tar.extractfile("manifest.json")
            manifest_data = json.load(manifest_file)

        database_name = manifest_data['database_name']

        # Create managers
        temp_db_path = to_location / f"{database_name}.sqlite"
        config = BackupConfig(backup_location=backup_location)

        backup_manager = BackupManager(temp_db_path, config=config)
        inc_manager = IncrementalBackupManager(backup_manager)
        pitr_manager = PointInTimeRecoveryManager(backup_manager, inc_manager)

        # Perform point-in-time recovery
        recovery_result = pitr_manager.restore_to_point_in_time(
            target_dt,
            to_location,
            tolerance_minutes=tolerance,
            dry_run=dry_run
        )

        if output_json:
            # Convert datetime objects to strings for JSON serialization
            if 'target_timestamp' in recovery_result:
                recovery_result['target_timestamp'] = recovery_result['target_timestamp'].isoformat()
            if 'actual_timestamp' in recovery_result:
                recovery_result['actual_timestamp'] = recovery_result['actual_timestamp'].isoformat()

            print_json_output(recovery_result)

        else:
            if recovery_result['success']:
                if dry_run:
                    click.secho("✓ Point-in-time recovery validation passed", fg="green")
                else:
                    click.secho("✓ Point-in-time recovery completed", fg="green")

                click.echo(f"  Target timestamp: {timestamp}")
                if 'actual_timestamp' in recovery_result:
                    click.echo(f"  Actual recovery point: {recovery_result['actual_timestamp']}")
                if 'time_difference_seconds' in recovery_result:
                    diff_min = recovery_result['time_difference_seconds'] / 60
                    click.echo(f"  Time difference: {diff_min:.1f} minutes")

                if not dry_run:
                    click.echo(f"  Restored to: {recovery_result['restore_location']}")

            else:
                click.secho("✗ Point-in-time recovery failed", fg="red")
                if 'error' in recovery_result:
                    click.echo(f"  Error: {recovery_result['error']}")

                if 'available_timeline' in recovery_result:
                    timeline = recovery_result['available_timeline']
                    if timeline:
                        click.echo("\nAvailable recovery points:")
                        for point in timeline[-5:]:  # Show last 5 points
                            click.echo(f"  - {point['timestamp']} ({point['backup_type']})")

    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'timestamp': timestamp
        }

        if output_json:
            print_json_output(result)
        else:
            click.secho(f"✗ Point-in-time recovery error: {e}", fg="red", err=True)

        raise click.exceptions.Exit(EXIT_CODE_ERROR)
