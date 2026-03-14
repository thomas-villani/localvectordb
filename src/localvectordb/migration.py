# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/migration.py

"""Metadata schema migration system for LocalVectorDB.

This module provides comprehensive metadata schema migration capabilities for
LocalVectorDB. It focuses on evolving the metadata schema using the existing
DatabaseSchema functionality rather than raw SQL operations.

The migration system handles:
- Adding, removing, and modifying metadata fields
- Column remapping and data migration
- Schema validation and compatibility checking
- Version tracking and rollback functionality

Classes:
    Migration: Base class for metadata schema migrations
    MigrationEngine: Core migration management engine
    MigrationScript: Individual migration script representation
"""

import hashlib
import importlib.util
import inspect
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from localvectordb._pools import ReadWriteLock
from localvectordb._schema import DatabaseSchema
from localvectordb.backup import BackupManager, BackupType
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.versioning import DatabaseVersion, VersionManager

logger = logging.getLogger(__name__)


def serialize_metadata_field(field: MetadataField) -> Dict[str, Any]:
    """Serialize a MetadataField object to a JSON-compatible dictionary."""
    return {
        'type': field.type.value if isinstance(field.type, MetadataFieldType) else str(field.type),
        'indexed': field.indexed,
        'required': field.required,
        'default_value': field.default_value,
        'embedding_enabled': getattr(field, 'embedding_enabled', False),
        'fts_enabled': getattr(field, 'fts_enabled', False)
    }


def deserialize_metadata_field(data: Dict[str, Any]) -> MetadataField:
    """Deserialize a dictionary back to a MetadataField object."""
    return MetadataField(
        type=MetadataFieldType(data['type']),
        indexed=data.get('indexed', False),
        required=data.get('required', False),
        default_value=data.get('default_value'),
        embedding_enabled=data.get('embedding_enabled', False),
        fts_enabled=data.get('fts_enabled', False)
    )


def serialize_schema_changes(changes: Dict[str, Any]) -> str:
    """Serialize schema changes containing MetadataField objects to JSON string."""
    serialized = {}

    if 'new_schema' in changes:
        serialized['new_schema'] = {
            name: serialize_metadata_field(field) if isinstance(field, MetadataField) else field
            for name, field in changes['new_schema'].items()
        }

    if 'column_mapping' in changes:
        serialized['column_mapping'] = changes['column_mapping']

    if 'drop_columns' in changes:
        serialized['drop_columns'] = changes['drop_columns']

    return json.dumps(serialized)


def deserialize_schema_changes(json_str: str) -> Dict[str, Any]:
    """Deserialize JSON string back to schema changes with MetadataField objects."""
    data = json.loads(json_str)
    result = {}

    if 'new_schema' in data:
        result['new_schema'] = {
            name: deserialize_metadata_field(field_data) if isinstance(field_data, dict) else field_data
            for name, field_data in data['new_schema'].items()
        }

    if 'column_mapping' in data:
        result['column_mapping'] = data['column_mapping']

    if 'drop_columns' in data:
        result['drop_columns'] = data['drop_columns']

    return result


class Migration(ABC):
    """
    Abstract base class for metadata schema migrations.

    All migration classes must inherit from this base class and implement
    the get_schema_changes() and get_rollback_changes() methods for
    forward and backward migrations.

    This class focuses on metadata schema evolution rather than raw SQL
    operations, leveraging the existing DatabaseSchema functionality.

    Attributes
    ----------
    version : str
        Target version for this migration
    description : str
        Human-readable description of the migration
    dependencies : List[str]
        List of migration versions that must be applied before this one
    """

    version: str = "0.0.0"
    description: str = "Base migration"
    dependencies: List[str] = []

    def __init__(self, database_path: Union[str, Path]):
        self.database_path = Path(database_path)
        self.version_manager = VersionManager(self.database_path)

    @abstractmethod
    def get_schema_changes(self) -> Dict[str, Any]:
        """
        Get the schema changes to apply in the forward migration.

        Returns
        -------
        Dict[str, Any]
            Schema changes specification with the following structure:
            {
                'new_schema': Dict[str, MetadataField],  # Complete new schema
                'column_mapping': Dict[str, str],        # Rename mappings: old -> new
                'drop_columns': bool                     # Whether to drop unused columns
            }

        Examples
        --------
        Add new fields::

            return {
                'new_schema': {
                    'user_id': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                    'priority': MetadataField(type=MetadataFieldType.INTEGER, default_value=0)
                }
            }

        Rename and modify fields::

            return {
                'new_schema': {
                    'author_name': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                    'created_date': MetadataField(type=MetadataFieldType.DATE, indexed=True)
                },
                'column_mapping': {
                    'author': 'author_name',
                    'timestamp': 'created_date'
                }
            }
        """
        pass

    @abstractmethod
    def get_rollback_changes(self) -> Dict[str, Any]:
        """
        Get the schema changes to apply for rollback.

        Returns
        -------
        Dict[str, Any]
            Schema changes specification for rolling back this migration
        """
        pass

    def validate_prerequisites(self, current_schema: Dict[str, MetadataField]) -> bool:
        """
        Validate that prerequisites for this migration are met.

        Parameters
        ----------
        current_schema : Dict[str, MetadataField]
            Current metadata schema

        Returns
        -------
        bool
            True if prerequisites are met
        """
        return True


class MigrationScript:
    """
    Represents a single migration script file.

    Parameters
    ----------
    file_path : Path
        Path to the migration script file
    migration_class : Type[Migration]
        Migration class loaded from the file
    """

    def __init__(self, file_path: Path, migration_class: Type[Migration]):
        self.file_path = file_path
        self.migration_class = migration_class
        self.version = migration_class.version
        self.description = migration_class.description
        self.dependencies = migration_class.dependencies

        # Calculate checksum of the migration file
        self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of the migration file."""
        sha256_hash = hashlib.sha256()
        with open(self.file_path, 'rb') as f:
            sha256_hash.update(f.read())
        return sha256_hash.hexdigest()

    def create_instance(self, database_path: Union[str, Path]) -> Migration:
        """Create an instance of the migration class."""
        return self.migration_class(database_path)


class MigrationEngine:
    """
    Core database migration management engine.

    Handles discovery, validation, execution, and rollback of database migrations.
    Integrates with the backup system for safety and the versioning system for
    tracking applied migrations.

    Parameters
    ----------
    database_path : Union[str, Path]
        Path to the database file
    migrations_directory : Union[str, Path], optional
        Directory containing migration scripts. Defaults to "./migrations"
    backup_manager : BackupManager, optional
        Backup manager for creating safety backups before migrations
    auto_backup : bool
        Whether to automatically create backups before migrations
    """

    def __init__(
            self,
            database_path: Union[str, Path],
            migrations_directory: Union[str, Path] = "./migrations",
            backup_manager: Optional[BackupManager] = None,
            auto_backup: bool = True
    ):
        self.database_path = Path(database_path)
        self.migrations_directory = Path(migrations_directory)
        self.backup_manager = backup_manager
        self.auto_backup = auto_backup

        # Initialize managers
        self.version_manager = VersionManager(self.database_path)
        self._read_write_lock = ReadWriteLock()
        self.database_schema = DatabaseSchema(self.database_path, self._read_write_lock)

        # Create migrations directory if it doesn't exist
        self.migrations_directory.mkdir(parents=True, exist_ok=True)

        # Cache for loaded migrations
        self._migration_cache: Dict[str, MigrationScript] = {}
        self._migration_order_cache: Optional[List[str]] = None

    def discover_migrations(self) -> Dict[str, MigrationScript]:
        """
        Discover and load all migration scripts from the migrations directory.

        Returns
        -------
        Dict[str, MigrationScript]
            Dictionary mapping version strings to MigrationScript objects
        """

        migrations = {}

        # Find all Python files in migrations directory
        for py_file in self.migrations_directory.glob("*.py"):
            if py_file.name.startswith("__"):
                continue  # Skip __init__.py, __pycache__, etc.

            try:
                migration_script = self._load_migration_from_file(py_file)
                if migration_script:
                    migrations[migration_script.version] = migration_script
                    logger.debug(f"Loaded migration {migration_script.version}: {migration_script.description}")

            except Exception as e:
                logger.error(f"Failed to load migration from {py_file}: {e}")

        self._migration_cache = migrations
        self._migration_order_cache = None  # Reset cache

        logger.info(f"Discovered {len(migrations)} migrations")
        return migrations

    def _load_migration_from_file(self, file_path: Path) -> Optional[MigrationScript]:
        """Load a migration class from a Python file."""

        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find Migration class in the module
            migration_class = None
            for _name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                        issubclass(obj, Migration) and
                        obj != Migration):
                    migration_class = obj
                    break

            if not migration_class:
                logger.warning(f"No Migration class found in {file_path}")
                return None

            return MigrationScript(file_path, migration_class)

        except Exception as e:
            logger.error(f"Error loading migration from {file_path}: {e}")
            return None

    def get_migration_order(self) -> List[str]:
        """
        Get the correct order for applying migrations based on dependencies.

        Returns
        -------
        List[str]
            List of version strings in the order they should be applied

        Raises
        ------
        ValueError
            If circular dependencies are detected
        """

        if self._migration_order_cache:
            return self._migration_order_cache

        migrations = self.discover_migrations()
        if not migrations:
            return []

        # Topological sort based on dependencies
        visited = set()
        temp_visited = set()
        order = []

        def visit(version: str) -> None:
            if version in temp_visited:
                raise ValueError(f"Circular dependency detected involving migration {version}")

            if version in visited:
                return

            temp_visited.add(version)

            # Visit dependencies first
            migration = migrations.get(version)
            if migration:
                for dep_version in migration.dependencies:
                    if dep_version in migrations:
                        visit(dep_version)
                    else:
                        raise ValueError(f"Migration {version} depends on missing migration {dep_version}")

            temp_visited.remove(version)
            visited.add(version)
            order.append(version)

        # Visit all migrations
        for version in migrations:
            visit(version)

        self._migration_order_cache = order
        return order

    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """
        Get list of migrations that have been applied to the database.

        Returns
        -------
        List[Dict[str, Any]]
            List of applied migration records
        """

        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute("""
                    SELECT version, applied_at, rollback_script, checksum
                    FROM migration_log
                    ORDER BY applied_at
                """)

                applied = []
                for row in cursor.fetchall():
                    applied.append({
                        'version': row[0],
                        'applied_at': row[1],
                        'rollback_script': row[2],
                        'checksum': row[3]
                    })

                return applied

        except sqlite3.OperationalError:
            # migration_log table doesn't exist
            return []

    def get_pending_migrations(self, target_version: Optional[str] = None) -> List[str]:
        """
        Get list of migrations that need to be applied.

        Parameters
        ----------
        target_version : str, optional
            Target version to migrate to. If None, migrates to latest.

        Returns
        -------
        List[str]
            List of migration versions that need to be applied
        """

        applied_versions = {m['version'] for m in self.get_applied_migrations()}
        migration_order = self.get_migration_order()

        if target_version:
            # Find migrations up to target version
            try:
                target_index = migration_order.index(target_version)
                migration_order = migration_order[:target_index + 1]
            except ValueError:
                logger.warning(f"Target version {target_version} not found in migrations")
                return []

        # Return migrations that haven't been applied yet
        return [version for version in migration_order if version not in applied_versions]

    def migrate(
            self,
            target_version: Optional[str] = None,
            dry_run: bool = False,
            create_backup: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Apply pending migrations up to the target version.

        Parameters
        ----------
        target_version : str, optional
            Target version to migrate to. If None, migrates to latest.
        dry_run : bool
            If True, validate migrations without applying them
        create_backup : bool, optional
            Whether to create a backup before migration. If None, uses auto_backup setting.

        Returns
        -------
        Dict[str, Any]
            Migration results with status and details
        """

        if create_backup is None:
            create_backup = self.auto_backup

        logger.info(f"Starting migration to {target_version or 'latest'}")

        # Get pending migrations
        pending_migrations = self.get_pending_migrations(target_version)

        if not pending_migrations:
            return {
                'success': True,
                'message': 'No pending migrations',
                'applied_migrations': [],
                'target_version': target_version,
                'dry_run': dry_run
            }

        logger.info(f"Found {len(pending_migrations)} pending migrations: {pending_migrations}")

        if dry_run:
            return {
                'success': True,
                'message': f'Dry run: would apply {len(pending_migrations)} migrations',
                'pending_migrations': pending_migrations,
                'target_version': target_version,
                'dry_run': True
            }

        # Create backup if requested
        backup_id = None
        if create_backup and self.backup_manager:
            try:
                backup_id = self.backup_manager.create_backup(BackupType.FULL)
                logger.info(f"Created pre-migration backup: {backup_id}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

        # Apply migrations
        applied_migrations = []
        migration_errors = []

        try:
            migrations = self.discover_migrations()

            with sqlite3.connect(self.database_path) as conn:
                for version in pending_migrations:
                    try:
                        logger.info(f"Applying migration {version}")

                        migration_script = migrations.get(version)
                        if not migration_script:
                            raise ValueError(f"Migration script not found for version {version}")

                        # Create migration instance
                        migration = migration_script.create_instance(self.database_path)

                        # Load current schema for validation
                        current_schema = self.database_schema.load_metadata_schema(conn)

                        # Validate prerequisites
                        if not migration.validate_prerequisites(current_schema):
                            raise ValueError(f"Prerequisites not met for migration {version}")

                        # Get schema changes
                        schema_changes = migration.get_schema_changes()

                        # Apply schema changes using DatabaseSchema
                        if 'new_schema' in schema_changes:
                            new_schema = schema_changes['new_schema']
                            column_mapping = schema_changes.get('column_mapping', {})
                            drop_columns = schema_changes.get('drop_columns', False)

                            # Apply schema changes
                            change_results = self.database_schema.update_metadata_schema(
                                new_schema,
                                db_connection=conn,
                                drop_columns=drop_columns,
                                column_mapping=column_mapping
                            )

                            # Check for errors
                            if change_results.get('errors'):
                                error_msg = f"Schema update errors: {', '.join(change_results['errors'])}"
                                raise ValueError(error_msg)

                            logger.debug(f"Schema changes applied: {change_results}")

                        # Record migration in log
                        rollback_changes = migration.get_rollback_changes()
                        rollback_script = None
                        if rollback_changes:
                            # Store rollback changes as JSON for later use
                            rollback_script = serialize_schema_changes(rollback_changes)

                        self.version_manager.record_migration(
                            version,
                            rollback_script=rollback_script,
                            checksum=migration_script.checksum,
                            conn=conn
                        )

                        # Update database version
                        db_version = DatabaseVersion(version)
                        self.version_manager.set_database_version(db_version, conn)

                        applied_migrations.append(version)
                        logger.info(f"Successfully applied migration {version}")

                    except Exception as e:
                        error_msg = f"Failed to apply migration {version}: {e}"
                        logger.error(error_msg)
                        migration_errors.append(error_msg)
                        break  # Stop on first error

            # Determine success
            success = len(migration_errors) == 0

            return {
                'success': success,
                'applied_migrations': applied_migrations,
                'migration_errors': migration_errors,
                'backup_id': backup_id,
                'target_version': target_version,
                'dry_run': False
            }

        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'applied_migrations': applied_migrations,
                'migration_errors': migration_errors,
                'backup_id': backup_id,
                'target_version': target_version,
                'dry_run': False
            }

    def rollback(
            self,
            target_version: str,
            dry_run: bool = False,
            create_backup: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Rollback migrations to a target version.

        Parameters
        ----------
        target_version : str
            Target version to rollback to
        dry_run : bool
            If True, validate rollback without applying it
        create_backup : bool, optional
            Whether to create a backup before rollback. If None, uses auto_backup setting.

        Returns
        -------
        Dict[str, Any]
            Rollback results with status and details
        """

        if create_backup is None:
            create_backup = self.auto_backup

        logger.info(f"Starting rollback to version {target_version}")

        # Get applied migrations in reverse order
        applied_migrations = self.get_applied_migrations()

        # Find target version index
        try:
            target_db_version = DatabaseVersion(target_version)
        except ValueError:
            return {
                'success': False,
                'error': f'Invalid target version: {target_version}',
                'dry_run': dry_run
            }

        # Find migrations to rollback (in reverse order)
        migrations_to_rollback = []
        for migration in reversed(applied_migrations):
            migration_version = DatabaseVersion(migration['version'])
            if migration_version > target_db_version:
                migrations_to_rollback.append(migration)
            else:
                break

        if not migrations_to_rollback:
            return {
                'success': True,
                'message': f'Already at or below target version {target_version}',
                'rolled_back_migrations': [],
                'target_version': target_version,
                'dry_run': dry_run
            }

        logger.info(f"Found {len(migrations_to_rollback)} migrations to rollback")

        if dry_run:
            return {
                'success': True,
                'message': f'Dry run: would rollback {len(migrations_to_rollback)} migrations',
                'migrations_to_rollback': [m['version'] for m in migrations_to_rollback],
                'target_version': target_version,
                'dry_run': True
            }

        # Create backup if requested
        backup_id = None
        if create_backup and self.backup_manager:
            try:
                backup_id = self.backup_manager.create_backup(BackupType.FULL)
                logger.info(f"Created pre-rollback backup: {backup_id}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

        # Perform rollback
        rolled_back_migrations = []
        rollback_errors = []

        try:
            migrations = self.discover_migrations()

            with sqlite3.connect(self.database_path) as conn:
                for migration_record in migrations_to_rollback:
                    version = migration_record['version']

                    try:
                        logger.info(f"Rolling back migration {version}")

                        migration_script = migrations.get(version)
                        if migration_script:
                            # Use migration class for rollback
                            migration_instance = migration_script.create_instance(self.database_path)
                            rollback_changes = migration_instance.get_rollback_changes()

                            if rollback_changes and 'new_schema' in rollback_changes:
                                # Apply rollback schema changes using DatabaseSchema
                                new_schema = rollback_changes['new_schema']
                                column_mapping = rollback_changes.get('column_mapping', {})
                                drop_columns = rollback_changes.get('drop_columns', False)

                                change_results = self.database_schema.update_metadata_schema(
                                    new_schema,
                                    db_connection=conn,
                                    drop_columns=drop_columns,
                                    column_mapping=column_mapping
                                )

                                # Check for errors
                                if change_results.get('errors'):
                                    error_msg = f"Schema rollback errors: {', '.join(change_results['errors'])}"
                                    raise ValueError(error_msg)

                                logger.debug(f"Schema rollback applied: {change_results}")

                            # Remove from migration log
                            conn.execute("DELETE FROM migration_log WHERE version = ?", (version,))

                            rolled_back_migrations.append(version)
                            logger.info(f"Successfully rolled back migration {version}")

                        elif migration_record['rollback_script']:
                            # Use stored rollback changes (JSON format)
                            try:
                                rollback_changes = deserialize_schema_changes(migration_record['rollback_script'])

                                if 'new_schema' in rollback_changes:
                                    new_schema = rollback_changes['new_schema']
                                    column_mapping = rollback_changes.get('column_mapping', {})
                                    drop_columns = rollback_changes.get('drop_columns', False)

                                    change_results = self.database_schema.update_metadata_schema(
                                        new_schema,
                                        db_connection=conn,
                                        drop_columns=drop_columns,
                                        column_mapping=column_mapping
                                    )

                                    # Check for errors
                                    if change_results.get('errors'):
                                        error_msg = f"Schema rollback errors: {', '.join(change_results['errors'])}"
                                        raise ValueError(error_msg)

                                # Remove from migration log
                                conn.execute("DELETE FROM migration_log WHERE version = ?", (version,))

                                rolled_back_migrations.append(version)
                                logger.info(f"Successfully rolled back migration {version} using stored changes")

                            except json.JSONDecodeError:
                                # Legacy SQL rollback script
                                conn.executescript(migration_record['rollback_script'])
                                conn.execute("DELETE FROM migration_log WHERE version = ?", (version,))
                                rolled_back_migrations.append(version)
                                logger.info(f"Successfully rolled back migration {version} using legacy SQL script")

                        else:
                            logger.warning(f"No rollback method available for migration {version}")
                            rollback_errors.append(f"No rollback method for migration {version}")

                    except Exception as e:
                        error_msg = f"Failed to rollback migration {version}: {e}"
                        logger.error(error_msg)
                        rollback_errors.append(error_msg)
                        break  # Stop on first error

                # Update database version to target
                if rolled_back_migrations and not rollback_errors:
                    self.version_manager.set_database_version(target_db_version, conn)

            success = len(rollback_errors) == 0

            return {
                'success': success,
                'rolled_back_migrations': rolled_back_migrations,
                'rollback_errors': rollback_errors,
                'backup_id': backup_id,
                'target_version': target_version,
                'dry_run': False
            }

        except Exception as e:
            logger.error(f"Rollback process failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'rolled_back_migrations': rolled_back_migrations,
                'rollback_errors': rollback_errors,
                'backup_id': backup_id,
                'target_version': target_version,
                'dry_run': False
            }

    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of database migrations.

        Returns
        -------
        Dict[str, Any]
            Migration status information
        """

        current_version = self.version_manager.get_database_version()
        applied_migrations = self.get_applied_migrations()
        available_migrations = self.discover_migrations()
        migration_order = self.get_migration_order()
        pending_migrations = self.get_pending_migrations()

        return {
            'current_version': str(current_version),
            'total_available_migrations': len(available_migrations),
            'applied_migrations_count': len(applied_migrations),
            'pending_migrations_count': len(pending_migrations),
            'migration_order': migration_order,
            'applied_migrations': applied_migrations,
            'pending_migrations': pending_migrations,
            'latest_available_version': migration_order[-1] if migration_order else None
        }

    def create_migration_template(
            self,
            version: str,
            description: str,
            template_type: str = "basic"
    ) -> Path:
        """
        Create a new migration template file.

        Parameters
        ----------
        version : str
            Version for the new migration
        description : str
            Description of the migration
        template_type : str
            Type of template to create ("basic", "schema", "data")

        Returns
        -------
        Path
            Path to the created migration file
        """

        # Validate version format
        try:
            DatabaseVersion(version)
        except ValueError as e:
            raise ValueError(f"Invalid version format: {e}") from e

        # Create filename
        safe_description = "".join(c if c.isalnum() or c in "_-" else "_" for c in description.lower())
        filename = f"migration_{version.replace('.', '_')}_{safe_description}.py"
        file_path = self.migrations_directory / filename

        # Generate template content
        template_content = self._get_migration_template(version, description, template_type)

        # Write template file
        with open(file_path, 'w') as f:
            f.write(template_content)

        logger.info(f"Created migration template: {file_path}")
        return file_path

    def _get_migration_template(
            self,
            version: str,
            description: str,
            template_type: str
    ) -> str:
        """Generate migration template content."""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if template_type == "schema":
            template = f'''# Migration: {description}
# Version: {version}
# Created: {timestamp}

from typing import Dict, Any
from localvectordb.migration import Migration
from localvectordb.core import MetadataField, MetadataFieldType


class Migration_{version.replace('.', '_')}(Migration):
    """
    {description}
    """

    version = "{version}"
    description = "{description}"
    dependencies = []  # Add version dependencies here

    def get_schema_changes(self) -> Dict[str, Any]:
        """Get schema changes to apply in forward migration."""

        # Load current schema first (optional - for reference)
        # current_schema = self.database_schema.load_metadata_schema()

        # Define the new complete metadata schema after this migration
        new_schema = {{
            # Example: Add new fields
            'user_id': MetadataField(
                type=MetadataFieldType.TEXT,
                indexed=True,
                required=False,
                default_value=None
            ),
            'priority': MetadataField(
                type=MetadataFieldType.INTEGER,
                indexed=True,
                required=False,
                default_value=0
            ),
            # Add existing fields that should remain...
        }}

        return {{
            'new_schema': new_schema,
            'column_mapping': {{}},  # Optional: rename columns {{'old_name': 'new_name'}}
            'drop_columns': False    # Whether to drop unused columns
        }}

    def get_rollback_changes(self) -> Dict[str, Any]:
        """Get schema changes to apply for rollback."""

        # Define the schema state before this migration (for rollback)
        rollback_schema = {{
            # Define schema without the changes from this migration
            # Remove fields that were added, restore old names, etc.
        }}

        return {{
            'new_schema': rollback_schema,
            'column_mapping': {{}},  # Reverse any column renames
            'drop_columns': False
        }}

    def validate_prerequisites(self, current_schema: Dict[str, MetadataField]) -> bool:
        """Validate migration prerequisites."""

        # Example: Check if required fields exist
        # if 'required_field' not in current_schema:
        #     return False

        return True
'''

        elif template_type == "data":
            template = f'''# Migration: {description}
# Version: {version}
# Created: {timestamp}

from typing import Dict, Any
from localvectordb.migration import Migration
from localvectordb.core import MetadataField, MetadataFieldType


class Migration_{version.replace('.', '_')}(Migration):
    """
    {description}

    Note: This migration only changes data/default values, not schema structure.
    Use 'schema' template if you need to add/remove/modify fields.
    """

    version = "{version}"
    description = "{description}"
    dependencies = []  # Add version dependencies here

    def get_schema_changes(self) -> Dict[str, Any]:
        """Get schema changes - data-only migration keeps existing schema."""

        # For data-only migrations, we usually don't change the schema structure
        # but might update default values or field properties

        # Load current schema and modify only default values or properties
        # current_schema = self.database_schema.load_metadata_schema()

        # Example: Update default values
        updated_schema = {{
            # Keep existing fields but update properties
            # 'existing_field': MetadataField(
            #     type=MetadataFieldType.TEXT,
            #     default_value='new_default_value'  # Changed default
            # )
        }}

        return {{
            'new_schema': updated_schema,
            'column_mapping': {{}},
            'drop_columns': False
        }}

    def get_rollback_changes(self) -> Dict[str, Any]:
        """Get rollback changes - restore previous default values."""

        # Restore previous default values
        rollback_schema = {{
            # Restore previous field properties
            # 'existing_field': MetadataField(
            #     type=MetadataFieldType.TEXT,
            #     default_value='old_default_value'  # Restore old default
            # )
        }}

        return {{
            'new_schema': rollback_schema,
            'column_mapping': {{}},
            'drop_columns': False
        }}

    def validate_prerequisites(self, current_schema: Dict[str, MetadataField]) -> bool:
        """Validate that required fields exist for data migration."""

        # Example: Ensure required fields exist
        # required_fields = ['field1', 'field2']
        # for field in required_fields:
        #     if field not in current_schema:
        #         return False

        return True
'''

        else:  # basic template
            template = f'''# Migration: {description}
# Version: {version}
# Created: {timestamp}

from typing import Dict, Any
from localvectordb.migration import Migration
from localvectordb.core import MetadataField, MetadataFieldType


class Migration_{version.replace('.', '_')}(Migration):
    """
    {description}
    """

    version = "{version}"
    description = "{description}"
    dependencies = []  # Add version dependencies here

    def get_schema_changes(self) -> Dict[str, Any]:
        """Get schema changes to apply in forward migration."""

        # TODO: Define your metadata schema changes
        new_schema = {{
            # Example: Add new fields
            # 'field_name': MetadataField(
            #     type=MetadataFieldType.TEXT,
            #     indexed=True,
            #     required=False,
            #     default_value=None
            # ),
        }}

        return {{
            'new_schema': new_schema,
            'column_mapping': {{}},  # Optional: {{'old_name': 'new_name'}}
            'drop_columns': False    # Set to True to drop unused columns
        }}

    def get_rollback_changes(self) -> Dict[str, Any]:
        """Get schema changes to apply for rollback."""

        # TODO: Define rollback schema (state before this migration)
        rollback_schema = {{
            # Revert the changes made in get_schema_changes()
        }}

        return {{
            'new_schema': rollback_schema,
            'column_mapping': {{}},
            'drop_columns': False
        }}

    def validate_prerequisites(self, current_schema: Dict[str, MetadataField]) -> bool:
        """Validate migration prerequisites."""

        # TODO: Add prerequisite validation
        # Example:
        # if 'required_field' not in current_schema:
        #     return False

        return True
'''

        return template
