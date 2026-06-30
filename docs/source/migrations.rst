.. _migrations:

===================
Migration System
===================

LocalVectorDB includes a migration system designed specifically for metadata schema
evolution. Unlike traditional database migrations that issue raw SQL, LocalVectorDB
migrations describe the *complete desired metadata schema* after the change (plus optional
column renames), and the engine reconciles the live schema to match using the built-in
``DatabaseSchema`` functionality. This keeps upgrades safe while providing rollback
capabilities.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The migration system provides:

- **Version Tracking**: Uses SQLite's ``PRAGMA user_version`` for database versioning
- **Metadata Schema Evolution**: Add, remove, and modify metadata fields safely
- **Forward & Backward Migrations**: Support for both upgrades and rollbacks
- **Column Remapping**: Rename fields and transfer data via ``column_mapping``
- **Type Safety**: Strongly typed metadata field definitions with validation
- **Dependency Resolution**: Automatic ordering of migration dependencies
- **Safety Features**: Built-in protections against destructive operations
- **Backup Integration**: Optional automatic backups before migration application

Key Concepts
^^^^^^^^^^^^

- **Migration**: A versioned change to the metadata schema definition
- **Version**: Semantic version number tracking metadata schema state
- **Migration Order**: Ordered sequence of migrations resolved from dependencies
- **Rollback**: Reverse operation that returns the schema to an earlier version
- **Migration Engine**: Core component that discovers and applies schema migrations using
  ``DatabaseSchema`` functionality

Quick Start
-----------

Checking Migration Status
^^^^^^^^^^^^^^^^^^^^^^^^^^

Check the current migration status of a database:

.. code-block:: bash

    $ lvdb migrate status mydatabase

View status as JSON:

.. code-block:: bash

    $ lvdb migrate status mydatabase --json

Applying Migrations
^^^^^^^^^^^^^^^^^^^

Apply all pending migrations:

.. code-block:: bash

    $ lvdb migrate apply mydatabase

Apply migrations up to a specific version:

.. code-block:: bash

    $ lvdb migrate apply mydatabase --to-version 1.2.0

Validate without applying (and skip the automatic backup):

.. code-block:: bash

    $ lvdb migrate apply mydatabase --dry-run --no-backup

Creating New Migrations
^^^^^^^^^^^^^^^^^^^^^^^

Create a new metadata schema migration:

.. code-block:: bash

    $ lvdb migrate create "Add user category field" --version 1.3.0

Create a migration from a specific template:

.. code-block:: bash

    $ lvdb migrate create "Add priority metadata" --version 1.3.0 --template schema

Rolling Back Migrations
^^^^^^^^^^^^^^^^^^^^^^^

Rollback to a previous version (``target_version`` is a positional argument):

.. code-block:: bash

    $ lvdb migrate rollback mydatabase 1.1.0

Validate a rollback without applying it:

.. code-block:: bash

    $ lvdb migrate rollback mydatabase 1.1.0 --dry-run

Migration Structure
-------------------

Migration Files
^^^^^^^^^^^^^^^

Migrations are Python files located in the migrations directory:

.. code-block:: none

    migrations/
    ├── __init__.py
    ├── migration_1_0_0_initial_schema.py
    ├── migration_1_1_0_add_metadata_fields.py
    └── migration_1_2_0_rename_fields.py

Migration Class Structure
^^^^^^^^^^^^^^^^^^^^^^^^^

Each migration file contains a subclass of ``Migration``. The two abstract methods
``get_schema_changes()`` and ``get_rollback_changes()`` each return a **dict** describing
the desired schema state:

- ``new_schema`` -- the complete metadata schema (``Dict[str, MetadataField]``) after the change
- ``column_mapping`` -- optional ``{old_name: new_name}`` renames whose data is transferred
- ``drop_columns`` -- whether to drop existing columns that are not present in ``new_schema``

.. code-block:: python

    from typing import Dict, Any
    from localvectordb.migration import Migration
    from localvectordb.core import MetadataField, MetadataFieldType


    class Migration_1_1_0(Migration):
        """Add category metadata field for document classification."""

        version = "1.1.0"
        description = "Add category field to metadata schema"
        dependencies = []  # versions that must be applied first

        def get_schema_changes(self) -> Dict[str, Any]:
            """Define the complete schema after this migration."""
            return {
                "new_schema": {
                    "category": MetadataField(
                        type=MetadataFieldType.TEXT,
                        indexed=True,
                        required=False,
                        default_value="general",
                    ),
                    # ...include any other fields that should remain in the schema
                },
                "column_mapping": {},
                "drop_columns": False,
            }

        def get_rollback_changes(self) -> Dict[str, Any]:
            """Define the schema to restore on rollback."""
            return {
                "new_schema": {},      # schema without the 'category' field
                "column_mapping": {},
                "drop_columns": True,  # drop 'category' on rollback
            }

Creating Migrations
-------------------

Using the CLI
^^^^^^^^^^^^^

The easiest way to create migrations is using the CLI:

.. code-block:: bash

    $ lvdb migrate create "Add priority metadata field" --version 1.4.0 --template schema

This writes a file named ``migration_1_4_0_add_priority_metadata_field.py`` containing a
schema-template migration:

.. code-block:: python

    # Migration: Add priority metadata field
    # Version: 1.4.0
    # Created: 2024-01-15 12:00:00

    from typing import Dict, Any
    from localvectordb.migration import Migration
    from localvectordb.core import MetadataField, MetadataFieldType


    class Migration_1_4_0(Migration):
        """
        Add priority metadata field
        """

        version = "1.4.0"
        description = "Add priority metadata field"
        dependencies = []  # Add version dependencies here

        def get_schema_changes(self) -> Dict[str, Any]:
            """Get schema changes to apply in forward migration."""

            # Define the complete metadata schema after this migration
            new_schema = {
                "priority": MetadataField(
                    type=MetadataFieldType.INTEGER,
                    indexed=True,
                    required=False,
                    default_value=0,
                ),
                # Add existing fields that should remain...
            }

            return {
                "new_schema": new_schema,
                "column_mapping": {},   # Optional: rename columns {'old_name': 'new_name'}
                "drop_columns": False,  # Whether to drop unused columns
            }

        def get_rollback_changes(self) -> Dict[str, Any]:
            """Get schema changes to apply for rollback."""
            return {
                "new_schema": {},
                "column_mapping": {},
                "drop_columns": True,
            }

        def validate_prerequisites(self, current_schema: Dict[str, MetadataField]) -> bool:
            """Validate migration prerequisites (optional)."""
            return True

Available ``--template`` values are ``basic``, ``schema``, and ``data``.

Manual Creation
^^^^^^^^^^^^^^^

You can also create migration files manually. The engine discovers any ``*.py`` file in the
migrations directory (other than ``__init__.py``) and loads the ``Migration`` subclass it
contains; the class's ``version`` attribute determines ordering. A descriptive filename is
recommended, e.g.:

- :file:`migration_1_0_0_initial_schema.py`
- :file:`migration_1_1_0_add_user_fields.py`
- :file:`migration_1_2_0_rename_fields.py`

Common Migration Patterns
-------------------------

Adding Metadata Fields
^^^^^^^^^^^^^^^^^^^^^^

Add new metadata fields by including them in ``new_schema``:

.. code-block:: python

    from typing import Dict, Any
    from localvectordb.migration import Migration
    from localvectordb.core import MetadataField, MetadataFieldType


    class Migration_1_1_0(Migration):
        """Add priority field for document ranking."""

        version = "1.1.0"
        description = "Add priority metadata field"

        def get_schema_changes(self) -> Dict[str, Any]:
            return {
                "new_schema": {
                    "priority": MetadataField(
                        type=MetadataFieldType.INTEGER,
                        indexed=True,
                        required=False,
                        default_value=0,
                    ),
                },
                "column_mapping": {},
                "drop_columns": False,
            }

        def get_rollback_changes(self) -> Dict[str, Any]:
            return {"new_schema": {}, "column_mapping": {}, "drop_columns": True}

Renaming Metadata Fields
^^^^^^^^^^^^^^^^^^^^^^^^

Rename a field with ``column_mapping``; the existing data is transferred to the new column:

.. code-block:: python

    class Migration_1_2_0(Migration):
        """Rename 'type' field to 'category' for clarity."""

        version = "1.2.0"
        description = "Rename type field to category"

        def get_schema_changes(self) -> Dict[str, Any]:
            return {
                "new_schema": {
                    "category": MetadataField(
                        type=MetadataFieldType.TEXT,
                        indexed=True,
                        required=False,
                    ),
                },
                "column_mapping": {"type": "category"},
                "drop_columns": True,
            }

        def get_rollback_changes(self) -> Dict[str, Any]:
            return {
                "new_schema": {
                    "type": MetadataField(
                        type=MetadataFieldType.TEXT,
                        indexed=True,
                        required=False,
                    ),
                },
                "column_mapping": {"category": "type"},
                "drop_columns": True,
            }

Modifying Field Types
^^^^^^^^^^^^^^^^^^^^^

Change a field's type by redefining it in ``new_schema``:

.. code-block:: python

    class Migration_1_3_0(Migration):
        """Change score field from INTEGER to REAL."""

        version = "1.3.0"
        description = "Change score field to support decimal values"

        def get_schema_changes(self) -> Dict[str, Any]:
            return {
                "new_schema": {
                    "score": MetadataField(
                        type=MetadataFieldType.REAL,
                        indexed=True,
                        required=False,
                        default_value=0.0,
                    ),
                },
                "column_mapping": {},
                "drop_columns": False,
            }

        def get_rollback_changes(self) -> Dict[str, Any]:
            return {
                "new_schema": {
                    "score": MetadataField(
                        type=MetadataFieldType.INTEGER,
                        indexed=True,
                        required=False,
                        default_value=0,
                    ),
                },
                "column_mapping": {},
                "drop_columns": False,
            }

Configuration
-------------

Migration settings are configured in the main configuration file under ``[migration]``.

TOML Configuration
^^^^^^^^^^^^^^^^^^

.. code-block:: toml

    [migration]
    enabled = true
    migration_dir = "./migrations"
    auto_migrate = false
    backup_before_migration = true

    # Safety settings
    require_confirmation = true
    allow_destructive_migrations = false
    max_rollback_steps = 10

    # Template settings
    migration_template_author = "Your Name"
    migration_template_format = "python"

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

Configure migrations using environment variables:

.. code-block:: bash

    export LVDB_MIGRATION_ENABLED=true
    export LVDB_MIGRATION_AUTO_MIGRATE=false
    export LVDB_MIGRATION_BACKUP_BEFORE_MIGRATION=true
    export LVDB_MIGRATION_MIGRATION_DIR=./custom-migrations

Configuration Options
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Migration Configuration Options
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``enabled``
     - ``true``
     - Enable/disable schema migration functionality
   * - ``migration_dir``
     - ``./migrations``
     - Directory containing migration files
   * - ``auto_migrate``
     - ``false``
     - Automatically apply pending migrations on startup
   * - ``backup_before_migration``
     - ``true``
     - Create backup before applying migrations
   * - ``require_confirmation``
     - ``true``
     - Require confirmation for destructive schema changes
   * - ``allow_destructive_migrations``
     - ``false``
     - Allow migrations that could drop metadata fields
   * - ``max_rollback_steps``
     - ``10``
     - Maximum number of schema versions to rollback
   * - ``migration_template_author``
     - ``None``
     - Default author for new migrations
   * - ``migration_template_format``
     - ``python``
     - Template format (``python`` or ``sql``)

Version Management
------------------

Version Schema
^^^^^^^^^^^^^^

LocalVectorDB uses semantic versioning for database schemas:

- **Major Version**: Incompatible schema changes
- **Minor Version**: Backward-compatible additions
- **Patch Version**: Backward-compatible fixes

Examples:

- ``1.0.0`` → ``1.1.0``: Added new optional metadata fields
- ``1.1.0`` → ``2.0.0``: Removed or changed existing metadata fields
- ``1.1.0`` → ``1.1.1``: Fixed metadata field definitions or constraints

Version Tracking
^^^^^^^^^^^^^^^^

Database version is tracked using:

1. **SQLite PRAGMA user_version**: Integer representation
2. **Config table**: Full semantic version string (for legacy databases)
3. **Migration log**: History of applied migrations

.. code-block:: python

    from localvectordb.versioning import VersionManager, DatabaseVersion

    # Get current database version
    vm = VersionManager("./mydatabase.sqlite")
    current_version = vm.get_database_version()
    print(f"Current version: {current_version}")

    # Check if migration to a target version is needed
    needs_migration = vm.needs_migration(DatabaseVersion("1.2.0"))

Migration Engine
----------------

Applying and Rolling Back
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``MigrationEngine`` discovers migration files and applies or rolls them back. Both
``migrate()`` and ``rollback()`` return a result dict and support ``dry_run``:

.. code-block:: python

    from localvectordb.migration import MigrationEngine

    engine = MigrationEngine(
        database_path="./mydatabase.sqlite",
        migrations_directory="./migrations",
    )

    # See what would be applied
    status = engine.get_migration_status()
    pending = engine.get_pending_migrations()  # list of version strings

    # Apply pending migrations up to the latest version (creates a backup if a
    # backup_manager is configured and auto_backup is enabled)
    result = engine.migrate(target_version="1.2.0")
    print(result["success"], result.get("applied_migrations"))

    # Roll back to an earlier version
    rollback_result = engine.rollback(target_version="1.1.0", dry_run=True)

How It Works
^^^^^^^^^^^^

The migration engine follows this process:

1. **Discovery**: Scans the migrations directory for migration files
2. **Validation**: Validates migration syntax and dependencies
3. **Ordering**: Resolves execution order from dependencies
4. **Backup**: Creates a backup if a backup manager is configured
5. **Execution**: Applies migrations in dependency order
6. **Logging**: Records each applied migration in the migration log

Dependency Resolution
^^^^^^^^^^^^^^^^^^^^^

Migrations can depend on other migrations via the ``dependencies`` attribute:

.. code-block:: python

    class Migration_1_2_0(Migration):
        version = "1.2.0"
        dependencies = ["1.1.0"]  # Requires version 1.1.0

        def get_schema_changes(self) -> Dict[str, Any]:
            return {
                "new_schema": {
                    "analytics_enabled": MetadataField(
                        type=MetadataFieldType.BOOLEAN,
                        indexed=False,
                        default_value=False,
                    ),
                },
                "column_mapping": {},
                "drop_columns": False,
            }

        def get_rollback_changes(self) -> Dict[str, Any]:
            return {"new_schema": {}, "column_mapping": {}, "drop_columns": True}

``get_migration_order()`` uses topological sorting to determine the correct execution order.

Safety Features
---------------

Destructive Operations
^^^^^^^^^^^^^^^^^^^^^^

Operations that can lose data -- dropping columns (``drop_columns=True``) or removing
fields from ``new_schema`` -- are guarded by configuration:

- ``allow_destructive_migrations`` must be enabled to permit them.
- ``require_confirmation`` causes the CLI to prompt before applying them.

Automatic Backups
^^^^^^^^^^^^^^^^^

When a ``BackupManager`` is supplied to the engine and ``auto_backup`` is enabled (or
``migrate(create_backup=True)`` is passed), a full backup is created before migrations are
applied:

.. code-block:: python

    from localvectordb.backup import BackupManager, BackupConfig
    from localvectordb.migration import MigrationEngine

    backup_manager = BackupManager(
        "./mydatabase.sqlite",
        config=BackupConfig(backup_location="./backups"),
    )
    engine = MigrationEngine(
        "./mydatabase.sqlite",
        backup_manager=backup_manager,
        auto_backup=True,
    )
    engine.migrate()  # creates a pre-migration backup automatically

Rollback System
---------------

How Rollbacks Work
^^^^^^^^^^^^^^^^^^

Each migration implements ``get_rollback_changes()``, returning the schema dict to restore
when the migration is rolled back:

.. code-block:: python

    def get_rollback_changes(self) -> Dict[str, Any]:
        # Restore the schema to its state before this migration
        return {
            "new_schema": {
                "old_score": MetadataField(type=MetadataFieldType.INTEGER, indexed=False),
            },
            "column_mapping": {},
            "drop_columns": True,
        }

Rollback Limitations
^^^^^^^^^^^^^^^^^^^^

- Data loss: Rollbacks may lose data added after the migration
- Irreversible operations: Some changes cannot be undone
- Dependency chains: Migrations are rolled back in reverse order

Safe Rollback Practices
^^^^^^^^^^^^^^^^^^^^^^^

1. **Test rollbacks** in a development environment
2. **Backup before rollback** to preserve recent data
3. **Validate data integrity** after rollback
4. **Consider forward fixes** instead of rollbacks

Advanced Features
-----------------

Custom Migration Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can subclass the engine to customize discovery. Note that ``discover_migrations()``
returns a ``Dict[str, MigrationScript]`` keyed by version:

.. code-block:: python

    from typing import Dict
    from localvectordb.migration import MigrationEngine, MigrationScript


    class CustomMigrationEngine(MigrationEngine):
        def discover_migrations(self) -> Dict[str, MigrationScript]:
            # Custom logic to find migrations (database, API, etc.)
            return super().discover_migrations()

Migration Validation
^^^^^^^^^^^^^^^^^^^^

Override ``validate_prerequisites()`` to guard a migration before it is applied:

.. code-block:: python

    class Migration_1_5_0(Migration):
        version = "1.5.0"
        description = "Validated migration"

        def validate_prerequisites(self, current_schema: Dict[str, MetadataField]) -> bool:
            """Return False (or raise) to block the migration."""
            return "existing_field" in current_schema

        def get_schema_changes(self) -> Dict[str, Any]:
            return {"new_schema": {}, "column_mapping": {}, "drop_columns": False}

        def get_rollback_changes(self) -> Dict[str, Any]:
            return {"new_schema": {}, "column_mapping": {}, "drop_columns": False}

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Migration fails with a version conflict**

Another process may have applied migrations. Solutions:

- Refresh migration status with ``lvdb migrate status``
- Check for concurrent migration processes
- Resolve version conflicts manually

**Rollback fails**

Some operations cannot be automatically reversed. Solutions:

- Restore from backup
- Create a forward migration to fix the issue
- Manually fix the database state

**Migration hangs or times out**

Large data migrations may take time. Solutions:

- Break large migrations into smaller steps
- Run migrations during maintenance windows

Best Practices
--------------

Migration Development
^^^^^^^^^^^^^^^^^^^^^

1. **Test thoroughly** in a development environment
2. **Keep migrations small** and focused
3. **Include rollback logic** for all migrations
4. **Document complex migrations** with clear descriptions
5. **Avoid destructive operations** when possible

Production Deployment
^^^^^^^^^^^^^^^^^^^^^

1. **Backup before migrations** in production
2. **Test the migration path** in staging
3. **Use** ``--dry-run`` **first** to preview changes
4. **Plan for rollback** if issues occur

API Reference
-------------

Migration Base Class
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pathlib import Path
    from typing import Dict, Any, List, Union
    from abc import ABC, abstractmethod
    from localvectordb.core import MetadataField

    class Migration(ABC):
        """Base class for metadata schema migrations."""

        version: str            # Semantic version (e.g., "1.2.0")
        description: str        # Human-readable description
        dependencies: List[str] = []  # Required versions

        def __init__(self, database_path: Union[str, Path]):
            """Migrations are instantiated with the target database path."""

        @abstractmethod
        def get_schema_changes(self) -> Dict[str, Any]:
            """Return the forward schema dict:
            {'new_schema': {...}, 'column_mapping': {...}, 'drop_columns': bool}."""

        @abstractmethod
        def get_rollback_changes(self) -> Dict[str, Any]:
            """Return the rollback schema dict (same structure as above)."""

        def validate_prerequisites(self, current_schema: Dict[str, MetadataField]) -> bool:
            """Optional pre-flight validation. Return False to block the migration."""
            return True

MigrationEngine
^^^^^^^^^^^^^^^

.. code-block:: python

    from pathlib import Path
    from typing import Dict, Any, List, Optional, Union
    from localvectordb.backup import BackupManager

    class MigrationEngine:
        """Core migration engine for metadata schema evolution."""

        def __init__(self, database_path: Union[str, Path],
                     migrations_directory: Union[str, Path] = "./migrations",
                     backup_manager: Optional[BackupManager] = None,
                     auto_backup: bool = True):
            """Initialize the migration engine."""

        def discover_migrations(self) -> Dict[str, "MigrationScript"]:
            """Discover migration scripts, keyed by version."""

        def get_migration_order(self) -> List[str]:
            """Return versions in dependency (topological) order."""

        def get_applied_migrations(self) -> List[Dict[str, Any]]:
            """Return the log of applied migrations."""

        def get_pending_migrations(self, target_version: Optional[str] = None) -> List[str]:
            """Return versions that have not yet been applied."""

        def migrate(self, target_version: Optional[str] = None,
                    dry_run: bool = False,
                    create_backup: Optional[bool] = None) -> Dict[str, Any]:
            """Apply pending migrations up to target_version (latest if None)."""

        def rollback(self, target_version: str,
                     dry_run: bool = False,
                     create_backup: Optional[bool] = None) -> Dict[str, Any]:
            """Roll back to a specific version."""

        def get_migration_status(self) -> Dict[str, Any]:
            """Return current migration status."""

        def create_migration_template(self, version: str, description: str,
                                      template_type: str = "basic") -> Path:
            """Create a new migration file (template_type: basic, schema, data)."""

VersionManager
^^^^^^^^^^^^^^

.. code-block:: python

    from pathlib import Path
    from typing import Optional, Union
    from localvectordb.versioning import DatabaseVersion

    class VersionManager:
        """Manages database versioning."""

        def __init__(self, db_path: Union[str, Path]):
            """Initialize version manager."""

        def get_database_version(self, conn=None) -> DatabaseVersion:
            """Get the current database version."""

        def set_database_version(self, version: DatabaseVersion, conn=None) -> None:
            """Set the database version."""

        def needs_migration(self, target_version: Optional[DatabaseVersion] = None,
                            conn=None) -> bool:
            """Check whether migration to target_version is needed."""

See Also
--------

- :doc:`/backup` - Backup and recovery system
- :doc:`/cli` - Command-line interface
- :doc:`/installation` - Installation and configuration
