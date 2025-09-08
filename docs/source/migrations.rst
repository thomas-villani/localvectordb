.. _migrations:

===================
Migration System
===================

LocalVectorDB includes a sophisticated migration system designed specifically for metadata schema evolution. Unlike traditional database migrations that modify tables and columns, LocalVectorDB migrations focus on evolving metadata field definitions using the built-in DatabaseSchema functionality, ensuring smooth upgrades while maintaining data integrity and providing rollback capabilities.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The migration system provides:

- **Version Tracking**: Uses SQLite's PRAGMA user_version for database versioning
- **Metadata Schema Evolution**: Add, remove, and modify metadata fields safely
- **Forward & Backward Migrations**: Support for both upgrades and rollbacks
- **Column Remapping**: Rename fields and transfer data automatically
- **Type Safety**: Strongly typed metadata field definitions with validation
- **Dependency Resolution**: Automatic ordering of migration dependencies
- **Safety Features**: Built-in protections against destructive operations
- **Backup Integration**: Automatic backups before migration application

Key Concepts
^^^^^^^^^^^^

- **Migration**: A versioned change to the metadata schema definition
- **Version**: Semantic version number tracking metadata schema state  
- **Migration Chain**: Ordered sequence of metadata schema migrations from one version to another
- **Rollback**: Reverse operation to undo a metadata schema migration
- **Migration Engine**: Core component that discovers and applies schema migrations using DatabaseSchema functionality

Quick Start
-----------

Checking Migration Status
^^^^^^^^^^^^^^^^^^^^^^^^^

Check the current migration status of a database:

.. code-block:: bash

    $ lvdb migrate status mydatabase

View detailed migration information:

.. code-block:: bash

    $ lvdb migrate status mydatabase --verbose

Applying Migrations
^^^^^^^^^^^^^^^^^^^

Apply all pending migrations:

.. code-block:: bash

    $ lvdb migrate apply mydatabase

Apply migrations up to a specific version:

.. code-block:: bash

    $ lvdb migrate apply mydatabase --to-version 1.2.0

Apply with automatic backup:

.. code-block:: bash

    $ lvdb migrate apply mydatabase --backup

Creating New Migrations
^^^^^^^^^^^^^^^^^^^^^^^

Create a new metadata schema migration:

.. code-block:: bash

    $ lvdb migrate create "Add user category field" --version 1.3.0

Create migration with specific template:

.. code-block:: bash

    $ lvdb migrate create "Add priority metadata" --template schema --author "John Doe"

Rolling Back Migrations
^^^^^^^^^^^^^^^^^^^^^^^

Rollback to a previous version:

.. code-block:: bash

    $ lvdb migrate rollback mydatabase 1.1.0

Rollback with confirmation prompt:

.. code-block:: bash

    $ lvdb migrate rollback mydatabase 1.1.0 --confirm

Migration Structure
-------------------

Migration Files
^^^^^^^^^^^^^^^

Migrations are Python files located in the migrations directory:

.. code-block:: none

    migrations/
    ├── __init__.py
    ├── 001_initial_schema_v1.0.0.py
    ├── 002_add_metadata_fields_v1.1.0.py
    └── 003_optimize_indexes_v1.2.0.py

Migration Class Structure
^^^^^^^^^^^^^^^^^^^^^^^^^

Each migration file contains a migration class focused on metadata schema changes:

.. code-block:: python

    from localvectordb.migration import Migration
    from localvectordb.core import MetadataField, MetadataFieldType
    from typing import Dict, List, Tuple

    class Migration_001_add_category_field(Migration):
        """Add category metadata field for document classification."""
        
        version = "1.1.0"
        description = "Add category field to metadata schema"
        dependencies = []  # No dependencies for first migration
        
        def get_schema_changes(self) -> List[Tuple[str, str, MetadataField]]:
            """Define schema changes to apply."""
            return [
                ('add', 'category', MetadataField(
                    type=MetadataFieldType.TEXT,
                    indexed=True,
                    required=False,
                    default_value='general'
                ))
            ]
        
        def get_rollback_changes(self) -> List[Tuple[str, str, MetadataField]]:
            """Define rollback changes."""
            return [
                ('remove', 'category', None)
            ]

Creating Migrations
-------------------

Using the CLI
^^^^^^^^^^^^^

The easiest way to create migrations is using the CLI:

.. code-block:: bash

    $ lvdb migrate create "Add priority metadata field" --version 1.4.0

This creates a migration file with a metadata schema template:

.. code-block:: python

    from localvectordb.migration import Migration
    from localvectordb.core import MetadataField, MetadataFieldType
    from typing import Dict, List, Tuple

    class Migration_004_add_priority_field(Migration):
        """Add priority metadata field."""
        
        version = "1.4.0"
        description = "Add priority metadata field"
        dependencies = ["1.3.0"]  # Depends on previous version
        
        def get_schema_changes(self) -> List[Tuple[str, str, MetadataField]]:
            """Define schema changes to apply."""
            # TODO: Implement schema changes
            return []
        
        def get_rollback_changes(self) -> List[Tuple[str, str, MetadataField]]:
            """Define rollback changes."""
            # TODO: Implement rollback logic
            return []

Manual Creation
^^^^^^^^^^^^^^^

You can also create migration files manually following the naming convention:

``{sequence}_{description}_v{version}.py``

Examples:

- :file:`001_initial_schema_v1.0.0.py`
- :file:`002_add_user_table_v1.1.0.py` 
- :file:`003_optimize_indexes_v1.2.0.py`

Common Migration Patterns
-------------------------

Adding Metadata Fields
^^^^^^^^^^^^^^^^^^^^^^

Add new metadata fields to the schema:

.. code-block:: python

    from localvectordb.migration import Migration
    from localvectordb.core import MetadataField, MetadataFieldType
    from typing import List, Tuple

    class Migration_002_add_priority_field(Migration):
        """Add priority field for document ranking."""
        
        version = "1.1.0"
        description = "Add priority metadata field"
        
        def get_schema_changes(self) -> List[Tuple[str, str, MetadataField]]:
            return [
                ('add', 'priority', MetadataField(
                    type=MetadataFieldType.INTEGER,
                    indexed=True,
                    required=False,
                    default_value=0
                ))
            ]
        
        def get_rollback_changes(self) -> List[Tuple[str, str, MetadataField]]:
            return [('remove', 'priority', None)]

Renaming Metadata Fields
^^^^^^^^^^^^^^^^^^^^^^^^

Rename existing metadata fields with data migration:

.. code-block:: python

    class Migration_003_rename_category_field(Migration):
        """Rename 'type' field to 'category' for clarity."""
        
        version = "1.2.0"
        description = "Rename type field to category"
        
        def get_schema_changes(self) -> List[Tuple[str, str, MetadataField]]:
            return [
                ('rename', 'type', 'category'),
                ('modify', 'category', MetadataField(
                    type=MetadataFieldType.TEXT,
                    indexed=True,
                    required=False
                ))
            ]
        
        def get_rollback_changes(self) -> List[Tuple[str, str, MetadataField]]:
            return [
                ('rename', 'category', 'type'),
                ('modify', 'type', MetadataField(
                    type=MetadataFieldType.TEXT,
                    indexed=True,
                    required=False
                ))
            ]

Modifying Field Types
^^^^^^^^^^^^^^^^^^^^^

Change metadata field types with validation:

.. code-block:: python

    class Migration_004_change_score_type(Migration):
        """Change score field from INTEGER to REAL."""
        
        version = "1.3.0"
        description = "Change score field to support decimal values"
        
        def get_schema_changes(self) -> List[Tuple[str, str, MetadataField]]:
            return [
                ('modify', 'score', MetadataField(
                    type=MetadataFieldType.REAL,
                    indexed=True,
                    required=False,
                    default_value=0.0
                ))
            ]
        
        def get_rollback_changes(self) -> List[Tuple[str, str, MetadataField]]:
            return [
                ('modify', 'score', MetadataField(
                    type=MetadataFieldType.INTEGER,
                    indexed=True,
                    required=False,
                    default_value=0
                ))
            ]

Complex Schema Changes
^^^^^^^^^^^^^^^^^^^^^^

Multiple field operations in one migration:

.. code-block:: python

    class Migration_005_restructure_metadata(Migration):
        """Restructure metadata schema for improved organization."""
        
        version = "2.0.0"
        description = "Major metadata schema restructuring"
        is_destructive = True  # Marks breaking changes
        
        def get_schema_changes(self) -> List[Tuple[str, str, MetadataField]]:
            return [
                # Remove deprecated fields
                ('remove', 'old_field1', None),
                ('remove', 'old_field2', None),
                
                # Add new structured fields
                ('add', 'document_info', MetadataField(
                    type=MetadataFieldType.JSON,
                    indexed=False,
                    required=False
                )),
                ('add', 'created_by', MetadataField(
                    type=MetadataFieldType.TEXT,
                    indexed=True,
                    required=True
                )),
                
                # Modify existing field
                ('modify', 'tags', MetadataField(
                    type=MetadataFieldType.JSON,
                    indexed=True,  # Now indexed
                    required=False
                ))
            ]
        
        def get_rollback_changes(self) -> List[Tuple[str, str, MetadataField]]:
            # Rollback in reverse order
            return [
                ('modify', 'tags', MetadataField(
                    type=MetadataFieldType.JSON,
                    indexed=False,
                    required=False
                )),
                ('remove', 'created_by', None),
                ('remove', 'document_info', None),
                ('add', 'old_field2', MetadataField(
                    type=MetadataFieldType.TEXT,
                    indexed=False
                )),
                ('add', 'old_field1', MetadataField(
                    type=MetadataFieldType.TEXT,
                    indexed=False
                ))
            ]

Configuration
-------------

Migration settings are configured in the main configuration file.

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
     - Allow migrations that could lose metadata fields
   * - ``max_rollback_steps``
     - ``10``
     - Maximum number of schema versions to rollback
   * - ``migration_template_author``
     - ``None``
     - Default author for new migrations
   * - ``migration_template_format``
     - ``schema``
     - Template format (schema, python)

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
2. **Config table**: Full semantic version string
3. **Migration log**: History of applied migrations

.. code-block:: python

    from localvectordb.versioning import VersionManager

    # Get current database version
    vm = VersionManager("./mydatabase.db")
    current_version = vm.get_current_version()
    print(f"Current version: {current_version}")
    
    # Check if migration is needed
    target_version = "1.2.0"
    needs_migration = vm.needs_migration(target_version)

Migration Engine
----------------

How It Works
^^^^^^^^^^^^

The migration engine follows this process:

1. **Discovery**: Scans migration directory for migration files
2. **Validation**: Validates migration syntax and dependencies
3. **Planning**: Creates migration plan from current to target version
4. **Backup**: Creates backup if configured
5. **Execution**: Applies migrations in dependency order
6. **Verification**: Verifies successful application
7. **Logging**: Records migration in migration log

Dependency Resolution
^^^^^^^^^^^^^^^^^^^^^

Migrations can depend on other migrations:

.. code-block:: python

    class Migration_003_add_analytics(Migration):
        version = "1.2.0"
        dependencies = ["1.1.0"]  # Requires version 1.1.0
        
        def get_schema_changes(self) -> List[Tuple[str, str, MetadataField]]:
            return [
                ('add', 'analytics_enabled', MetadataField(
                    type=MetadataFieldType.BOOLEAN,
                    indexed=False,
                    default_value=False
                ))
            ]

The engine uses topological sorting to determine the correct execution order.

Migration States
^^^^^^^^^^^^^^^^

Migrations can be in various states:

- **Pending**: Not yet applied
- **Applied**: Successfully applied
- **Failed**: Application failed
- **Rolled Back**: Previously applied but rolled back

Safety Features
---------------

Destructive Operations
^^^^^^^^^^^^^^^^^^^^^^

The system protects against destructive operations:

.. code-block:: python

    class Migration_004_remove_old_field(Migration):
        version = "1.3.0"
        is_destructive = True  # Mark as destructive
        
        def get_schema_changes(self) -> List[Tuple[str, str, MetadataField]]:
            # This will require explicit confirmation
            return [
                ('remove', 'deprecated_field', None)
            ]
        
        def get_rollback_changes(self) -> List[Tuple[str, str, MetadataField]]:
            return [
                ('add', 'deprecated_field', MetadataField(
                    type=MetadataFieldType.TEXT,
                    indexed=False
                ))
            ]

Automatic Backups
^^^^^^^^^^^^^^^^^

When ``backup_before_migration`` is enabled, the system:

1. Creates a full backup before applying migrations
2. Tags the backup with migration information
3. Provides restore instructions if migration fails

Confirmation Prompts
^^^^^^^^^^^^^^^^^^^^

For destructive operations, the system prompts for confirmation:

.. code-block:: bash

    $ lvdb migrate apply mydatabase --to-version 1.3.0
    
    WARNING: Migration 1.3.0 contains destructive operations:
    - Remove metadata field 'deprecated_field'
    
    This operation will permanently remove the field and its data.
    Continue? [y/N]: 

Rollback System
---------------

How Rollbacks Work
^^^^^^^^^^^^^^^^^^

Each migration must implement rollback logic via ``get_rollback_changes()``:

.. code-block:: python

    def get_rollback_changes(self) -> List[Tuple[str, str, MetadataField]]:
        """Define rollback changes for the migration."""
        # Return changes that reverse the forward migration
        return [
            ('remove', 'priority', None),  # Remove added field
            ('add', 'old_score', MetadataField(  # Restore removed field
                type=MetadataFieldType.INTEGER,
                indexed=False
            ))
        ]

Rollback Limitations
^^^^^^^^^^^^^^^^^^^^

- Data loss: Rollbacks may lose data added after migration
- Irreversible operations: Some changes cannot be undone
- Dependency chains: Must rollback in reverse dependency order

Safe Rollback Practices
^^^^^^^^^^^^^^^^^^^^^^^

1. **Test rollbacks** in development environment
2. **Backup before rollback** to preserve recent data
3. **Validate data integrity** after rollback
4. **Consider forward fixes** instead of rollbacks

Advanced Features
-----------------

Custom Migration Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can implement custom migration discovery:

.. code-block:: python

    from localvectordb.migration import MigrationEngine

    class CustomMigrationEngine(MigrationEngine):
        def discover_migrations(self) -> List[Migration]:
            # Custom logic to find migrations
            # Could load from database, API, etc.
            return super().discover_migrations()

Migration Validation
^^^^^^^^^^^^^^^^^^^^

Add validation to ensure migrations can be safely applied:

.. code-block:: python

    class Migration_005_validated(Migration):
        def validate_forward(self, current_schema: Dict[str, MetadataField]) -> bool:
            """Validate that forward migration can be applied."""
            # Check that required fields exist
            if 'existing_field' not in current_schema:
                raise ValidationError("Required field 'existing_field' not found")
            return True
        
        def validate_rollback(self, current_schema: Dict[str, MetadataField]) -> bool:
            """Validate that rollback can be applied."""
            # Check that fields to be restored don't conflict
            if 'conflicting_field' in current_schema:
                raise ValidationError("Cannot rollback: conflicting field exists")
            return True

Data Migration Support
^^^^^^^^^^^^^^^^^^^^^^

Migrations can include data transformation logic:

.. code-block:: python

    class Migration_006_with_data_migration(Migration):
        """Migration that includes data transformation."""
        
        def get_data_migration(self) -> Dict[str, callable]:
            """Return data transformation functions for each field."""
            return {
                'priority': self.transform_priority_data,
                'category': self.transform_category_data
            }
        
        def transform_priority_data(self, old_value: Any) -> Any:
            """Transform priority field data."""
            # Convert string priority to integer
            priority_map = {'low': 1, 'medium': 2, 'high': 3}
            return priority_map.get(old_value, 0)
        
        def transform_category_data(self, old_value: Any) -> Any:
            """Transform category field data."""
            # Normalize category names
            return old_value.lower().strip() if old_value else 'general'

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Migration Fails with "Version conflict"**

Another process may have applied migrations. Solutions:

- Refresh migration status
- Check for concurrent migration processes
- Resolve version conflicts manually

**Rollback Fails with "Cannot reverse operation"**

Some operations cannot be automatically reversed. Solutions:

- Restore from backup
- Create a forward migration to fix issues
- Manually fix the database state

**Migration hangs or times out**

Large data migrations may take time. Solutions:

- Increase database timeout settings
- Break large migrations into smaller chunks
- Run migrations during maintenance windows

Error Messages
^^^^^^^^^^^^^^

.. list-table:: Common Error Messages
   :header-rows: 1
   :widths: 30 70

   * - Error
     - Solution
   * - ``MigrationNotFoundError``
     - Check migration file exists and is properly named
   * - ``VersionConflictError``
     - Resolve version conflicts or refresh status
   * - ``DependencyError``
     - Check migration dependencies are satisfied
   * - ``DestructiveOperationError`` 
     - Use ``--force`` or enable destructive metadata migrations
   * - ``RollbackError``
     - Restore from backup or create forward fix

Best Practices
--------------

Migration Development
^^^^^^^^^^^^^^^^^^^^^

1. **Test thoroughly** in development environment
2. **Keep migrations small** and focused
3. **Include rollback logic** for all migrations  
4. **Document complex migrations** with clear descriptions
5. **Avoid destructive operations** when possible

Version Management
^^^^^^^^^^^^^^^^^^

- Use semantic versioning consistently
- Plan version increments in advance
- Consider backward compatibility
- Document breaking changes

Production Deployment
^^^^^^^^^^^^^^^^^^^^^

1. **Backup before migrations** in production
2. **Test migration path** in staging environment
3. **Plan for rollback** if issues occur
4. **Monitor performance** during migrations
5. **Communicate downtime** to users

API Reference
-------------

Migration Base Class
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from localvectordb.core import MetadataField
    from typing import List, Tuple, Dict, Any, Optional
    from abc import ABC, abstractmethod

    class Migration(ABC):
        """Base class for metadata schema migrations."""
        
        version: str  # Semantic version (e.g., "1.2.0")
        description: str  # Human-readable description
        dependencies: List[str] = []  # Required versions
        is_destructive: bool = False  # Breaking changes flag
        
        @abstractmethod
        def get_schema_changes(self) -> List[Tuple[str, str, MetadataField]]:
            """Return list of schema changes: (operation, field_name, field_def)."""
            pass
        
        @abstractmethod  
        def get_rollback_changes(self) -> List[Tuple[str, str, MetadataField]]:
            """Return rollback changes to reverse this migration."""
            pass
        
        def get_data_migration(self) -> Dict[str, callable]:
            """Return data transformation functions (optional)."""
            return {}
        
        def validate_forward(self, current_schema: Dict[str, MetadataField]) -> bool:
            """Validate forward migration can be applied."""
            return True
        
        def validate_rollback(self, current_schema: Dict[str, MetadataField]) -> bool:
            """Validate rollback can be applied."""
            return True

MigrationEngine
^^^^^^^^^^^^^^^

.. code-block:: python

    from localvectordb.core import DatabaseSchema, MetadataField
    from localvectordb.versioning import VersionManager
    from typing import List, Dict, Any, Optional

    class MigrationEngine:
        """Core migration engine for metadata schema evolution."""
        
        def __init__(self, db_path: str, migration_dir: str = "./migrations"):
            """Initialize migration engine."""
        
        def get_pending_migrations(self, target_version: str) -> List[Migration]:
            """Get list of pending schema migrations."""
        
        def apply_migrations(self, target_version: str, 
                           backup: bool = True) -> List[str]:
            """Apply schema migrations to target version."""
        
        def rollback_to_version(self, target_version: str) -> List[str]:
            """Rollback schema to specific version."""
        
        def get_migration_status(self) -> Dict[str, Any]:
            """Get current schema migration status."""
        
        def validate_migration_chain(self, migrations: List[Migration]) -> bool:
            """Validate that migration chain is consistent."""
        
        def apply_schema_changes(self, changes: List[Tuple[str, str, MetadataField]], 
                               db_schema: DatabaseSchema) -> None:
            """Apply schema changes using DatabaseSchema functionality."""

VersionManager
^^^^^^^^^^^^^^

.. code-block:: python

    class VersionManager:
        """Manages database versioning."""
        
        def __init__(self, db_path: str):
            """Initialize version manager."""
        
        def get_current_version(self) -> DatabaseVersion:
            """Get current database version."""
        
        def set_version(self, version: DatabaseVersion) -> None:
            """Set database version."""
        
        def needs_migration(self, target_version: str) -> bool:
            """Check if migration is needed."""

See Also
--------

- :ref:`backup` - Backup and recovery system
- :ref:`configuration` - Configuration options  
- :ref:`cli` - Command-line interface
- :ref:`api` - Python API reference