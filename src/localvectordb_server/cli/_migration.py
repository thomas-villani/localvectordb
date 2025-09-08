# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/cli/_migration.py

"""Metadata schema migration CLI commands for LocalVectorDB.

Provides command-line interface for metadata schema migration operations including:
- Viewing migration status
- Applying pending schema migrations 
- Rolling back to previous schema versions
- Creating new migration templates for schema changes
- Managing migration dependencies

These commands focus on evolving the metadata schema using LocalVectorDB's 
built-in DatabaseSchema functionality rather than raw SQL operations.
"""

import os
from datetime import datetime
from pathlib import Path

import click

from localvectordb.backup import BackupManager, BackupConfig
from localvectordb.migration import MigrationEngine
from localvectordb.versioning import DatabaseVersion
from localvectordb_server.cli._utils import (
    EXIT_CODE_ERROR, format_table, print_json_output
)


@click.group('migrate')
@click.pass_context
def migrate_group(ctx):
    """
    Metadata schema migration and evolution commands.
    
    Provides comprehensive metadata schema migration functionality including 
    adding/removing/modifying metadata fields, version management, and rollback 
    capabilities for LocalVectorDB databases.
    
    \b
    Examples:
        \b
        lvdb migrate status mydb
        lvdb migrate apply mydb --to-version 1.2.0
        lvdb migrate rollback mydb --to-version 1.1.0
        lvdb migrate create "add user fields" --version 1.3.0
    """
    pass


@migrate_group.command('status')
@click.argument('database_name')
@click.option('--migrations-dir', '-m',
              type=click.Path(exists=True, file_okay=False),
              help='Directory containing migration files (default: ./migrations)')
@click.option('--json', 'output_json', is_flag=True,
              help='Output status in JSON format')
@click.pass_context
def migration_status(ctx, database_name, migrations_dir, output_json):
    """
    Show migration status for a database.
    
    Displays current database version, available migrations, applied migrations,
    and pending migrations that need to be applied.
    
    \b
    Examples:
        \b
        lvdb migrate status mydb
        lvdb migrate status mydb --migrations-dir ./custom_migrations --json
    """
    
    db_folder = ctx.obj.get("db_folder")
    if not db_folder:
        click.secho("Database folder not specified", fg="red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)
    
    try:
        # Set up database paths
        db_path = Path(db_folder) / f"{database_name}.sqlite"
        
        if not db_path.exists():
            click.secho(f"Database '{database_name}' not found in {db_folder}", 
                       fg="red", err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)
        
        # Set up migrations directory
        if not migrations_dir:
            migrations_dir = "./migrations"
        migrations_dir = Path(migrations_dir)
        
        # Create migration engine
        migration_engine = MigrationEngine(
            database_path=db_path,
            migrations_directory=migrations_dir
        )
        
        # Get migration status
        status = migration_engine.get_migration_status()
        
        if output_json:
            print_json_output(status)
        
        else:
            click.secho(f"Migration Status for '{database_name}'", fg="blue", bold=True)
            click.echo()
            
            click.secho("Current State:", fg="cyan")
            click.echo(f"  Database Version: {status['current_version']}")
            click.echo(f"  Latest Available: {status['latest_available_version'] or 'None'}")
            click.echo(f"  Applied Migrations: {status['applied_migrations_count']}")
            click.echo(f"  Pending Migrations: {status['pending_migrations_count']}")
            click.echo()
            
            # Show applied migrations
            if status['applied_migrations']:
                click.secho("Applied Migrations:", fg="green")
                for migration in status['applied_migrations'][-5:]:  # Show last 5
                    applied_at = datetime.fromisoformat(migration['applied_at'])
                    click.echo(f"  ✓ {migration['version']} - {applied_at.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if len(status['applied_migrations']) > 5:
                    click.echo(f"  ... and {len(status['applied_migrations']) - 5} more")
                click.echo()
            
            # Show pending migrations
            if status['pending_migrations']:
                click.secho("Pending Migrations:", fg="yellow")
                for version in status['pending_migrations']:
                    click.echo(f"  ○ {version}")
                click.echo()
            
            # Show overall status
            if status['pending_migrations_count'] == 0:
                click.secho("✓ Database is up to date", fg="green")
            else:
                click.secho(f"⚠ {status['pending_migrations_count']} migration(s) pending", fg="yellow")
    
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'database': database_name
        }
        
        if output_json:
            print_json_output(result)
        else:
            click.secho(f"✗ Error getting migration status: {e}", fg="red", err=True)
        
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@migrate_group.command('apply')
@click.argument('database_name')
@click.option('--to-version', '-v',
              help='Target version to migrate to (default: latest)')
@click.option('--migrations-dir', '-m',
              type=click.Path(exists=True, file_okay=False),
              help='Directory containing migration files (default: ./migrations)')
@click.option('--backup/--no-backup', default=True,
              help='Create backup before migration (default: enabled)')
@click.option('--backup-location', '-b',
              type=click.Path(exists=False, file_okay=False),
              help='Backup storage location (default: ./backups)')
@click.option('--dry-run', is_flag=True,
              help='Validate migrations without applying them')
@click.option('--json', 'output_json', is_flag=True,
              help='Output result in JSON format')
@click.pass_context
def apply_migrations(ctx, database_name, to_version, migrations_dir, backup, 
                     backup_location, dry_run, output_json):
    """
    Apply pending migrations to a database.
    
    Applies all pending migrations up to the specified target version.
    If no target version is specified, applies all pending migrations.
    Creates a backup before migration unless disabled.
    
    \b
    Examples:
        \b
        lvdb migrate apply mydb
        lvdb migrate apply mydb --to-version 1.2.0
        lvdb migrate apply mydb --dry-run --no-backup
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
        
        # Set up migrations directory
        if not migrations_dir:
            migrations_dir = "./migrations"
        migrations_dir = Path(migrations_dir)
        
        # Create backup manager if needed
        backup_manager = None
        if backup:
            backup_config = BackupConfig(
                backup_location=Path(backup_location) if backup_location else Path("./backups")
            )
            backup_manager = BackupManager(db_path, faiss_path, backup_config)
        
        # Create migration engine
        migration_engine = MigrationEngine(
            database_path=db_path,
            migrations_directory=migrations_dir,
            backup_manager=backup_manager,
            auto_backup=backup
        )
        
        # Apply migrations
        result = migration_engine.migrate(
            target_version=to_version,
            dry_run=dry_run,
            create_backup=backup
        )
        
        if output_json:
            print_json_output(result)
        
        else:
            if result['success']:
                if dry_run:
                    click.secho(f"✓ Migration validation passed", fg="green")
                    if 'pending_migrations' in result:
                        click.echo(f"  Would apply {len(result['pending_migrations'])} migration(s):")
                        for version in result['pending_migrations']:
                            click.echo(f"    - {version}")
                else:
                    click.secho(f"✓ Migration completed successfully", fg="green")
                    click.echo(f"  Applied {len(result.get('applied_migrations', []))} migration(s)")
                    
                    if result.get('backup_id'):
                        click.echo(f"  Backup created: {result['backup_id'][:8]}")
                    
                    if result.get('applied_migrations'):
                        for version in result['applied_migrations']:
                            click.echo(f"    ✓ {version}")
            
            else:
                click.secho(f"✗ Migration failed", fg="red")
                if result.get('error'):
                    click.echo(f"  Error: {result['error']}")
                
                if result.get('migration_errors'):
                    click.echo("  Migration errors:")
                    for error in result['migration_errors']:
                        click.secho(f"    - {error}", fg="red")
                
                if result.get('backup_id'):
                    click.echo(f"  Backup available for rollback: {result['backup_id'][:8]}")
    
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'database': database_name
        }
        
        if output_json:
            print_json_output(result)
        else:
            click.secho(f"✗ Migration error: {e}", fg="red", err=True)
        
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@migrate_group.command('rollback')
@click.argument('database_name')
@click.argument('target_version')
@click.option('--migrations-dir', '-m',
              type=click.Path(exists=True, file_okay=False),
              help='Directory containing migration files (default: ./migrations)')
@click.option('--backup/--no-backup', default=True,
              help='Create backup before rollback (default: enabled)')
@click.option('--backup-location', '-b',
              type=click.Path(exists=False, file_okay=False),
              help='Backup storage location (default: ./backups)')
@click.option('--dry-run', is_flag=True,
              help='Validate rollback without applying it')
@click.option('--json', 'output_json', is_flag=True,
              help='Output result in JSON format')
@click.pass_context
def rollback_migrations(ctx, database_name, target_version, migrations_dir, 
                        backup, backup_location, dry_run, output_json):
    """
    Rollback database to a previous version.
    
    Rolls back applied migrations to reach the specified target version.
    This will undo schema changes and data transformations made by newer migrations.
    
    \b
    Examples:
        \b
        lvdb migrate rollback mydb 1.1.0
        lvdb migrate rollback mydb 1.0.0 --dry-run --no-backup
    """
    
    db_folder = ctx.obj.get("db_folder")
    if not db_folder:
        click.secho("Database folder not specified", fg="red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)
    
    try:
        # Validate target version format
        try:
            DatabaseVersion(target_version)
        except ValueError as e:
            click.secho(f"✗ Invalid target version format: {target_version}", fg="red", err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)
        
        # Set up database paths
        db_path = Path(db_folder) / f"{database_name}.sqlite"
        faiss_path = Path(db_folder) / f"{database_name}.faiss"
        
        if not db_path.exists():
            click.secho(f"Database '{database_name}' not found in {db_folder}", 
                       fg="red", err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)
        
        # Set up migrations directory
        if not migrations_dir:
            migrations_dir = "./migrations"
        migrations_dir = Path(migrations_dir)
        
        # Create backup manager if needed
        backup_manager = None
        if backup:
            backup_config = BackupConfig(
                backup_location=Path(backup_location) if backup_location else Path("./backups")
            )
            backup_manager = BackupManager(db_path, faiss_path, backup_config)
        
        # Create migration engine
        migration_engine = MigrationEngine(
            database_path=db_path,
            migrations_directory=migrations_dir,
            backup_manager=backup_manager,
            auto_backup=backup
        )
        
        # Perform rollback
        result = migration_engine.rollback(
            target_version=target_version,
            dry_run=dry_run,
            create_backup=backup
        )
        
        if output_json:
            print_json_output(result)
        
        else:
            if result['success']:
                if dry_run:
                    click.secho(f"✓ Rollback validation passed", fg="green")
                    if 'migrations_to_rollback' in result:
                        click.echo(f"  Would rollback {len(result['migrations_to_rollback'])} migration(s)")
                        for version in result['migrations_to_rollback']:
                            click.echo(f"    - {version}")
                else:
                    click.secho(f"✓ Rollback completed successfully", fg="green")
                    click.echo(f"  Rolled back {len(result.get('rolled_back_migrations', []))} migration(s)")
                    click.echo(f"  Database version: {target_version}")
                    
                    if result.get('backup_id'):
                        click.echo(f"  Backup created: {result['backup_id'][:8]}")
                    
                    if result.get('rolled_back_migrations'):
                        for version in result['rolled_back_migrations']:
                            click.echo(f"    ✓ {version}")
            
            else:
                click.secho(f"✗ Rollback failed", fg="red")
                if result.get('error'):
                    click.echo(f"  Error: {result['error']}")
                
                if result.get('rollback_errors'):
                    click.echo("  Rollback errors:")
                    for error in result['rollback_errors']:
                        click.secho(f"    - {error}", fg="red")
                
                if result.get('backup_id'):
                    click.echo(f"  Backup available: {result['backup_id'][:8]}")
    
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'database': database_name,
            'target_version': target_version
        }
        
        if output_json:
            print_json_output(result)
        else:
            click.secho(f"✗ Rollback error: {e}", fg="red", err=True)
        
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@migrate_group.command('create')
@click.argument('description')
@click.option('--version', '-v', required=True,
              help='Version number for the migration (e.g., 1.2.0)')
@click.option('--migrations-dir', '-m',
              type=click.Path(exists=False, file_okay=False),
              help='Directory to create migration in (default: ./migrations)')
@click.option('--template', '-t',
              type=click.Choice(['basic', 'schema', 'data']),
              default='basic',
              help='Type of migration template to create')
@click.option('--json', 'output_json', is_flag=True,
              help='Output result in JSON format')
def create_migration(description, version, migrations_dir, template, output_json):
    """
    Create a new migration template file.
    
    Creates a new migration file with the specified version and description.
    The template type determines the structure of the generated migration.
    
    \b
    Template types:
        basic: Minimal migration template with empty up/down methods
        schema: Template for database schema changes (tables, indexes)
        data: Template for data transformations and migrations
    
    \b
    Examples:
        \b
        lvdb migrate create "add user table" --version 1.2.0 --template schema
        lvdb migrate create "migrate old data format" --version 1.2.1 --template data
    """
    
    try:
        # Validate version format
        try:
            DatabaseVersion(version)
        except ValueError as e:
            error = f"Invalid version format: {version}. Use semantic versioning (e.g., '1.2.0')"
            if output_json:
                print_json_output({'success': False, 'error': error})
            else:
                click.secho(f"✗ {error}", fg="red", err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)
        
        # Set up migrations directory
        if not migrations_dir:
            migrations_dir = "./migrations"
        migrations_dir = Path(migrations_dir)
        
        # Create dummy migration engine to access template functionality
        # We use a dummy path since we're just creating templates
        dummy_db_path = Path("/tmp/dummy.sqlite")
        migration_engine = MigrationEngine(
            database_path=dummy_db_path,
            migrations_directory=migrations_dir
        )
        
        # Create migration template
        migration_file = migration_engine.create_migration_template(
            version=version,
            description=description,
            template_type=template
        )
        
        result = {
            'success': True,
            'migration_file': str(migration_file),
            'version': version,
            'description': description,
            'template_type': template,
            'migrations_directory': str(migrations_dir)
        }
        
        if output_json:
            print_json_output(result)
        else:
            click.secho(f"✓ Migration template created successfully", fg="green")
            click.echo(f"  File: {migration_file}")
            click.echo(f"  Version: {version}")
            click.echo(f"  Description: {description}")
            click.echo(f"  Template: {template}")
            click.echo()
            click.secho("Next steps:", fg="cyan")
            click.echo(f"  1. Edit {migration_file} to implement your migration")
            click.echo(f"  2. Test the migration with: lvdb migrate apply <database> --dry-run")
            click.echo(f"  3. Apply the migration with: lvdb migrate apply <database>")
    
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'version': version,
            'description': description
        }
        
        if output_json:
            print_json_output(result)
        else:
            click.secho(f"✗ Migration creation failed: {e}", fg="red", err=True)
        
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@migrate_group.command('list')
@click.option('--migrations-dir', '-m',
              type=click.Path(exists=True, file_okay=False),
              help='Directory containing migration files (default: ./migrations)')
@click.option('--show-dependencies', '-d', is_flag=True,
              help='Show migration dependencies')
@click.option('--json', 'output_json', is_flag=True,
              help='Output in JSON format')
def list_migrations(migrations_dir, show_dependencies, output_json):
    """
    List available migration files.
    
    Shows all discovered migration files with their versions, descriptions,
    and optionally their dependency relationships.
    
    \b
    Examples:
        \b
        lvdb migrate list
        lvdb migrate list --show-dependencies --json
    """
    
    try:
        # Set up migrations directory
        if not migrations_dir:
            migrations_dir = "./migrations"
        migrations_dir = Path(migrations_dir)
        
        if not migrations_dir.exists():
            if output_json:
                print_json_output({'migrations': [], 'message': 'Migrations directory not found'})
            else:
                click.echo(f"Migrations directory not found: {migrations_dir}")
            return
        
        # Create dummy migration engine to discover migrations
        dummy_db_path = Path("/tmp/dummy.sqlite")
        migration_engine = MigrationEngine(
            database_path=dummy_db_path,
            migrations_directory=migrations_dir
        )
        
        # Discover migrations
        migrations = migration_engine.discover_migrations()
        migration_order = migration_engine.get_migration_order()
        
        if output_json:
            migration_list = []
            for version in migration_order:
                migration_script = migrations[version]
                migration_list.append({
                    'version': version,
                    'description': migration_script.description,
                    'dependencies': migration_script.dependencies,
                    'file_path': str(migration_script.file_path),
                    'checksum': migration_script.checksum
                })
            
            print_json_output({
                'migrations': migration_list,
                'total_count': len(migration_list),
                'migrations_directory': str(migrations_dir)
            })
        
        else:
            if not migrations:
                click.echo(f"No migrations found in {migrations_dir}")
                return
            
            click.secho(f"Available Migrations in {migrations_dir}:", fg="blue", bold=True)
            click.echo()
            
            # Format as table
            headers = ['Version', 'Description', 'File']
            if show_dependencies:
                headers.append('Dependencies')
            
            rows = []
            for version in migration_order:
                migration_script = migrations[version]
                row = [
                    version,
                    migration_script.description[:50] + ("..." if len(migration_script.description) > 50 else ""),
                    migration_script.file_path.name
                ]
                
                if show_dependencies:
                    deps = ', '.join(migration_script.dependencies) if migration_script.dependencies else '-'
                    row.append(deps)
                
                rows.append(row)
            
            click.echo(format_table(headers, rows))
            click.echo()
            click.echo(f"Total: {len(migrations)} migration(s)")
    
    except Exception as e:
        if output_json:
            print_json_output({'error': str(e), 'migrations': []})
        else:
            click.secho(f"✗ Error listing migrations: {e}", fg="red", err=True)
        
        raise click.exceptions.Exit(EXIT_CODE_ERROR)