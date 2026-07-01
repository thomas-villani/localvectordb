"""
LocalVectorDB Server Command-Line Interface

This module provides a comprehensive command-line interface for managing and interacting
with LocalVectorDB vector databases. It includes commands for starting the server,
managing configuration, and performing database operations.

Main Components:

    - serve: Start the LocalVectorDB server
    - config: View and modify server configuration
    - list: List available databases
    - create: Create a new vector database
    - delete: Delete an existing database
    - rename: Rename a database
    - db: Commands for interacting with a specific database
    - backup: Backup and restore operations (create, list, restore, verify, cleanup, pitr)
    - migrate: Database migration and schema evolution (status, apply, rollback, create, list)

Examples:

    Start the server::

        $ lvdb serve --host 0.0.0.0 --port 5000

    Initialize configuration::

        $ lvdb config init --format toml

    View configuration::

        $ lvdb config show

    List available databases::

        $ lvdb list

    Create a database::

        $ lvdb create mydatabase --embedding-model nomic-embed-text --chunk-size 500

    Add documents to a database::

        $ lvdb db mydatabase add document.txt
        $ lvdb db mydatabase add "documents/*.py"
        $ cat document.txt | lvdb db mydatabase add -

    Search documents::

        $ lvdb db mydatabase search "query text" --limit 5
        $ lvdb db mydatabase search "query text" --search-type hybrid --metadata-filter '{"author":"Smith"}'

    Get document by ID::

        $ lvdb db mydatabase get doc_1

    Manage database interactively::

        $ lvdb db mydatabase shell

    Create and manage backups::

        $ lvdb backup create mydatabase --type full
        $ lvdb backup list --database mydatabase
        $ lvdb backup restore backup-id --to-location ./restored
        $ lvdb backup pitr "2024-01-15 14:30:00" --to-location ./pitr-restored

    Manage database migrations::

        $ lvdb migrate status mydatabase
        $ lvdb migrate create "add new field" --version 1.2.0
        $ lvdb migrate apply mydatabase --to-version 1.2.0
        $ lvdb migrate rollback mydatabase 1.1.0

Notes:

    - Database configuration is stored in a configuration file (default: .lvdb-config.toml)
    - The location of the configuration file can be specified with --config or the
      LVDB_SERVER_CONFIG environment variable
    - Database files are stored in the directory specified by DB_ROOT_DIR in the configuration
    - Authentication can be enabled with the auth commands
"""

import logging
import os

import click


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--config",
    "-c",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
    help="Path to config file.",
    envvar="LVDB_SERVER_CONFIG",
)
@click.option(
    "--db-folder",
    "-d",
    default=None,
    type=click.Path(dir_okay=True, exists=True, resolve_path=True, file_okay=False),
    help="The directory containing vector databases.",
    envvar="LVDB_DATABASE_ROOT_DIR",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose (DEBUG) logging.")
@click.option("--quiet", "-q", is_flag=True, help="Only log errors (suppress warnings/info).")
@click.version_option(None, "-V", "--version", package_name="localvectordb", message="%(version)s")
@click.pass_context
def cli(ctx, config, db_folder, verbose, quiet):
    """LocalVectorDB Server command-line interface.

    Main entry point for the LocalVectorDB server CLI. Provides commands for
    managing and running the vector database server.
    """
    # --verbose/--quiet control library logging verbosity (most chatty work
    # happens in the library, which logs via the standard logging module).
    log_level = logging.DEBUG if verbose else (logging.ERROR if quiet else logging.WARNING)
    logging.basicConfig(level=log_level)

    if ctx.obj is None:
        ctx.obj = {}
    elif not config:
        # This is to allow llmcli to pass the config path through the context when we embed it later.
        config = ctx.obj.get("lvdb_config_path")

    # Centralized config resolution — subcommands inherit config from this group callback.
    from localvectordb_server.cli._utils import find_config_file

    config_path = find_config_file(config)

    if not config_path:
        if ctx.invoked_subcommand not in ("config", "mcp", "tuning", "maintenance", "version", "chunk"):
            click.secho("No configuration file found. Create one with 'lvdb config init'", fg="bright_red", err=True)
            raise click.exceptions.Exit(1)
        cfg = config_path = api_key_path = db_folder = None
    else:
        from localvectordb_server.config import load_config

        cfg = load_config(config_path)
        api_key_path = cfg.server.security.key_database_path or os.path.join(cfg.database.root_dir, "api_keys.db")

        if not db_folder:
            db_folder = cfg.database.root_dir
        else:
            cfg.database.root_dir = db_folder

    ctx.obj = {
        "config": cfg,
        "config_path": config_path,
        "api_key_db_path": api_key_path,
        "db_folder": db_folder,
        "verbose": verbose,
        "quiet": quiet,
    }


from localvectordb_server.cli._auth import auth  # noqa: E402
from localvectordb_server.cli._backup import backup_group  # noqa: E402
from localvectordb_server.cli._basic import (  # noqa: E402
    create_vector_database,
    delete_database,
    list_databases,
    rename_database,
    serve,
    version,
)
from localvectordb_server.cli._chunk import chunk_command  # noqa: E402
from localvectordb_server.cli._config import config_group  # noqa: E402
from localvectordb_server.cli._db import db_group  # noqa: E402
from localvectordb_server.cli._mcp import mcp_commands  # noqa: E402
from localvectordb_server.cli._migration import migrate_group  # noqa: E402
from localvectordb_server.cli._tuning import maintenance_group, tuning_group  # noqa: E402

cli.add_command(serve)
cli.add_command(create_vector_database)
cli.add_command(list_databases)
cli.add_command(delete_database)
cli.add_command(rename_database)
cli.add_command(version)
cli.add_command(chunk_command)
cli.add_command(db_group)
cli.add_command(config_group)
cli.add_command(auth)
cli.add_command(backup_group)
cli.add_command(migrate_group)
cli.add_command(mcp_commands)
cli.add_command(tuning_group)
cli.add_command(maintenance_group)

__all__ = ["cli"]

if __name__ == "__main__":
    cli()
