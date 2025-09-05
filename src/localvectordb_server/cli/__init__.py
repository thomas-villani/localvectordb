# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/cli/__init__.py
"""
LocalVectorDB Server Command-Line Interface v1.0

This module provides a comprehensive command-line interface for managing and interacting
with LocalVectorDB v1.0 vector databases. It includes commands for starting the server,
managing configuration, and performing database operations.

Main Components:

    - serve: Start the LocalVectorDB server
    - config: View and modify server configuration
    - list: List available databases
    - create: Create a new vector database
    - delete: Delete an existing database
    - rename: Rename a database
    - db: Commands for interacting with a specific database

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

    Find similar documents::

        $ lvdb db mydatabase knn doc_1 --k 5

    Manage database interactively::

        $ lvdb db mydatabase shell

Notes:

    - Database configuration is stored in a configuration file (default: .lvdb-config.toml)
    - The location of the configuration file can be specified with --config or the
      LVDB_SERVER_CONFIG environment variable
    - Database files are stored in the directory specified by DB_ROOT_DIR in the configuration
    - Authentication can be enabled with the auth commands
"""
import os

import click


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option('--config', '-c',
              type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
              help='Path to config file.',
              envvar='LVDB_SERVER_CONFIG')
@click.option('--db-folder', '-d', default=None,
              type=click.Path(dir_okay=True, exists=True, resolve_path=True, file_okay=False),
              help='The directory containing vector databases.',
              envvar='LVDB_DATABASE_ROOT_DIR')
@click.version_option(None, "-V", "--version", package_name="localvectordb", message="%(version)s")
@click.pass_context
def cli(ctx, config, db_folder):
    """LocalVectorDB Server command-line interface v1.0.

    Main entry point for the LocalVectorDB server CLI. Provides commands for
    managing and running the vector database server.
    """
    if ctx.obj is None:
        ctx.obj = {}
    elif not config:
        # This is to allow llmcli to pass the config path through the context when we embed it later.
        config = ctx.obj.get("lvdb_config_path")

    # TODO: THE FOLLOWING HAS NOT BEEN TESTED
    # TODO: if it works, remove it from the other commands
    from localvectordb_server.cli._utils import find_config_file
    config_path = find_config_file(config)

    if not config_path and ctx.invoked_subcommand != "config":
        click.secho("No configuration file found. Create one with 'lvdb config init'", fg="bright_red", err=True)
        raise click.exceptions.Exit(1)

    from localvectordb_server.config import load_config
    cfg = load_config(config_path)
    api_key_path = cfg.server.key_database_path or os.path.join(cfg.database.root_dir, "api_keys.db")

    if not db_folder:
        db_folder = cfg.database.root_dir

    ctx.obj = {'config': cfg, 'config_path': config_path, 'api_key_db_path': api_key_path, 'db_folder': db_folder}


from localvectordb_server.cli._auth import auth
from localvectordb_server.cli._basic import create_vector_database, delete_database, list_databases, serve
from localvectordb_server.cli._config import config_group
from localvectordb_server.cli._db import db_group

cli.add_command(serve)
cli.add_command(create_vector_database)
cli.add_command(list_databases)
cli.add_command(delete_database)
cli.add_command(db_group)
cli.add_command(config_group)
cli.add_command(auth)

__all__ = ["cli"]

if __name__ == '__main__':
    cli()
