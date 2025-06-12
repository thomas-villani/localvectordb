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

import click

@click.group()
def cli():
    """LocalVectorDB Server command-line interface v1.0.

    Main entry point for the LocalVectorDB server CLI. Provides commands for
    managing and running the vector database server.
    """
    pass

from localvectordb_server.cli._basic import serve, create_vector_database, list_databases, delete_database
from localvectordb_server.cli._db import db_group
from localvectordb_server.cli._config import config_group
from localvectordb_server.cli._auth import auth

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