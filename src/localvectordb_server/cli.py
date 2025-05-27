# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb_server/cli.py
"""
LocalVectorDB Server Command-Line Interface v2.0

This module provides a comprehensive command-line interface for managing and interacting
with LocalVectorDB v2.0 vector databases. It includes commands for starting the server,
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
        $ lvdb db mydatabase add "documents/*.pdf"
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

    - Database configuration is stored in a configuration file (default: server-cfg.toml)
    - The location of the configuration file can be specified with --config or the
      LVDB_SERVER_CONFIG environment variable
    - Database files are stored in the directory specified by DB_ROOT_DIR in the configuration
    - Authentication can be enabled with the auth commands
"""
import glob
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click

EXIT_CODE_SUCCESS = 0
EXIT_CODE_ERROR = 1
EXIT_CODE_CONFIGURATION_ERROR = 2
EXIT_CODE_OLLAMA_ERROR = 3
EXIT_CODE_PERMISSION_ERROR = 4


def find_config_file(config_path: str = None) -> str:
    """Find configuration file in order of precedence"""
    # First check explicit path
    if config_path and os.path.exists(config_path):
        return config_path

    # Check environment variable
    env_path = os.environ.get("LVDB_SERVER_CONFIG")
    if env_path and os.path.exists(env_path):
        return env_path

    # Check common locations
    default_locations = [
        "./server-cfg.toml",
        "./server-cfg.py",
        "./production.toml",
        "./instance/server-cfg.toml",
        "./instance/server-cfg.py",
        "./instance/production.toml",
        os.path.expanduser("~/localvectordb_server/server-cfg.toml")
    ]

    for path in default_locations:
        if os.path.exists(path):
            return path

    return None


def get_stdin_input(input_required=True, err_msg=None):
    err_msg = err_msg or "Error: No input data in stdin!"

    input_data_stream = click.get_text_stream('stdin')
    if input_data_stream.isatty():
        click.secho(err_msg, fg='bright_red', err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)
    data_from_stdin = input_data_stream.read().rstrip()
    if not data_from_stdin and input_required:
        click.secho(err_msg, fg='bright_red', err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    return data_from_stdin


def _print_db_stats(db: "LocalVectorDB"):
    """Print database statistics for v2.0"""
    try:
        stats = db.stats

        click.secho("Database Statistics:", fg="blue", bold=True)

        click.secho("\nGeneral:", fg="cyan")
        click.echo(f"  Name: {db.name}")
        click.echo(f"  Total documents: {stats['documents']:,}")
        click.echo(f"  Total chunks: {stats['chunks']:,}")
        if stats['documents'] > 0:
            click.echo(f"  Avg. chunks per document: {stats['chunks'] / stats['documents']:.2f}")

        click.secho("\nVector Information:", fg="cyan")
        click.echo(f"  Embedding model: {stats['embedding_model']}")
        click.echo(f"  Provider: {stats['embedding_provider']}")
        click.echo(f"  Vector dimension: {stats['embedding_dimension']}")
        click.echo(f"  Vector count: {stats['index_vectors']:,}")

        click.secho("\nConfiguration:", fg="cyan")
        click.echo(f"  Chunking method: {stats['chunking_method']}")
        click.echo(f"  Chunk size: {stats['chunk_size']}")
        click.echo(f"  Chunk overlap: {stats['chunk_overlap']}")
        click.echo(f"  FTS search: {'enabled' if stats['fts_enabled'] else 'disabled'}")

        # Show metadata schema if available
        if hasattr(db, 'metadata_schema') and db.metadata_schema:
            click.secho("\nMetadata Schema:", fg="cyan")
            for field_name, field_def in db.metadata_schema.items():
                indexed = " (indexed)" if field_def.indexed else ""
                required = " (required)" if field_def.required else ""
                click.echo(f"  {field_name}: {field_def.type.value}{indexed}{required}")

    except Exception as e:
        click.secho(f"Error retrieving statistics: {str(repr(e))}", fg="bright_red")
        # Fall back to basic stats
        click.secho("Basic Database Statistics:", fg="blue")
        click.echo(f"  Name: {db.name}")
        click.echo(f"  Embedding provider: {db.embedding_provider.provider_name}")
        click.echo(f"  Embedding model: {db.embedding_provider.model}")
        click.echo(f"  Vector dimension: {db.embedding_dimension}")


@click.group()
def cli():
    """LocalVectorDB Server command-line interface v2.0.

    Main entry point for the LocalVectorDB server CLI. Provides commands for
    managing and running the vector database server.
    """
    pass


@cli.command()
@click.option('--host', '-h', default=None, help='The interface to bind to (e.g. 127.0.0.1 for local serving).')
@click.option('--port', '-p', default=None, type=int, help='The port to bind to (default = 5000).')
@click.option('--debug', is_flag=True, help='Enable Flask debug mode.')
@click.option('--config', '-c', type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
              help='Path to config file.', envvar='LVDB_SERVER_CONFIG')
@click.option('--db-folder', '-d', default=None, type=click.Path(dir_okay=True, exists=True, resolve_path=True),
              help='The directory containing vector databases.')
@click.option(
    '--log-level', '-l', default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help='Set the logging level. Must be one of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"'
)
@click.option('--disable-ollama-check', '-x', is_flag=True, help='Disable checking for ollama on startup')
def serve(host, port, debug, config, db_folder, log_level, disable_ollama_check):
    """Start the LocalVectorDB server."""
    config_path = find_config_file(config)

    if config_path:
        click.secho(f"Loading configuration from `{config_path}`", fg='blue')
    else:
        click.secho('No configuration file found. Using default configuration.', fg='yellow')

    from localvectordb_server import create_app
    from localvectordb.exceptions import ConfigurationError

    try:
        app = create_app(
            configuration=config_path,
            database_directory=db_folder,
            debug=debug,
            log_level=log_level,
            host=host,
            port=port
        )

        # Get final configuration
        config = app.lvdb_config

        if not disable_ollama_check:
            from localvectordb_server._checkdeps import check_ollama_installation, check_ollama_service
            try:
                version = check_ollama_installation()
                click.echo(f"Found Ollama version: {version}")

                if check_ollama_service():
                    click.echo("Ollama service is running")
            except Exception as e:
                click.echo(f"Ollama check failed: {e}")
                raise click.exceptions.Exit(EXIT_CODE_OLLAMA_ERROR)

        # Run the Flask app with final config values
        app.run(
            host=host or config.server.host,
            port=port or config.server.port,
            debug=debug
        )

    except ConfigurationError as e:
        click.secho(f"Configuration error: {e}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_CONFIGURATION_ERROR)
    except Exception as e:
        click.secho(f"Error: {e}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@cli.group('config', invoke_without_command=True)
@click.option('--config', '-c',
              type=click.Path(file_okay=True, dir_okay=False, resolve_path=True),
              help='Path to config file.',
              envvar='LVDB_SERVER_CONFIG')
@click.pass_context
def config_group(ctx, config):
    """View or modify the server configuration."""
    config_path = find_config_file(config)
    if not config_path and ctx.invoked_subcommand != "init":
        click.secho("No configuration file found. Create one with 'lvdb config init'", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    from localvectordb_server.config import load_config
    # Load existing config
    cfg = load_config(config_path)
    ctx.obj = {'config': cfg, 'config_path': config_path}

    # If no subcommand was invoked, display current config
    if ctx.invoked_subcommand is None:
        ctx.invoke(show_config)


@config_group.command('show')
@click.option('--format', '-f', type=click.Choice(['toml', 'yaml', 'json', 'ini']), default=None,
              help='Output format (defaults to format of config file)')
@click.option('--toml', 'format', flag_value='toml', help="Output in `toml` format")
@click.option('--yaml', 'format', flag_value='yaml', help="Output in `yaml` format")
@click.option('--ini', 'format', flag_value='ini', help="Output in `ini` format")
@click.option('--json', 'format', flag_value='json', help="Output in `json` format")
@click.option('--section', '-s', type=click.Choice(['database', 'embedding', 'server', 'migration']), default=None,
              help='Only show specific section')
@click.pass_context
def show_config(ctx, format, section):
    """Display current configuration."""
    cfg = ctx.obj['config']
    config_path = ctx.obj['config_path']

    # Determine output format based on file extension if not specified
    if not format:
        suffix = Path(config_path).suffix.lower()
        if suffix == '.toml':
            format = 'toml'
        elif suffix in ['.yaml', '.yml']:
            format = 'yaml'
        elif suffix == '.json':
            format = 'json'
        elif suffix in ['.ini', '.cfg']:
            format = 'ini'
        else:
            format = 'toml'  # Default to TOML for unknown formats

    # Generate configuration string
    if format == 'toml':
        config_str = cfg.generate_toml()
    else:
        # For other formats, convert to dict and handle appropriately
        config_dict = {
            'database': asdict(cfg.database),
            'embedding': asdict(cfg.embedding),
            'server': asdict(cfg.server),
            'migration': asdict(cfg.migration)
        }

        if format == 'json':
            config_str = json.dumps(config_dict, indent=2)
        else:
            config_str = cfg.generate_toml()  # Fallback to TOML

    # Filter by section if requested
    if section:
        if format == 'toml' or format == 'ini':
            section_header = f"[{section}]"
            lines = config_str.split('\n')
            section_start = -1
            section_end = len(lines)

            for i, line in enumerate(lines):
                if line.strip() == section_header:
                    section_start = i
                elif section_start >= 0 and line.strip().startswith('[') and i > section_start:
                    section_end = i
                    break

            if section_start >= 0:
                config_str = '\n'.join(lines[section_start:section_end])
            else:
                click.secho(f"Section '{section}' not found in configuration", fg="bright_red")
                return
        elif format == "json":
            json_obj = json.loads(config_str)
            if section in json_obj:
                json_obj = {section: json_obj[section]}
                config_str = json.dumps(json_obj, indent=2)
            else:
                click.secho(f"Section '{section}' not found in configuration", fg="bright_red")
                return

    # Display the configuration
    title = f"Configuration from: {config_path}"
    click.secho(title, fg="cyan")
    click.secho("=" * len(title), fg="cyan")
    click.echo(config_str)


@config_group.command('init')
@click.option('--format', '-f', type=click.Choice(['toml', 'yaml', 'ini', 'json']), default='toml',
              help='Configuration file format (default: toml)')
@click.option('--output', type=click.Path(resolve_path=True), help='Path to create config file')
@click.option('--schema', type=click.Choice(['documents', 'research_papers', 'code_repository', 'customer_support']),
              help='Apply a predefined metadata schema')
@click.pass_context
def init_config(ctx, format, output, schema):
    """Initialize a new configuration file with default settings."""
    if not output:
        output = f"./server-cfg.{format}"

    if os.path.exists(output):
        click.echo(f"Configuration file `{output}` exists! Overwrite (Y/n)?")
        char = click.getchar()
        if char.lower() != "y":
            return 0

    from localvectordb_server.config import Config

    # Create default configuration
    config = Config()

    # Apply common schema if requested
    if schema:
        config.apply_common_schema(schema)

    # Generate and save configuration
    config_text = config.generate_toml()
    with open(output, "w", encoding="utf-8") as f:
        f.write(config_text)

    click.secho(f"Configuration file `{output}` created!", fg="green")
    click.echo(f"To run the server with this configuration:\n")
    click.echo(f"   $ lvdb serve --config {output}\n")


@cli.command('list')
@click.option('--config', '-c',
              type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
              help='Path to config file.',
              envvar='LVDB_SERVER_CONFIG')
@click.option('--db-folder', '-d', default=None,
              type=click.Path(dir_okay=True, exists=True, resolve_path=True, file_okay=False),
              help='The directory containing vector databases.',
              envvar='LVDB_DATABASE_ROOT_DIR')
@click.option("--details", "-v", is_flag=True, default=False, help="Show details")
def list_databases(config, db_folder, details):
    """List databases"""
    if not db_folder:
        config_path = find_config_file(config)
        if not config_path:
            click.secho("No configuration file found and `--db-folder` not specified.", fg="bright_red", err=True)
            click.echo(
                "Use the `--db-folder` option or create a configuration file with `lvdb config init`, or "
                "specify the location of an existing config file using `--config <path-to-config>`", err=True
            )
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

        from localvectordb_server.config import load_config
        # Load config
        cfg = load_config(config_path)
        db_folder = cfg.database.root_dir

    if os.path.isdir(db_folder):
        click.secho(f"Databases in {os.path.abspath(db_folder)}", fg="blue", err=True)
        if details:
            click.secho(f"{'Name':<25}{'Documents':<10}{'Chunks':<10}{'Model':<25}{'Method':<20}", fg="cyan")
            click.secho("=" * 92, fg="cyan")

        for file in os.listdir(db_folder):
            if not file.lower().endswith(".sqlite"):
                continue
            name, _ = os.path.splitext(os.path.basename(file))
            if not details:
                click.echo(name)
            else:
                try:
                    from localvectordb.database import LocalVectorDB
                    db = LocalVectorDB(name, db_folder, create_if_not_exists=False)
                    stats = db.stats
                    click.echo(f"{name:<25}{stats['documents']:<10}{stats['chunks']:<10}"
                               f"{stats['embedding_model']:<25}{stats['chunking_method']:<20}")
                    db.close()
                except Exception as e:
                    click.echo(f"{name:<25}{'ERROR':<10}{'ERROR':<10}{'ERROR':<25}{'ERROR':<20}")

    else:
        click.secho(f"Database folder {os.path.abspath(db_folder)} not found!", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@cli.command('create')
@click.argument('name')
@click.option('--embedding-model', default=None, type=str, help='Embedding model to use')
@click.option('--embedding-provider', default=None, type=click.Choice(['ollama', 'openai']), help='Embedding provider')
@click.option('--chunk-size', default=None, type=int, help='Max tokens per chunk')
@click.option('--chunking-method', default=None,
              type=click.Choice(['sentences', 'tokens', 'characters', 'words', 'lines', 'sections']),
              help='Chunking method')
@click.option('--chunk-overlap', default=None, type=int, help='Overlap between chunks')
@click.option('--metadata-schema', default=None,
              type=click.Choice(['documents', 'research_papers', 'code_repository', 'customer_support']),
              help='Predefined metadata schema to use')
@click.option('--config', '-c',
              type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
              help='Path to config file.',
              envvar='LVDB_SERVER_CONFIG')
@click.option('--db-folder', '-d', default=None,
              type=click.Path(dir_okay=True, exists=True, resolve_path=True, file_okay=False),
              help='The directory containing vector databases.',
              envvar='LVDB_DATABASE_ROOT_DIR')
def create_vector_database(
        name, embedding_model, embedding_provider, chunk_size, chunking_method,
        chunk_overlap, metadata_schema, config, db_folder
        ):
    """Create a new vector database."""
    config_path = None
    cfg = None

    if not db_folder:
        config_path = find_config_file(config)
        from localvectordb_server.config import load_config
        cfg = load_config(config_path)
        db_folder = cfg.database.root_dir

        # Use config defaults if not specified
        embedding_model = embedding_model or cfg.embedding.model
        embedding_provider = embedding_provider or cfg.embedding.provider
        chunk_size = chunk_size or cfg.database.chunk_size
        chunking_method = chunking_method or cfg.database.chunking_method
        chunk_overlap = chunk_overlap or cfg.database.chunk_overlap

    if not db_folder:
        click.secho("No configuration found and `--db-folder` not specified.", fg="bright_red")
        click.echo(
            "Use the `--db-folder` option or create a configuration file with `lvdb config init`, or "
            "specify the location of an existing config file using `--config <path-to-config>`"
        )
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    os.makedirs(db_folder, exist_ok=True)

    # Set defaults
    embedding_model = embedding_model or "nomic-embed-text"
    embedding_provider = embedding_provider or "ollama"
    chunk_size = chunk_size or 500
    chunking_method = chunking_method or "sentences"
    chunk_overlap = chunk_overlap or 1

    # Prepare metadata schema
    schema_dict = None
    if metadata_schema:
        from localvectordb_server.config import Config
        temp_config = Config()
        temp_config.apply_common_schema(metadata_schema)
        schema_dict = temp_config.database.default_metadata_schema

    try:
        from localvectordb.database import LocalVectorDB

        db = LocalVectorDB(
            name=name,
            base_path=db_folder,
            metadata_schema=schema_dict,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunking_method=chunking_method,
            chunk_overlap=chunk_overlap,
        )
        db.close()

        click.secho(f"Created database '{name}' in {os.path.abspath(db_folder)}", fg="green")
        click.echo(f"   embedding_model: {embedding_model}")
        click.echo(f"   embedding_provider: {embedding_provider}")
        click.echo(f"   chunk_size: {chunk_size}")
        click.echo(f"   chunking_method: {chunking_method}")
        click.echo(f"   chunk_overlap: {chunk_overlap}")
        if metadata_schema:
            click.echo(f"   metadata_schema: {metadata_schema}")

    except Exception as e:
        click.secho(f"Error creating database: {str(repr(e))}", fg='bright_red', err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@cli.command('delete')
@click.argument('name')
@click.option('--config', '-c',
              type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
              help='Path to config file.',
              envvar='LVDB_SERVER_CONFIG')
@click.option('--db-folder', '-d', default=None,
              type=click.Path(dir_okay=True, exists=True, resolve_path=True, file_okay=False),
              help='The directory containing vector databases.',
              envvar='LVDB_DATABASE_ROOT_DIR')
@click.option('--confirm', '-y', flag_value=True, default=False, help='Pre-confirm deletion (danger!)')
def delete_database(name, config, db_folder, confirm):
    """Delete a database"""
    if not db_folder:
        config_path = find_config_file(config)
        from localvectordb_server.config import load_config
        cfg = load_config(config_path)
        db_folder = cfg.database.root_dir

    if not db_folder or not os.path.exists(db_folder):
        click.secho(
            f"DB_FOLDER {'not specified and not found in configuration' if not db_folder else 'does not exist'}.",
            fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    sqlite_file = os.path.abspath(os.path.join(db_folder, f"{name}.sqlite"))
    faiss_file = os.path.abspath(os.path.join(db_folder, f"{name}.faiss"))

    if os.path.exists(sqlite_file):
        files = [sqlite_file]
        if os.path.exists(faiss_file):
            files.append(faiss_file)
        if not confirm:
            confirm = click.prompt(
                click.style(f'Are you sure you want to delete the database "{name}"?', fg="bright_red") +
                f'\nThis will remove the following file(s):\n'
                f'{chr(10).join("- " + f for f in files)}\n' +
                click.style(f'Warning: this action cannot be undone!', fg="bright_red", bold=True) +
                f'\nEnter "confirm" to delete, anything else to exit.'
            )
            if confirm != "confirm":
                click.echo("Aborted by user!")
                return 0
        try:
            for f in files:
                os.remove(f)
                click.secho(f"- {f} deleted", fg="magenta")
            click.secho(f"Database '{name}' was deleted", fg="magenta")
        except Exception as e:
            click.secho(f"Error deleting database '{name}': {str(repr(e))}", fg="bright_red")
            raise click.exceptions.Exit(EXIT_CODE_ERROR) from e
    else:
        click.echo(f"Database {name} was not found in {os.path.abspath(db_folder)}! No action taken.")


@cli.group()
@click.option('--config', '-c',
              type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
              help='Path to config file.',
              envvar='LVDB_SERVER_CONFIG')
@click.pass_context
def auth(ctx, config):
    """Manage API authentication settings."""
    config_path = find_config_file(config)
    if not config_path:
        click.secho("No configuration file found. Create one with 'lvdb config init'", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    from localvectordb_server.config import load_config
    cfg = load_config(config_path)
    ctx.obj = {'config': cfg, 'config_path': config_path}


@auth.command('status')
@click.pass_context
def auth_status(ctx):
    """Show the current authentication status."""
    try:
        cfg = ctx.obj['config']
        config_path = ctx.obj['config_path']

        auth_enabled = cfg.server.require_api_key
        api_keys = cfg.server.authorized_api_keys

        click.echo(f"Configuration file: " + click.style(f"{config_path}", fg="blue"))
        click.echo(f"API Authentication: " + click.style(f"{'Enabled' if auth_enabled else 'Disabled'}",
                                                         fg="green" if auth_enabled else "red"))
        click.echo(f"API Keys configured: " + click.style(f"{len(api_keys)}", fg="blue"))

    except Exception as e:
        click.secho(f"Error reading auth status: {str(repr(e))}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@cli.group('db')
@click.argument("name")
@click.option('--config', '-c',
              type=click.Path(file_okay=True, dir_okay=False, exists=True, resolve_path=True),
              help='Path to config file.',
              envvar='LVDB_SERVER_CONFIG')
@click.option('--db-folder', '-d', default=None,
              type=click.Path(dir_okay=True, exists=True, resolve_path=True, file_okay=False),
              help='The directory containing vector databases.',
              envvar='LVDB_DATABASE_ROOT_DIR')
@click.pass_context
def db_group(ctx, name, config, db_folder):
    """Commands related to a specific database NAME."""
    if not db_folder:
        config_path = find_config_file(config)
        from localvectordb_server.config import load_config
        cfg = load_config(config_path)
        db_folder = cfg.database.root_dir

    if not db_folder or not os.path.exists(db_folder):
        click.secho(
            f"DB_FOLDER {'not specified and not found in configuration' if not db_folder else 'does not exist'}.",
            fg="bright_red", err=True
            )
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    try:
        from localvectordb.database import LocalVectorDB
        db = LocalVectorDB(name=name, base_path=db_folder, create_if_not_exists=False)
    except Exception as e:
        click.secho(f"Database '{name}' was not found in {os.path.abspath(db_folder)}!",
                    fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e

    ctx.obj = {"db_name": name, "db_folder": db_folder, "db": db}


@db_group.command('info')
@click.pass_context
def show_db_info(ctx):
    """Show the configuration info for a database"""
    db = ctx.obj["db"]

    try:
        stats = db.stats
        click.echo("Database Info\n"
                   "-------------")
        click.echo(f"  Database: {db.name}")
        click.echo(f"  Path: {os.path.abspath(ctx.obj['db_folder'])}")
        click.echo(f"  Embedding model: {stats['embedding_model']}")
        click.echo(f"  Embedding provider: {stats['embedding_provider']}")
        click.echo(f"  Chunk size: {stats['chunk_size']}")
        click.echo(f"  Chunking method: {stats['chunking_method']}")
        click.echo(f"  Chunk overlap: {stats['chunk_overlap']}")
        click.echo(f"  FTS search available: {stats['fts_enabled']}")
        click.echo(f"  Total Documents: {stats['documents']}")
        click.echo(f"  Total Chunks: {stats['chunks']}")

        # Show metadata schema if available
        if hasattr(db, 'metadata_schema') and db.metadata_schema:
            click.echo(f"  Metadata fields: {len(db.metadata_schema)}")
            for field_name in db.metadata_schema:
                click.echo(f"    - {field_name} {db.metadata_schema[field_name].type.upper()}")

    except Exception as e:
        click.secho(f"Error reading database info: {str(repr(e))}", fg='bright_red', err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@db_group.command('stats')
@click.pass_context
def show_db_stats(ctx):
    """Show database statistics"""
    db = ctx.obj["db"]
    _print_db_stats(db)


@db_group.command('list')
@click.option('--limit', '-n', type=int, default=None, help="Limit number of ids returned")
@click.option('--offset', '-s', type=int, default=0, help="Offset of ids returned")
@click.option('--output', '-o', type=click.Path(exists=False, file_okay=True), default=None, help="Output to file")
@click.option('--json', '-j', 'output_as_json', is_flag=True, default=False, help="Output in json format")
@click.pass_context
def list_document_ids(ctx, limit, offset, output, output_as_json):
    """List document IDs in database"""
    db = ctx.obj["db"]

    # Get all documents and apply pagination
    all_docs = db.filter(limit=limit, offset=offset)
    ids = [doc.id for doc in all_docs]

    if output_as_json:
        output_str = json.dumps(ids)
    else:
        output_str = '\n'.join(ids)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(output_str)
        click.secho(f"Results written to `{output}`", fg="blue", err=True)
    else:
        click.secho(f"Document IDs in {db.name}", fg="cyan", err=True)
        click.echo(output_str)


@db_group.command('search')
@click.argument('query')
@click.option('--limit', '-k', '-n', default=5, help='Maximum number of results')
@click.option('--search-type', '-t', default='vector',
              type=click.Choice(['vector', 'keyword', 'hybrid']),
              help='Type of search to perform')
@click.option('--return-type', '-r', default='documents',
              type=click.Choice(['documents', 'chunks']),
              help='Whether to return documents or chunks')
@click.option('--score-threshold', default=0.0, type=float, help='Minimum score threshold')
@click.option('--vector-weight', default=0.7, type=float, help='Weight for vector search in hybrid mode')
@click.option('--metadata-filter', help='Metadata filter in JSON format')
@click.option('--json', '-j', 'output_as_json', is_flag=True, default=False)
@click.option('--output', '-o', type=click.Path(file_okay=True, dir_okay=False), help='Output file for results')
@click.option('--metadata/--no-metadata', '-m', default=False, help='Include metadata in output')
@click.option('--pretty', '-p', default=False, is_flag=True)
@click.pass_context
def search(
        ctx, query, limit, search_type, return_type, score_threshold, vector_weight,
        metadata_filter, output_as_json, output, metadata, pretty
        ):
    """Search a vector database using the unified query interface."""
    # Parse metadata filter if provided
    filter_dict = None
    if metadata_filter:
        try:
            filter_dict = json.loads(metadata_filter)
        except json.JSONDecodeError:
            click.secho("Error: Metadata filter must be valid JSON", fg='red', err=True)
            raise click.Abort()

    db = ctx.obj["db"]

    # Read from stdin
    if query == "-":
        query = get_stdin_input(True, "Error: No query provided!")

    click.secho(f"Performing {search_type} search for `{query[:100]}`...", fg="blue", err=True)

    try:
        results = db.query(
            query=query,
            search_type=search_type,
            return_type=return_type,
            k=limit,
            score_threshold=score_threshold,
            filters=filter_dict,
            vector_weight=vector_weight
        )
    except Exception as e:
        click.secho(f"Search error: {str(e)}", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    if not results:
        click.secho("No results found.", fg="red", err=True)
        return

    # Format and display results
    if not output_as_json:
        output_str = ""
        if pretty:
            if len(query) > 100:
                query = query[:100] + "..."
            query = query.strip().replace("\n", " \\ ")
            title = f"{search_type.title()} Search Results for `{query}`: {len(results)} Results"
            header = title + "\n" + ("=" * len(title)) + "\n"
            if not output:
                header = click.style(header, fg="magenta")
            output_str += header

        for i, result in enumerate(results, 1):
            if pretty:
                doc_header = f"\n{i}. Document: {result.id} (Score: {result.score:.4f})\n"
                doc_header += ("-" * 40) + "\n"
                if not output:
                    doc_header = click.style(doc_header, fg="cyan")
                output_str += doc_header

                if not output:
                    output_str += click.style(result.content, fg="bright_white") + "\n"
                else:
                    output_str += result.content + "\n"
            else:
                output_str += f"Document: {result.id}\n"
                output_str += result.content + "\n"

            if metadata:
                json_str = json.dumps(result.metadata, indent=2 if pretty else None)
                if pretty and not output:
                    output_str += click.style("\n~~~~~\n\n", fg="yellow")
                    output_str += click.style("Metadata: ", fg="yellow")
                    json_str = click.style(json_str, fg="yellow")
                else:
                    output_str += "\n~~~~~\n\n"
                    output_str += "Metadata: "
                output_str += json_str + "\n"

            if i < len(results):
                output_str += click.style(f"\n{'-' * 40}\n\n", fg="cyan") if (pretty and not output) else "\n-----\n\n"
    else:
        result_data = [{
            'id': result.id,
            'type': result.type,
            'content': result.content,
            'score': result.score,
            'metadata': result.metadata
        } for result in results]

        if not metadata:
            for d in result_data:
                d.pop("metadata", None)
        output_str = json.dumps(result_data, indent=2 if pretty else None)

    if output:
        with open(output, 'w') as f:
            f.write(output_str)
        click.echo(f"Results saved to {output}", err=True)
    else:
        click.echo(output_str)


@db_group.command('add')
@click.argument('files_or_text', nargs=-1)
@click.option('--metadata', '-m', default=None,
              help='Metadata for the document in JSON format or path to .json file. '
                   'Use `-m auto` to populate with basic file information')
@click.option('--id', '-i', default=None, help='Set the id(s) for the document, separated by ",".')
@click.pass_context
def add_to_database(ctx, files_or_text, metadata, id):
    """Add document(s) to the database."""
    db = ctx.obj['db']

    all_inputs = []
    auto_metadata = []

    if len(files_or_text) == 0:
        click.secho(
            f"Error: FILES_OR_TEXT is required. Must be file path, glob, str to add, or '-' "
            "to read from stdin\n"
            "Usage:\n"
            "   $ lvdb db <DB_NAME> add path/to/the/file.txt [OPTIONS]\n"
            "   $ lvdb db <DB_NAME> add path/to/the/*.glob [OPTIONS]\n"
            "   $ echo 'text to add' | lvdb db <DB_NAME> add - [OPTIONS]",
            fg='bright_red', err=True
        )
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    if len(files_or_text) == 1 and files_or_text[0] == '-':
        input_data = get_stdin_input(True, "No input provided to stdin")
        all_inputs.append(input_data)
        auto_metadata.append({"source": "stdin"})
    else:
        for file_or_text_input in files_or_text:
            file_or_text_input = file_or_text_input.strip("'").strip('"')

            if os.path.isfile(file_or_text_input):
                click.secho(f"Reading {file_or_text_input}...", fg="blue", err=True)
                with open(file_or_text_input, "r", encoding="utf-8") as f:
                    data = f.read()
                all_inputs.append(data)
                auto_metadata.append({
                    "filename": os.path.basename(file_or_text_input),
                    "path": os.path.abspath(file_or_text_input),
                    "ext": os.path.splitext(file_or_text_input)[1],
                    "bytes": len(data.encode("utf-8"))
                })
            elif os.path.isdir(os.path.dirname(file_or_text_input)):
                glob_pattern = os.path.basename(file_or_text_input)
                if any(c in glob_pattern for c in '*?[]'):
                    matching_files = glob.glob(file_or_text_input, recursive=True)
                    for file in matching_files:
                        click.echo(f"Reading {file}...", err=True)
                        try:
                            with open(file, "r", encoding="utf-8") as f:
                                data = f.read()
                        except UnicodeDecodeError:
                            click.secho(f"Unicode Decoding error, file `{file}` is probably binary, skipping!",
                                        fg="bright_red", err=True)
                            continue
                        all_inputs.append(data)
                        auto_metadata.append({
                            "filename": os.path.basename(file),
                            "path": os.path.abspath(file),
                            "ext": os.path.splitext(file)[1],
                            "bytes": len(data.encode("utf-8"))
                        })
                else:
                    click.secho(f"Error: invalid pattern: {file_or_text_input}", fg="bright_red", err=True)
            else:
                all_inputs.append(file_or_text_input)
                auto_metadata.append({"source": "cli"})

    # Handle metadata
    if metadata:
        if metadata == "auto":
            metadata = auto_metadata
        elif os.path.isfile(metadata):
            with open(metadata, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                click.secho("Error: if `--metadata` is provided, must be valid JSON", fg='bright_red', err=True)
                raise click.exceptions.Exit(EXIT_CODE_ERROR) from e

        if isinstance(metadata, dict):
            metadata = [metadata]
        if len(metadata) != len(all_inputs):
            click.secho("Error: if providing `--metadata`, length must match number of documents. "
                        f"Found: {len(metadata)}, expected: {len(all_inputs)}.",
                        fg='bright_red', err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

    # Handle IDs
    if id is not None:
        if os.path.isfile(id):
            with open(id, "r", encoding="utf-8") as f:
                data = f.read()
            if id.lower().endswith(".json"):
                id = json.loads(data)
            else:
                id = [line.strip() for line in data.split("\n") if line.strip()]
        else:
            id = [i.strip() for i in id.split(",")]

        if len(id) != len(all_inputs):
            click.secho(
                "Error: if providing `--id`, length must match number of documents. "
                f"Found: {len(id)}, expected: {len(all_inputs)}.",
                fg='bright_red', err=True
            )
            raise click.exceptions.Exit(EXIT_CODE_ERROR)

    try:
        click.secho(f"Adding {len(all_inputs)} document(s)...", fg="blue", err=True)

        new_ids = db.upsert(
            documents=all_inputs,
            metadata=metadata,
            ids=id
        )

        click.echo(f"Successfully added {len(all_inputs)} document(s)!\nCreated ids:", err=True)
        click.echo(','.join(new_ids))

    except Exception as e:
        click.secho(f"Error: Unexpected error while adding documents: {str(repr(e))}", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e


@db_group.command('get')
@click.argument('doc_id')
@click.option('--json', '-j', 'output_as_json', is_flag=True, default=False)
@click.option('--output', '-o', type=click.Path(file_okay=True, dir_okay=False), help='Output file for results')
@click.option('--metadata/--no-metadata', '-m', default=False, help='Enable/Disable retrieving document metadata')
@click.option('--pretty', '-p', is_flag=True, default=False, help='Output results with title and formatting')
@click.pass_context
def get_document(ctx, doc_id, output_as_json, output, metadata, pretty):
    """Retrieve document DOC_ID from database"""
    db = ctx.obj['db']

    try:
        doc = db.get(doc_id)
        if doc is None:
            click.echo(f"Document {doc_id} was not found in '{db.name}'")
            return

        content = doc.content
        meta = doc.metadata

        if output_as_json:
            output_dict = {
                'id': doc_id,
                'content': content
            }
            if metadata:
                output_dict['metadata'] = meta

            output_str = json.dumps(output_dict)
        else:
            output_str = ""
            if pretty:
                title = f"Document: {doc_id}"
                if not output:
                    output_str += click.style(title + "\n", fg="cyan")
                    output_str += click.style("=" * len(title), fg="cyan") + "\n"
                    output_str += click.style(content, fg="bright_white") + "\n"
                else:
                    output_str += title + "\n"
                    output_str += "=" * len(title) + "\n"
                    output_str += content + "\n"
            else:
                output_str += content + "\n"

            if metadata:
                if pretty and not output:
                    output_str += click.style("\n~~~~~\n\n", fg="yellow")
                    output_str += click.style("Metadata: ", fg="cyan")
                else:
                    output_str += "\n~~~~~\n\n"
                    output_str += "Metadata: "
                output_str += json.dumps(meta, indent=2 if pretty else None) + "\n"

        if output:
            with open(output, 'w', encoding="utf-8") as f:
                f.write(output_str)
            click.echo(f"Results saved to {output}", err=True)
        else:
            click.echo(output_str)

    except Exception as e:
        click.secho(f"Error retrieving document: {str(e)}", fg="bright_red")
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@db_group.command('update')
@click.argument('doc_id')
@click.argument('file_or_text')
@click.option('--metadata', '-m', default=None, help='Metadata for the document in JSON format')
@click.pass_context
def update_document(ctx, doc_id, file_or_text, metadata):
    """Update document DOC_ID with new content and/or metadata"""
    db = ctx.obj['db']

    if file_or_text == "-":
        file_or_text = get_stdin_input(True, "Error: No data found in stdin")
    elif os.path.isfile(file_or_text):
        with open(file_or_text, "r", encoding="utf-8") as f:
            file_or_text = f.read()

    # Parse metadata if provided
    metadata_dict = None
    if metadata:
        if os.path.isfile(metadata):
            with open(metadata, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)
        else:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError as e:
                click.secho("Error: if `--metadata` is provided, must be valid JSON", fg='bright_red', err=True)
                raise click.exceptions.Exit(EXIT_CODE_ERROR) from e

    try:
        updated = db.update(doc_id, content=file_or_text, metadata=metadata_dict)
        if updated:
            click.echo(f"Successfully updated document: {doc_id}")
        else:
            click.echo(f"Document {doc_id} not found")

    except Exception as e:
        click.secho(f"Error: Unexpected error while updating document: {str(repr(e))}", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e


@db_group.command('delete')
@click.argument('doc_id')
@click.pass_context
def delete_document(ctx, doc_id):
    """Delete document DOC_ID from database"""
    db = ctx.obj['db']

    try:
        if not db.exists(doc_id):
            click.echo(f"Document {doc_id} not found")
            return

        deleted_count = db.delete(doc_id)
        if deleted_count > 0:
            click.echo(f"Successfully deleted document: {doc_id}")
        else:
            click.echo(f"No documents were deleted")

    except Exception as e:
        click.secho(f"Error: Unexpected error while deleting document: {str(repr(e))}", fg='bright_red')
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e


@db_group.command('shell')
@click.pass_context
def shell(ctx):
    """Start an interactive shell for database operations."""
    import glob

    db = ctx.obj['db']

    try:
        click.echo(click.style(f"Connected to database: ", fg="green")
                   + click.style(db.name, fg="green", underline=True))

        stats = db.stats
        click.secho(f"Documents: {stats['documents']}, Chunks: {stats['chunks']}", fg="blue")
        click.echo(f"Type 'help' for available commands, 'exit' to quit")

        # Simple REPL
        while True:
            try:
                command = click.prompt(f"{db.name}> ", type=str)

                if command.lower() in ('exit', 'quit', 'q'):
                    break

                if command.lower() in ('help', '?'):
                    click.echo("Available commands:")
                    click.echo("  search \"<query>\" [limit] [type] - Search for documents")
                    click.echo("    Types: vector (default), keyword, hybrid")
                    click.echo("  get <id>                       - Get document by ID")
                    click.echo("  add <file or glob>             - Add file(s) to database")
                    click.echo("  delete <id>                    - Delete document by ID")
                    click.echo("  list [limit] [offset]          - List document IDs")
                    click.echo("  count                          - Show document count")
                    click.echo("  stats                          - Show database statistics")
                    click.echo("  info                           - Show database information")
                    click.echo("  clear                          - Clear the console")
                    click.echo("  exit/quit                      - Exit shell")
                    continue

                if command.lower().startswith('search'):
                    parts = command.split(' ', 1)
                    if len(parts) < 2:
                        click.secho("Usage: search <query> [limit] [type]", fg="magenta")
                        continue

                    args = parts[1]
                    limit = 5
                    search_type = "vector"

                    # Parse query in quotes
                    if args.count('"') >= 2:
                        start_quote = args.index('"')
                        end_quote = args.index('"', start_quote + 1)
                        query_str = args[start_quote + 1:end_quote]
                        leftover = args[end_quote + 1:].strip()

                        # Parse remaining args
                        remaining_parts = leftover.split()
                        if len(remaining_parts) >= 1 and remaining_parts[0].isdigit():
                            limit = int(remaining_parts[0])
                        if len(remaining_parts) >= 2 and remaining_parts[1] in ['vector', 'keyword', 'hybrid']:
                            search_type = remaining_parts[1]
                    else:
                        query_str = args
                        arg_split = args.rsplit(" ", 2)
                        if len(arg_split) >= 2 and arg_split[-1] in ['vector', 'keyword', 'hybrid']:
                            search_type = arg_split[-1]
                            query_str = " ".join(arg_split[:-1])
                        if len(arg_split) >= 2 and arg_split[-2].isdigit():
                            limit = int(arg_split[-2])
                            query_str = " ".join(arg_split[:-2])

                    click.secho(f"{search_type.title()} search for `{query_str[:100]}`...", fg="blue")

                    try:
                        results = db.query(
                            query=query_str,
                            search_type=search_type,
                            k=limit
                        )

                        click.echo("Results:\n========\n")
                        if not results:
                            click.secho("No results found.", fg="yellow")
                        else:
                            for i, result in enumerate(results, 1):
                                click.echo(f"{i}. {result.id} (Score: {result.score:.4f}):")
                                content_preview = result.content[:200]
                                click.echo(f"   {content_preview}")
                                if len(result.content) > 200:
                                    click.echo("   ...")
                                click.secho("\n-----\n", fg="cyan")
                    except Exception as e:
                        click.secho(f"Search error: {str(e)}", fg="bright_red")
                    continue

                if command.lower().startswith('get'):
                    parts = command.split(' ', 1)
                    if len(parts) < 2:
                        click.secho("Usage: get <id>", fg="magenta")
                        continue
                    doc_id = parts[1].strip()

                    try:
                        doc = db.get(doc_id)
                        if doc:
                            click.secho(f"Document: {doc_id}\n------------------", fg="cyan")
                            click.echo(doc.content)
                            if doc.metadata:
                                click.secho("\nMetadata:", fg="cyan")
                                click.echo(json.dumps(doc.metadata, indent=2))
                        else:
                            click.secho(f"Document `{doc_id}` not found.", fg="bright_red")
                    except Exception as e:
                        click.secho(f"Error: {str(e)}", fg="bright_red")
                    continue

                if command.lower().startswith('delete'):
                    parts = command.split(' ', 1)
                    if len(parts) < 2:
                        click.secho("Usage: delete <id>", fg="magenta")
                        continue
                    doc_id = parts[1].strip()

                    try:
                        if db.exists(doc_id):
                            confirm = click.confirm(f"Are you sure you want to delete document '{doc_id}'?")
                            if confirm:
                                db.delete(doc_id)
                                click.secho(f"Document '{doc_id}' deleted.", fg="green")
                            else:
                                click.secho("Deletion canceled.", fg="yellow")
                        else:
                            click.secho(f"Document '{doc_id}' does not exist.", fg="bright_red")
                    except Exception as e:
                        click.secho(f"Error: {str(e)}", fg="bright_red")
                    continue

                if command.lower().startswith('list'):
                    parts = command.split()
                    limit = 10
                    offset = 0

                    if len(parts) > 1 and parts[1].isdigit():
                        limit = int(parts[1])
                    if len(parts) > 2 and parts[2].isdigit():
                        offset = int(parts[2])

                    try:
                        docs = db.filter(limit=limit, offset=offset)
                        total = len(db.filter())  # Get total count

                        if not docs:
                            click.secho("No documents found.", fg="yellow")
                        else:
                            click.secho(f"Document IDs (showing {len(docs)} of {total}):", fg="blue")
                            for i, doc in enumerate(docs, offset + 1):
                                click.echo(f"{i}. {doc.id}")

                            if offset + limit < total:
                                click.secho(f"\nUse 'list {limit} {offset + limit}' to see the next page", fg="yellow")
                    except Exception as e:
                        click.secho(f"Error: {str(e)}", fg="bright_red")
                    continue

                if command.lower() == 'count':
                    try:
                        stats = db.stats
                        click.secho(f"Document count: {stats['documents']}, Chunk count: {stats['chunks']}", fg="blue")
                    except Exception as e:
                        click.secho(f"Error: {str(e)}", fg="bright_red")
                    continue

                if command.lower() == 'stats':
                    _print_db_stats(db)
                    continue

                if command.lower() == 'info':
                    try:
                        stats = db.stats
                        click.secho("Database Information:", fg="blue")
                        click.echo(f"  Name: {db.name}")
                        click.echo(f"  Embedding model: {stats['embedding_model']}")
                        click.echo(f"  Embedding provider: {stats['embedding_provider']}")
                        click.echo(f"  Vector dimension: {stats['embedding_dimension']}")
                        click.echo(f"  Chunking method: {stats['chunking_method']}")
                        click.echo(f"  Chunk size: {stats['chunk_size']}")
                        click.echo(f"  Chunk overlap: {stats['chunk_overlap']}")
                        click.echo(f"  FTS search: {'enabled' if stats['fts_enabled'] else 'disabled'}")
                    except Exception as e:
                        click.secho(f"Error: {str(e)}", fg="bright_red")
                    continue

                if command.lower() == 'clear':
                    click.clear()
                    continue

                if command.lower().startswith('add '):
                    parts = command.split(' ', 1)
                    if len(parts) < 2:
                        click.secho("Usage: add <file or glob>", fg="magenta")
                        continue

                    file_pattern = parts[1].strip()
                    matching_files = glob.glob(file_pattern, recursive=True)

                    if not matching_files:
                        click.secho(f"No files found matching '{file_pattern}'", fg="bright_red")
                        continue

                    click.secho(f"Found {len(matching_files)} files. Adding to database...", fg="blue")

                    documents = []
                    metadata = []

                    for file_path in matching_files:
                        try:
                            path = Path(file_path)
                            if not path.is_file():
                                click.secho(f"Skipping {file_path} (not a file)", fg="yellow")
                                continue

                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                            except UnicodeError:
                                click.secho(f"Cannot decode {file_path} as unicode, skipping!", fg="yellow")
                                continue

                            documents.append(content)
                            metadata.append({
                                "source": file_path,
                                "filename": path.name,
                                "extension": path.suffix,
                                "added_at": datetime.now().isoformat()
                            })

                        except Exception as e:
                            click.secho(f"Error processing {file_path}: {str(e)}", fg="bright_red")

                    if documents:
                        try:
                            doc_ids = db.upsert(documents=documents, metadata=metadata)
                            click.secho(f"Successfully added {len(documents)} documents", fg="green")
                            click.echo(f"Created IDs: {', '.join(doc_ids)}")
                        except Exception as e:
                            click.secho(f"Error adding documents: {str(e)}", fg="bright_red")
                    continue

                # Unknown command
                click.secho(f"Unknown command: {command}", fg="bright_red")
                click.echo("Type 'help' for available commands")

            except click.exceptions.Abort:
                click.secho("\nCtrl+C detected, Exiting!", fg="red")
                break
            except Exception as e:
                click.secho(f"Error: {str(e)}", fg="bright_red")
                continue

        click.secho("Database connection closed.", fg="green")

    except Exception as e:
        click.secho(f"Fatal error: {str(e)}", fg="bright_red")
        raise click.Abort()
    finally:
        db.close()


if __name__ == '__main__':
    cli()