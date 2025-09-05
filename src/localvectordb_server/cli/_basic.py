# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/cli/_basic.py

"""Basic commands for the CLI.

`lvdb serve`
    Start database server
`lvdb list`
    List all databases
`lvdb create NAME [OPTIONS]`
    Create a new database
`lvdb delete NAME`
    Delete a database
"""
import os

import click

from localvectordb_server.cli._utils import EXIT_CODE_CONFIGURATION_ERROR, EXIT_CODE_ERROR, EXIT_CODE_OLLAMA_ERROR


@click.command()
@click.option('--host', '-h', default=None, help='The interface to bind to (e.g. 127.0.0.1 for local serving).')
@click.option('--port', '-p', default=None, type=int, help='The port to bind to (default = 5000).')
@click.option('--debug', is_flag=True, help='Enable Flask debug mode.')
@click.option(
    '--log-level', '-l', default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help='Set the logging level. Must be one of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"'
)
@click.option('--disable-ollama-check', '-x', is_flag=True, help='Disable checking for ollama on startup')
@click.pass_context
def serve(ctx, host, port, debug, log_level, disable_ollama_check):
    """
    Start the LocalVectorDB server.

    Launches the LocalVectorDB server using the specified configuration file and options.
    You can control the network interface, port, logging level, and database folder. By default,
    the server checks for Ollama installation and service unless explicitly disabled.

    \b
    Examples:
        \b
        lvdb serve --host 0.0.0.0 --port 5000
        lvdb serve --config ./.lvdb-config.toml --db-folder ./dbs

    """
    config_path = ctx.obj["config_path"]
    db_folder = ctx.obj["db_folder"]

    from localvectordb.exceptions import ConfigurationError
    from localvectordb_server import create_app

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


@click.command('list')
@click.option("--details", "-v", is_flag=True, default=False, help="Show details")
@click.pass_context
def list_databases(ctx, details):
    """
    List databases

    Lists all available vector databases in the specified folder. Optionally shows details such as
    document count, chunk count, embedding model, and chunking method.

    \b
    Examples:
        \b
        lvdb list
        lvdb list --details

    """
    db_folder = ctx.obj["db_folder"]

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
                    stats = db.get_stats()
                    click.echo(f"{name:<25}{stats['documents']:<10}{stats['chunks']:<10}"
                               f"{stats['embedding_model']:<25}{stats['chunking_method']:<20}")
                    db.close()
                except Exception:
                    click.echo(f"{name:<25}{'ERROR':<10}{'ERROR':<10}{'ERROR':<25}{'ERROR':<20}")

    else:
        click.secho(f"Database folder {os.path.abspath(db_folder)} not found!", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


# TODO: expand to allow a json file/str input for schema
@click.command('create')
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
@click.pass_context
def create_vector_database(
        ctx, name, embedding_model, embedding_provider, chunk_size, chunking_method,
        chunk_overlap, metadata_schema
        ):
    """
    Create a new vector database.

    Creates a new vector database with the specified name and options. You can specify embedding
    model/provider, chunking settings, and a predefined metadata schema.

    \b
    Examples:
        \b
        lvdb create mydb --embedding-model nomic-embed-text --chunk-size 500
        lvdb create mydb --metadata-schema research_papers

    """
    db_folder = ctx.obj["db_folder"]
    if not db_folder:
        click.secho("No configuration found and `--db-folder` not specified.", fg="bright_red")
        click.echo(
            "Use the `--db-folder` option or create a configuration file with `lvdb config init`, or "
            "specify the location of an existing config file using `--config <path-to-config>`"
        )
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    cfg = ctx.obj["config"]

    # Use config defaults if not specified
    embedding_model = embedding_model or cfg.embedding.model
    embedding_provider = embedding_provider or cfg.embedding.provider
    chunk_size = chunk_size or cfg.database.chunk_size
    chunking_method = chunking_method or cfg.database.chunking_method
    chunk_overlap = chunk_overlap or cfg.database.chunk_overlap

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


@click.command('delete')
@click.argument('name')
@click.option('--confirm', '-y', flag_value=True, default=False, help='Pre-confirm deletion (danger!)')
@click.pass_context
def delete_database(ctx, name, db_folder, confirm):
    """
    Delete a database

    Permanently deletes the specified database and its associated files. Prompts for confirmation unless
    the --confirm flag is used.

    \b
    Examples:
        \b
        lvdb delete mydb
        lvdb delete mydb --confirm

    """
    db_folder = ctx.obj["db_folder"]

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
                click.style('Warning: this action cannot be undone!', fg="bright_red", bold=True) +
                '\nEnter "confirm" to delete, anything else to exit.'
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
