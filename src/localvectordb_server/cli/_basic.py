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

import glob
import os

import click

from localvectordb.chunking import ChunkerFactory
from localvectordb_server.cli._utils import (
    EXIT_CODE_CONFIGURATION_ERROR,
    EXIT_CODE_ERROR,
    EXIT_CODE_OLLAMA_ERROR,
    EXIT_CODE_PERMISSION_ERROR,
    error,
    info,
    require_config,
    success,
)


def _binds_localhost(host: str) -> bool:
    """True if ``host`` binds only the loopback interface (safe default)."""
    return host in ("127.0.0.1", "localhost", "::1")


@click.command()
@click.option(
    "--host",
    "-H",
    default=None,
    help="The interface to bind to (e.g. 127.0.0.1 for local serving).",
    envvar="LVDB_HOST",
)
@click.option(
    "--port",
    "-p",
    default=None,
    type=int,
    show_default="8000",
    help="The port to bind to. Falls back to server.port from config when omitted.",
    envvar="LVDB_PORT",
)
@click.option("--debug", is_flag=True, help="Enable debug mode.", envvar="LVDB_DEBUG")
@click.option(
    "--log-level",
    "-l",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help='Set the logging level. Must be one of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"',
    envvar="LVDB_LOG_LEVEL",
)
@click.option(
    "--disable-ollama-check",
    "-x",
    is_flag=True,
    help="Disable checking for ollama on startup",
    envvar="LVDB_DISABLE_OLLAMA_CHECK",
)
@click.pass_context
def serve(ctx, host, port, debug, log_level, disable_ollama_check):
    """
    Start the LocalVectorDB server.

    Launches the LocalVectorDB server using the specified configuration file and options.
    You can control the network interface, port, logging level, and database folder. When the
    configured embedding provider is Ollama, the server verifies the Ollama installation and
    service on startup unless the check is explicitly disabled.

    \b
    Examples:
        \b
        lvdb serve --host 0.0.0.0 --port 8000
        lvdb serve --config ./.lvdb-config.toml --db-folder ./dbs

    """
    config_path = ctx.obj["config_path"]
    db_folder = ctx.obj["db_folder"]

    from localvectordb.exceptions import ConfigurationError
    from localvectordb_server.config import load_config

    # The HTTP server stack (fastapi/uvicorn/...) ships in the [server] extra,
    # which the lighter [cli] install omits. Every other command works without it;
    # only `lvdb serve` needs it, so import it here and fail with an actionable
    # hint instead of a raw ModuleNotFoundError.
    try:
        from localvectordb_server.app import create_app
    except ImportError as exc:
        missing = getattr(exc, "name", None)
        click.secho(
            f"`lvdb serve` requires the server extra (missing dependency: {missing or exc}).\n"
            'Install it with:  pip install "localvectordb[server]"',
            fg="bright_red",
            err=True,
        )
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from exc

    if config_path is None:
        # No config file found. Fall back to built-in defaults ONLY when the
        # default host binds loopback; refuse to silently expose all interfaces.
        from localvectordb_server.config import Config

        default_host = Config().server.host
        if not _binds_localhost(default_host):
            click.secho(
                f"No config file found and the built-in default host ({default_host}) is not "
                "localhost-only. Refusing to serve on all interfaces without an explicit config. "
                "Run 'lvdb config init' to create one.",
                fg="bright_red",
                err=True,
            )
            raise click.exceptions.Exit(EXIT_CODE_CONFIGURATION_ERROR)
        click.secho(
            "No config file found — using built-in defaults (localhost only). " "Run 'lvdb config init' to customize.",
            fg="yellow",
            err=True,
        )

    try:
        # Load config first for Ollama check before building the app
        config = load_config(config_path)
        if db_folder:
            config.database.root_dir = db_folder

        # Only probe for Ollama when it is the configured embedding provider.
        # A server backed by OpenAI/Jina/etc. has no dependency on a local
        # Ollama install, so requiring one at startup was spurious.
        if not disable_ollama_check and config.embedding.provider == "ollama":
            from localvectordb.exceptions import OllamaNotFoundError
            from localvectordb_server.utils.checkdeps import check_ollama_installation, check_ollama_service

            try:
                version = check_ollama_installation()
                click.secho(f"✓ Found Ollama {version}", fg="green")

                if check_ollama_service():
                    click.secho("✓ Ollama service is running", fg="green")
                else:
                    click.secho("⚠ Ollama is installed but the service is not running", fg="yellow")
                    click.secho("  To start Ollama service, run: ollama serve", fg="blue")
                    click.secho("  Or disable this check with --disable-ollama-check", fg="blue")
                    raise click.exceptions.Exit(EXIT_CODE_OLLAMA_ERROR) from None

            except OllamaNotFoundError as e:
                click.secho(f"✗ Ollama installation check failed: {e}", fg="red")
                click.secho("  Install Ollama from: https://ollama.ai/download", fg="blue")
                click.secho("  Or disable this check with --disable-ollama-check", fg="blue")
                click.secho("  Or set environment variable: LVDB_DISABLE_OLLAMA_CHECK=true", fg="blue")
                raise click.exceptions.Exit(EXIT_CODE_OLLAMA_ERROR) from e
            except Exception as e:
                click.secho(f"✗ Ollama check failed with unexpected error: {e}", fg="red")
                click.secho("  You can disable this check with --disable-ollama-check", fg="blue")
                click.secho("  Or set environment variable: LVDB_DISABLE_OLLAMA_CHECK=true", fg="blue")
                raise click.exceptions.Exit(EXIT_CODE_OLLAMA_ERROR) from e

        app = create_app(
            configuration=config_path,
            database_directory=db_folder,
            debug=debug,
            log_level=log_level,
            host=host,
            port=port,
        )

        # Run with uvicorn
        import uvicorn

        final_host = host or config.server.host
        final_port = port or config.server.port

        uvicorn.run(
            app,
            host=final_host,
            port=final_port,
            reload=debug,
            log_level="debug" if debug else (log_level or "info").lower(),
        )

    except ConfigurationError as e:
        click.secho(f"Configuration error: {e}", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_CONFIGURATION_ERROR) from e
    except Exception as e:
        click.secho(f"Error: {e}", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e


@click.command("list")
@click.option("--details", is_flag=True, default=False, help="Show details")
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
    require_config(ctx)
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
                    click.echo(
                        f"{name:<25}{stats['documents']:<10}{stats['chunks']:<10}"
                        f"{stats['embedding_model']:<25}{stats['chunking_method']:<20}"
                    )
                    db.close()
                except Exception:
                    click.echo(f"{name:<25}{'ERROR':<10}{'ERROR':<10}{'ERROR':<25}{'ERROR':<20}")

    else:
        click.secho(f"Database folder {os.path.abspath(db_folder)} not found!", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


# Schema input currently uses CLI options only; JSON file/string input may be added later.
@click.command("create")
@click.argument("name")
@click.option("--embedding-model", default=None, type=str, help="Embedding model to use")
@click.option(
    "--embedding-provider",
    default=None,
    type=str,
    help="Embedding provider (any provider registered with localvectordb, e.g. ollama, openai, "
    "google, jina, sentence_transformers, huggingface, huggingface_local)",
)
@click.option("--chunk-size", default=None, type=int, help="Max tokens per chunk")
@click.option(
    "--chunking-method",
    default=None,
    # Derived from the registry, not hardcoded: a hardcoded list silently drops
    # chunkers as they are added (it was missing `paragraphs` and `code-blocks`).
    type=click.Choice(ChunkerFactory.list_methods()),
    help="Chunking method",
)
@click.option("--chunk-overlap", default=None, type=int, help="Overlap between chunks")
@click.option(
    "--chunk-delimiter",
    default=None,
    type=str,
    help="Delimiter for --chunking-method delimiter (default: a blank line). "
    r"Escapes \n, \t, \r are interpreted; use e.g. --chunk-delimiter '\n---\n'.",
)
@click.option(
    "--metadata-schema",
    default=None,
    type=click.Choice(["files", "documents", "research_papers", "code_repository", "customer_support"]),
    help="Predefined metadata schema to use",
)
@click.pass_context
def create_vector_database(
    ctx,
    name,
    embedding_model,
    embedding_provider,
    chunk_size,
    chunking_method,
    chunk_overlap,
    chunk_delimiter,
    metadata_schema,
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
    require_config(ctx)
    db_folder = ctx.obj["db_folder"]
    if not db_folder:
        click.secho("No configuration found and `--db-folder` not specified.", fg="bright_red", err=True)
        click.secho(
            "Use the `--db-folder` option or create a configuration file with `lvdb config init`, or "
            "specify the location of an existing config file using `--config <path-to-config>`",
            err=True,
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
    # Interpret the common escape sequences so a delimiter can carry newlines/tabs
    # from a shell that does not (e.g. --chunk-delimiter '\n---\n'). Left literal
    # when the flag is omitted; the DB default ("\n\n") then applies.
    if chunk_delimiter is not None:
        chunk_delimiter = chunk_delimiter.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

    from localvectordb.embeddings import EmbeddingRegistry

    available_providers = EmbeddingRegistry.list()
    if embedding_provider.lower() not in available_providers:
        click.secho(
            f"Unknown embedding provider '{embedding_provider}'. "
            f"Available: {', '.join(sorted(available_providers))}",
            fg="bright_red",
            err=True,
        )
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    # Prepare metadata schema
    schema_dict = None
    if metadata_schema:
        from localvectordb_server.config import Config

        temp_config = Config()
        temp_config.apply_common_schema(metadata_schema)
        schema_dict = temp_config.database.default_metadata_schema

    # Only forward chunk_delimiter when supplied so the library default applies
    # otherwise (and old configs without the key stay unaffected).
    extra_kwargs = {}
    if chunk_delimiter is not None:
        extra_kwargs["chunk_delimiter"] = chunk_delimiter

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
            **extra_kwargs,
        )
        resolved_delimiter = db.chunk_delimiter
        db.close()

        click.secho(f"Created database '{name}' in {os.path.abspath(db_folder)}", fg="green")
        click.echo(f"   embedding_model: {embedding_model}")
        click.echo(f"   embedding_provider: {embedding_provider}")
        click.echo(f"   chunk_size: {chunk_size}")
        click.echo(f"   chunking_method: {chunking_method}")
        click.echo(f"   chunk_overlap: {chunk_overlap}")
        if chunking_method == "delimiter":
            click.echo(f"   chunk_delimiter: {resolved_delimiter!r}")
        if metadata_schema:
            click.echo(f"   metadata_schema: {metadata_schema}")

    except Exception as e:
        click.secho(f"Error creating database: {str(repr(e))}", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e


@click.command("delete")
@click.argument("name")
@click.option("--confirm", "-y", flag_value=True, default=False, help="Pre-confirm deletion (danger!)")
@click.pass_context
def delete_database(ctx, name, confirm):
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
    require_config(ctx)
    db_folder = ctx.obj["db_folder"]

    if not db_folder or not os.path.exists(db_folder):
        click.secho(
            f"DB_FOLDER {'not specified and not found in configuration' if not db_folder else 'does not exist'}.",
            fg="bright_red",
            err=True,
        )
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    sqlite_file = os.path.abspath(os.path.join(db_folder, f"{name}.sqlite"))
    faiss_file = os.path.abspath(os.path.join(db_folder, f"{name}.faiss"))

    if os.path.exists(sqlite_file):
        files = [sqlite_file]
        if os.path.exists(faiss_file):
            files.append(faiss_file)
        if not confirm:
            confirm = click.prompt(
                click.style(f'Are you sure you want to delete the database "{name}"?', fg="bright_red")
                + f"\nThis will remove the following file(s):\n"
                f'{chr(10).join("- " + f for f in files)}\n'
                + click.style("Warning: this action cannot be undone!", fg="bright_red", bold=True)
                + '\nEnter "confirm" to delete, anything else to exit.'
            )
            if confirm != "confirm":
                click.echo("Aborted by user!")
                return
        try:
            for f in files:
                os.remove(f)
                click.secho(f"- {f} deleted", fg="magenta")
            click.secho(f"Database '{name}' was deleted", fg="magenta")
        except PermissionError as e:
            error(f"Permission denied deleting database '{name}': {e}", EXIT_CODE_PERMISSION_ERROR)
        except Exception as e:
            click.secho(f"Error deleting database '{name}': {str(repr(e))}", fg="bright_red", err=True)
            raise click.exceptions.Exit(EXIT_CODE_ERROR) from e
    else:
        click.secho(
            f"Database {name} was not found in {os.path.abspath(db_folder)}! No action taken.",
            fg="bright_red",
            err=True,
        )
        raise click.exceptions.Exit(EXIT_CODE_ERROR)


@click.command("rename")
@click.argument("old_name")
@click.argument("new_name")
@click.pass_context
def rename_database(ctx, old_name, new_name):
    """
    Rename a database.

    Renames the database OLD_NAME to NEW_NAME by moving its on-disk files
    (the SQLite database, the FAISS index, and any hierarchical sidecar
    indexes). Database paths derive from the name, so no internal rewrite
    is needed.

    \b
    Example:
        \b
        lvdb rename oldname newname
    """
    require_config(ctx)
    db_folder = ctx.obj["db_folder"]
    if not db_folder or not os.path.exists(db_folder):
        error("DB folder not specified or does not exist.")

    if not os.path.exists(os.path.join(db_folder, f"{old_name}.sqlite")):
        error(f"Database '{old_name}' was not found in {os.path.abspath(db_folder)}!")
    if os.path.exists(os.path.join(db_folder, f"{new_name}.sqlite")):
        error(f"A database named '{new_name}' already exists. No action taken.")

    # Match the DB's own files only: "<name>.<ext>" and "<name>_<suffix>"
    # (e.g. .sqlite, .sqlite-wal, .faiss, _sections.faiss, _documents.faiss).
    moved = []
    for pattern in (f"{old_name}.*", f"{old_name}_*"):
        for path in glob.glob(os.path.join(db_folder, pattern)):
            base = os.path.basename(path)
            new_base = new_name + base[len(old_name) :]
            os.rename(path, os.path.join(db_folder, new_base))
            moved.append((base, new_base))

    success(f"Renamed database '{old_name}' to '{new_name}'")
    for old_base, new_base in moved:
        info(f"  {old_base} -> {new_base}")


@click.command("version")
def version():
    """Show the installed localvectordb version."""
    from importlib.metadata import version as _pkg_version

    click.echo(_pkg_version("localvectordb"))
