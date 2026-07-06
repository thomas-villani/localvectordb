import json
import os
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn, Optional, Union, get_type_hints

import click

from localvectordb import LocalVectorDB

# ``parse_range_spec`` lives in the core library so the CLI and MCP share one
# implementation; re-exported here for backward compatibility (existing imports
# and tests reference ``localvectordb_server.cli._utils.parse_range_spec``).
from localvectordb.document_portions import parse_range_spec  # noqa: F401

if TYPE_CHECKING:
    from localvectordb_server.config import Config

EXIT_CODE_SUCCESS = 0
EXIT_CODE_ERROR = 1
EXIT_CODE_CONFIGURATION_ERROR = 2
EXIT_CODE_OLLAMA_ERROR = 3
EXIT_CODE_PERMISSION_ERROR = 4
DEFAULT_CONFIG_FILE = ".lvdb-config"


# --- Consistent console messaging ------------------------------------------
# Status/diagnostic messages go to stderr so stdout carries only command data
# (keeps `lvdb ... | jq` and friends clean). `error()` centralizes the
# red-message-then-exit pattern with a meaningful exit code.


def error(message: str, exit_code: int = EXIT_CODE_ERROR) -> NoReturn:
    """Print an error to stderr (red) and exit with ``exit_code``."""
    click.secho(message, fg="red", err=True)
    raise click.exceptions.Exit(exit_code)


def warn(message: str) -> None:
    """Print a warning to stderr (yellow)."""
    click.secho(message, fg="yellow", err=True)


def info(message: str) -> None:
    """Print a progress/status message to stderr (cyan)."""
    click.secho(message, fg="cyan", err=True)


def success(message: str) -> None:
    """Print a success message to stderr (green)."""
    click.secho(message, fg="green", err=True)


def find_config_file(config_path: Optional[str] = None) -> Optional[str]:
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
        "./.lvdb-config.toml",
        "./.lvdb-config.json",
        "./.lvdb/.lvdb-config.toml",
        "./.lvdb/.lvdb-config.json",
        "./instance/.lvdb-config.toml",
        "./instance/.lvdb-config.json",
        os.path.expanduser("~/.lvdb/.lvdb-config.toml"),
        os.path.expanduser("~/.lvdb/.lvdb-config.json"),
    ]

    for path in default_locations:
        if os.path.exists(path):
            return path

    return None


def get_ctx_db(ctx: "click.Context") -> LocalVectorDB:
    """Open the database for the current ``lvdb db NAME`` invocation, lazily.

    The ``db`` group callback only records the database name; the database is
    opened here on first use so that ``lvdb db NAME <cmd> --help`` (and shell
    completion) work without the database — or even the DB folder — existing.
    If ``ctx.obj["db"]`` is already populated (e.g. by tests), it is returned
    as-is.
    """
    db: Optional[LocalVectorDB] = ctx.obj.get("db")
    if db is not None:
        return db

    name = ctx.obj["db_name"]
    db_folder = ctx.obj.get("db_folder")

    if not db_folder or not os.path.exists(db_folder):
        click.secho(
            f"DB_FOLDER {'not specified and not found in configuration' if not db_folder else 'does not exist'}.",
            fg="bright_red",
            err=True,
        )
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    from localvectordb.exceptions import DatabaseNotFoundError

    try:
        db = LocalVectorDB(name=name, base_path=db_folder, create_if_not_exists=False)
    except DatabaseNotFoundError as e:
        click.secho(f"Database '{name}' was not found in {os.path.abspath(db_folder)}!", fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR) from e

    ctx.obj["db"] = db
    return db


def get_stdin_input(input_required=True, err_msg=None):
    err_msg = err_msg or "Error: No input data in stdin!"

    input_data_stream = click.get_text_stream("stdin")
    if input_data_stream.isatty():
        click.secho(err_msg, fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)
    data_from_stdin = input_data_stream.read().rstrip()
    if not data_from_stdin and input_required:
        click.secho(err_msg, fg="bright_red", err=True)
        raise click.exceptions.Exit(EXIT_CODE_ERROR)

    return data_from_stdin


def _detect_source_format(raw: bytes, filename: str) -> Optional[str]:
    """Return all2md's detected format for ``raw``/``filename``, or None.

    Returns the sentinel ``"plaintext"`` for text/code/undetectable input, a real
    format name (``"pdf"``, ``"html"``, ...) for everything all2md recognizes, and
    ``None`` when all2md is not installed (so callers fall back to plain reading).
    """
    try:
        from all2md import registry
    except Exception:
        return None
    try:
        return str(registry.detect_format(raw, hint=filename))
    except Exception:
        # Detection failure is not fatal — treat as undetectable plain text.
        return "plaintext"


class IngestResult(NamedTuple):
    """Outcome of turning a file into ingestible text.

    Exactly one of ``text`` / ``error`` is populated. ``metadata`` accompanies a
    successful read; ``error`` carries a human-readable reason on failure so the
    caller can decide whether to skip (globs) or abort (single file).
    """

    text: Optional[str]
    metadata: Optional[dict]
    error: Optional[str]


def load_file_for_ingest(path: str, *, force_extract: bool = False) -> IngestResult:
    """Load a file as ingestible text, using extractors for rich formats.

    Plain-text and source files are read directly as UTF-8 (fast, unchanged).
    Files all2md recognizes as a real document format (PDF, DOCX, HTML, CSV, ...)
    are routed through :class:`ExtractorRegistry` and converted to Markdown. A
    binary file that slips past detection falls back to the extractor when a plain
    UTF-8 read fails.

    Parameters
    ----------
    path : str
        Path to an existing file.
    force_extract : bool
        Always run the extractor, even for text/plaintext-detected files.
    """
    import mimetypes

    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        return IngestResult(None, None, f"could not read file: {e}")

    base_meta = {
        "filename": os.path.basename(path),
        "path": os.path.abspath(path),
        "ext": os.path.splitext(path)[1],
        "bytes": len(raw),
    }
    mimetype = mimetypes.guess_type(path)[0]

    detected = _detect_source_format(raw, os.path.basename(path))
    wants_extract = force_extract or (detected is not None and detected != "plaintext")

    def _extract() -> IngestResult:
        from localvectordb.extractors import ExtractorRegistry

        result = ExtractorRegistry.extract_text(raw, os.path.basename(path), mimetype)
        if result.success:
            meta = dict(base_meta)
            src_fmt = result.metadata.get("source_format") if result.metadata else None
            if src_fmt:
                meta["source_format"] = src_fmt
            meta["extraction_method"] = result.method
            return IngestResult(result.text, meta, None)
        return IngestResult(None, None, result.error or "extraction failed")

    if wants_extract:
        extracted = _extract()
        if extracted.text is not None:
            return extracted
        # Extraction failed — fall through to a plain read as a last resort.

    try:
        return IngestResult(raw.decode("utf-8"), dict(base_meta), None)
    except UnicodeDecodeError:
        # Undetected binary (e.g. an image): give the extractor a final chance.
        if not wants_extract:
            extracted = _extract()
            if extracted.text is not None:
                return extracted
            return IngestResult(None, None, extracted.error)
        return IngestResult(None, None, "file is binary and could not be extracted as text")


def print_json_output(data: Any):
    """Print data in JSON format with proper formatting."""
    click.echo(json.dumps(data, indent=2, default=str))


def print_db_stats(db: LocalVectorDB):
    """Print database statistics."""
    try:
        stats = db.get_stats()

        click.secho("Database Statistics:", fg="blue", bold=True)

        click.secho("\nGeneral:", fg="cyan")
        click.echo(f"  Name: {db.name}")
        click.echo(f"  Total documents: {stats['documents']:,}")
        click.echo(f"  Total chunks: {stats['chunks']:,}")
        if stats["documents"] > 0:
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
        if hasattr(db, "metadata_schema") and db.metadata_schema:
            click.secho("\nMetadata Schema:", fg="cyan")
            for field_name, field_def in db.metadata_schema.items():
                indexed = " (indexed)" if field_def.indexed else ""
                required = " (required)" if field_def.required else ""
                field_type_str = field_def.type.value if hasattr(field_def.type, "value") else str(field_def.type)
                click.echo(f"  {field_name}: {field_type_str}{indexed}{required}")

    except Exception as e:
        click.secho(f"Error retrieving statistics: {str(repr(e))}", fg="bright_red")
        # Fall back to basic stats
        click.secho("Basic Database Statistics:", fg="blue")
        click.echo(f"  Name: {db.name}")
        click.echo(f"  Embedding provider: {db.embedding_provider.provider_name}")
        click.echo(f"  Embedding model: {db.embedding_provider.model}")
        click.echo(f"  Vector dimension: {db.embedding_dimension}")


def get_nested_value(config: "Config", key_path: str) -> Any:
    """Get value from config using dot notation."""
    parts = key_path.split(".")

    if len(parts) < 2:
        raise ValueError("Key must be in format 'section.key' or 'section.subsection.key'")

    section_name = parts[0]
    if section_name not in ["database", "embedding", "server"]:
        raise ValueError(f"Invalid section '{section_name}'. Must be one of: database, embedding, server")

    section_obj = getattr(config, section_name)

    # Handle server.security.* as server.*
    if len(parts) == 3 and section_name == "server" and parts[1] == "security":
        attr_name = parts[2]
        if hasattr(section_obj.security, attr_name):
            return getattr(section_obj.security, attr_name)
        else:
            raise ValueError(f"Server setting '{attr_name}' not found")

    # Handle database.metadata_schema.*
    if len(parts) == 3 and section_name == "database" and parts[1] == "metadata_schema":
        schema_field = parts[2]
        metadata_schema = getattr(section_obj, "default_metadata_schema", {})
        if schema_field in metadata_schema:
            return metadata_schema[schema_field]
        else:
            raise ValueError(f"Metadata schema field '{schema_field}' not found")

    # Normal nested access
    current = section_obj
    for part in parts[1:]:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            raise ValueError(f"Invalid key path '{key_path}': '{part}' not found")

    return current


def set_nested_value(config: "Config", key_path: str, value_str: str) -> None:
    """Set value in config using dot notation with intelligent type conversion."""
    parts = key_path.split(".")

    if len(parts) < 2:
        raise ValueError("Key must be in format 'section.key' or 'section.subsection.key'")

    section_name = parts[0]
    if section_name not in ["database", "embedding", "server"]:
        raise ValueError(f"Invalid section '{section_name}'. Must be one of: database, embedding, server")

    section_obj = getattr(config, section_name)

    # Handle server.security.* as server.*
    if len(parts) == 3 and section_name == "server" and parts[1] == "security":
        attr_name = parts[2]
        if not hasattr(section_obj.security, attr_name):
            raise ValueError(f"Server setting '{attr_name}' not found")

        current_value = getattr(section_obj.security, attr_name)
        converted_value = _convert_string_to_type(value_str, type(current_value), section_obj.security, attr_name)
        setattr(section_obj.security, attr_name, converted_value)
        return

    # Handle database.metadata_schema.*
    if len(parts) == 3 and section_name == "database" and parts[1] == "metadata_schema":
        schema_field = parts[2]
        metadata_schema = getattr(section_obj, "default_metadata_schema", {})
        from localvectordb.core import MetadataField, MetadataFieldType

        try:
            field_config = json.loads(value_str)
            if isinstance(field_config, dict):
                metadata_schema[schema_field] = MetadataField(**field_config)
            else:
                metadata_schema[schema_field] = MetadataField(type=MetadataFieldType(field_config))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid metadata field configuration: {e}") from e

        section_obj.default_metadata_schema = metadata_schema
        return

    # Normal nested access
    if len(parts) == 2:
        attr_name = parts[1]
        if not hasattr(section_obj, attr_name):
            raise ValueError(f"Invalid key path '{key_path}': '{attr_name}' not found")

        current_value = getattr(section_obj, attr_name)
        converted_value = _convert_string_to_type(value_str, type(current_value), section_obj, attr_name)
        setattr(section_obj, attr_name, converted_value)
    else:
        # Deeper nesting - walk the path
        current = section_obj
        for part in parts[1:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise ValueError(f"Invalid key path '{key_path}': '{part}' not found")

        final_attr = parts[-1]
        if not hasattr(current, final_attr):
            raise ValueError(f"Invalid key path '{key_path}': '{final_attr}' not found")

        current_value = getattr(current, final_attr)
        converted_value = _convert_string_to_type(value_str, type(current_value), current, final_attr)
        setattr(current, final_attr, converted_value)


def _convert_string_to_type(value_str: str, target_type: type, obj: Any, attr_name: str) -> Any:
    """Convert string value to the appropriate type with validation."""
    # Handle None/Optional types by checking type hints
    hints = get_type_hints(obj.__class__)
    if attr_name in hints:
        hint = hints[attr_name]
        # Handle Optional[T] (Union[T, None])
        if hasattr(hint, "__origin__") and hint.__origin__ is Union:
            non_none_types = [arg for arg in hint.__args__ if arg is not type(None)]
            if non_none_types:
                target_type = non_none_types[0]

    # Handle special "null" or "none" values
    if value_str.lower() in ["null", "none", ""]:
        return None

    # Boolean conversion
    if target_type is bool:
        return value_str.lower() in ["true", "yes", "1", "on", "y"]

    # Integer conversion
    if target_type is int:
        try:
            return int(value_str)
        except ValueError as e:
            raise ValueError(f"Cannot convert '{value_str}' to integer") from e

    # Float conversion
    if target_type is float:
        try:
            return float(value_str)
        except ValueError as e:
            raise ValueError(f"Cannot convert '{value_str}' to float") from e

    # String conversion (default)
    if target_type is str:
        return value_str

    # List conversion
    if target_type is list or (hasattr(target_type, "__origin__") and target_type.__origin__ is list):
        # Try JSON first
        if value_str.startswith("[") and value_str.endswith("]"):
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                pass

        # Fallback to comma-separated
        if "," in value_str:
            return [item.strip().strip("\"'") for item in value_str.split(",") if item.strip()]
        else:
            return [value_str.strip().strip("\"'")]

    # Dict conversion
    if target_type is dict or (hasattr(target_type, "__origin__") and target_type.__origin__ is dict):
        try:
            return json.loads(value_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Cannot convert '{value_str}' to dict. Expected JSON format.") from e

    # For other types, try JSON parsing first, then string
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        return value_str


def _format_value_for_display(value: Any) -> str:
    """Format a value for human-readable display."""
    from localvectordb import MetadataField

    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (list, dict)):
        return json.dumps(value, indent=2)
    elif isinstance(value, MetadataField):
        return json.dumps(
            {
                "type": value.type.value if hasattr(value.type, "value") else str(value.type),
                "indexed": value.indexed,
                "required": value.required,
                "default_value": value.default_value,
            },
            indent=2,
        )
    else:
        return str(value)


def format_table(headers, rows):
    """Create a simple ASCII table without external dependencies"""
    if not rows:
        return "No data to display"

    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for row in rows:
            if i < len(row):
                max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width + 2)  # Add padding

    # Create table
    output = []

    # Top border
    border = "+" + "+".join("-" * width for width in col_widths) + "+"
    output.append(border)

    # Header row
    header_row = "|"
    for i, header in enumerate(headers):
        header_row += f" {header:<{col_widths[i] - 1}}|"
    output.append(header_row)

    # Header separator
    output.append(border)

    # Data rows
    for row in rows:
        row_str = "|"
        for i, cell in enumerate(row):
            if i < len(col_widths):
                cell_str = str(cell) if cell is not None else ""
                row_str += f" {cell_str:<{col_widths[i] - 1}}|"
        output.append(row_str)

    # Bottom border
    output.append(border)

    return "\n".join(output)
