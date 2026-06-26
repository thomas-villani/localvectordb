import json
import os
from typing import TYPE_CHECKING, Any, Optional, Union, get_type_hints

import click

from localvectordb import LocalVectorDB

if TYPE_CHECKING:
    from localvectordb_server.config import Config

EXIT_CODE_SUCCESS = 0
EXIT_CODE_ERROR = 1
EXIT_CODE_CONFIGURATION_ERROR = 2
EXIT_CODE_OLLAMA_ERROR = 3
EXIT_CODE_PERMISSION_ERROR = 4
DEFAULT_CONFIG_FILE = ".lvdb-config"


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


def print_json_output(data: Any):
    """Print data in JSON format with proper formatting."""
    click.echo(json.dumps(data, indent=2, default=str))


def print_db_stats(db: LocalVectorDB):
    """Print database statistics for v1.0"""
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
