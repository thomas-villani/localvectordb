"""
MCP server CLI commands for LocalVectorDB
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from localvectordb_server.cli._utils import error
from localvectordb_server.mcp.config import MCPConfig


@click.group(name="mcp")
def mcp_commands():
    """MCP (Model Context Protocol) server commands"""
    pass


@mcp_commands.command()
@click.option(
    "--mode",
    type=click.Choice(["read-only", "read-write"]),
    default="read-only",
    help="Server mode (default: read-only)",
)
@click.option("--config", help="Configuration file path (TOML format)")
@click.option("--databases-root", help="Root directory for databases")
@click.option("--databases-map", help="Database name to path/URL mapping (JSON format)")
@click.option(
    "--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="INFO", help="Logging level"
)
def serve(mode, config, databases_root, databases_map, log_level):
    """Start the MCP server (stdio-based for Claude Desktop)"""
    import os

    # Set environment variables
    os.environ["LVDB_MCP_MODE"] = mode
    os.environ["LVDB_MCP_LOG_LEVEL"] = log_level

    if config:
        os.environ["LVDB_MCP_CONFIG"] = config

    if databases_root:
        os.environ["LVDB_MCP_DATABASES_ROOT"] = databases_root

    if databases_map:
        try:
            # Parse JSON and convert to environment variable format
            mapping = json.loads(databases_map)
            env_format = ",".join(f"{k}={v}" for k, v in mapping.items())
            os.environ["LVDB_MCP_DATABASES_MAP"] = env_format
        except json.JSONDecodeError:
            error("Error: Invalid JSON for databases-map")

    # Configure logging (to stderr so it doesn't interfere with stdio)
    logging.basicConfig(
        level=getattr(logging, log_level),
        stream=sys.stderr,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    click.echo(f"Starting LocalVectorDB MCP server in {mode} mode...", err=True)

    # Import and run server
    from localvectordb_server.mcp.server import run_mcp_server

    asyncio.run(run_mcp_server(mode=mode))


@mcp_commands.command()
@click.option("--config", help="Configuration file path")
def status(config):
    """Check MCP server status and configuration"""
    try:
        # Load configuration
        if config:
            import os

            os.environ["LVDB_MCP_CONFIG"] = config

        config_obj = MCPConfig.load(config_path=config)

        click.echo("LocalVectorDB MCP Server Status")
        click.echo("=" * 40)
        click.echo(f"Mode: {config_obj.mode}")
        click.echo(f"Database root: {config_obj.databases_root}")
        click.echo(f"Log level: {config_obj.log_level}")
        click.echo(f"Max concurrent operations: {config_obj.max_concurrent_operations}")

        # Check database mappings
        if config_obj.databases_map:
            click.echo("\nDatabase Mappings:")
            for name, path in config_obj.databases_map.items():
                click.echo(f"  - {name}: {path}")

        # Check if database directory exists
        db_path = Path(config_obj.databases_root)
        if db_path.exists():
            databases = list(db_path.glob("*.sqlite"))
            click.echo(f"\nAvailable local databases: {len(databases)}")
            for db_file in databases[:5]:  # Show first 5
                click.echo(f"  - {db_file.stem}")
            if len(databases) > 5:
                click.echo(f"  ... and {len(databases) - 5} more")
        else:
            click.echo(f"\nWarning: Database directory does not exist: {db_path}")

        # Show enabled tools
        enabled_tools = config_obj.get_enabled_tools()
        click.echo(f"\nEnabled tools ({len(enabled_tools)}):")
        for tool in enabled_tools[:10]:  # Show first 10
            click.echo(f"  - {tool}")
        if len(enabled_tools) > 10:
            click.echo(f"  ... and {len(enabled_tools) - 10} more")

    except Exception as e:
        error(f"Error checking status: {e}")


@mcp_commands.command()
@click.option("--mode", type=click.Choice(["read-only", "read-write"]), default="read-only", help="Server mode to test")
@click.option("--config", help="Configuration file path")
def test(mode, config):
    """Test MCP server functionality"""
    click.echo(f"Testing MCP server in {mode} mode...")

    try:
        import os

        # Set environment
        os.environ["LVDB_MCP_MODE"] = mode
        if config:
            os.environ["LVDB_MCP_CONFIG"] = config

        # Test configuration loading
        config_obj = MCPConfig.load(config_path=config)
        click.echo("Configuration loaded successfully")

        # Test manager initialization
        from localvectordb_server.mcp.server import MCPManager

        async def test_manager():
            manager = MCPManager(config_obj)

            # Test basic operations
            databases = await manager.list_databases()
            click.echo(f"Found {len(databases)} databases")

            # Test permission checking
            if mode == "read-only":
                try:
                    config_obj.check_write_permission("test")
                    click.echo("Error: Read-only mode should prevent write operations")
                except PermissionError:
                    click.echo("Read-only permissions working correctly")
            else:
                click.echo("Read-write mode enabled")

            await manager.cleanup()
            click.echo("Cleanup completed")

        asyncio.run(test_manager())
        click.echo("All tests passed!")

    except Exception as e:
        error(f"Test failed: {e}")


@mcp_commands.command()
def tools():
    """List available MCP tools"""
    click.echo("Available MCP Tools")
    click.echo("=" * 40)

    read_tools = [
        ("list_databases", "List all available databases"),
        ("get_database_info", "Get database statistics and configuration"),
        ("query_database", "Search using vector, keyword, or hybrid search"),
        ("find_related_documents", "Find documents related to a given document (nearest neighbours)"),
        ("filter_documents", "Filter documents by metadata"),
        ("get_document", "Retrieve a document by ID, or a portion of it"),
        ("check_documents_exist", "Check if documents exist"),
        ("get_metadata_schema", "Get database metadata schema"),
        ("get_system_info", "Get system information"),
    ]

    write_tools = [
        ("create_database", "Create a new database"),
        ("delete_database", "Delete a database"),
        ("upsert_documents", "Insert or update documents"),
        ("update_document", "Update document content/metadata"),
        ("delete_document", "Delete a document"),
        ("update_metadata_schema", "Update database schema"),
    ]

    click.echo("\nRead-Only Tools (available in both modes):")
    for tool_name, description in read_tools:
        click.echo(f"  {tool_name}")
        click.echo(f"    {description}")

    click.echo("\nWrite Tools (read-write mode only):")
    for tool_name, description in write_tools:
        click.echo(f"  {tool_name}")
        click.echo(f"    {description}")


def _toml_scalar(value) -> str:
    """Render a Python scalar as its TOML literal (bools lowercased, strings quoted)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _commented_tool_list(name: str, items: list) -> str:
    """Render a commented-out TOML array (one item per line) for the example config."""
    lines = [f"# {name} = ["]
    lines += [f'#     "{item}",' for item in items[:-1]]
    if items:
        lines.append(f'#     "{items[-1]}"')
    lines.append("# ]")
    return "\n".join(lines)


def _render_example_config(config: Optional[MCPConfig] = None) -> str:
    """Render an example MCP config TOML from ``MCPConfig``'s canonical defaults.

    Values are sourced from a live :class:`MCPConfig` rather than hardcoded, so the
    ``lvdb mcp config-example`` output can never drift from the real defaults the way
    a duplicated literal would. The ``[defaults]`` and ``[remote]`` tables are emitted
    field-for-field from ``db_defaults`` / ``remote_defaults``, so new keys appear
    automatically.
    """
    cfg = config or MCPConfig()

    defaults_lines = "\n".join(f"{k} = {_toml_scalar(v)}" for k, v in cfg.db_defaults.items())
    remote_lines = "\n".join(f"{k} = {_toml_scalar(v)}" for k, v in cfg.remote_defaults.items())

    return f"""# LocalVectorDB MCP Server Configuration

[mcp]
mode = {_toml_scalar(cfg.mode)}  # or "read-write"
log_level = {_toml_scalar(cfg.log_level)}
log_operations = {_toml_scalar(cfg.log_operations)}
max_concurrent_operations = {cfg.max_concurrent_operations}
operation_timeout = {cfg.operation_timeout}

# Optional: Customize which tools are available (defaults shown)
{_commented_tool_list("read_only_tools", cfg.read_only_tools)}

{_commented_tool_list("write_tools", cfg.write_tools)}

[databases]
# Root directory for local databases
root = {_toml_scalar(cfg.databases_root)}

# Map specific database names to paths or URLs
# [databases.map]
# docs = "./my_databases"
# remote_docs = "http://localhost:8000"

[defaults]
# Default parameters for database creation
{defaults_lines}

[remote]
# Settings for remote database connections
{remote_lines}
"""


@mcp_commands.command()
@click.option("--output", default="-", help="Output file (default: stdout)")
def config_example(output):
    """Generate example MCP configuration file"""
    example_config = _render_example_config()

    if output == "-":
        click.echo(example_config)
    else:
        with open(output, "w") as f:
            f.write(example_config)
        click.echo(f"Example configuration written to {output}")


# Export the command group
__all__ = ["mcp_commands"]
