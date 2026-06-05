# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
MCP server CLI commands for LocalVectorDB
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

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
        except json.JSONDecodeError as e:
            click.echo("Error: Invalid JSON for databases-map", err=True)
            raise click.Abort() from e

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
        click.echo(f"Error checking status: {e}", err=True)
        raise click.Abort() from e


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
        click.echo(f"Test failed: {e}", err=True)
        raise click.Abort() from e


@mcp_commands.command()
def tools():
    """List available MCP tools"""
    click.echo("Available MCP Tools")
    click.echo("=" * 40)

    read_tools = [
        ("list_databases", "List all available databases"),
        ("get_database_info", "Get database statistics and configuration"),
        ("query_database", "Search using vector, keyword, or hybrid search"),
        ("filter_documents", "Filter documents by metadata"),
        ("get_document", "Retrieve a specific document"),
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
        ("get_embeddings", "Generate embeddings"),
    ]

    click.echo("\nRead-Only Tools (available in both modes):")
    for tool_name, description in read_tools:
        click.echo(f"  {tool_name}")
        click.echo(f"    {description}")

    click.echo("\nWrite Tools (read-write mode only):")
    for tool_name, description in write_tools:
        click.echo(f"  {tool_name}")
        click.echo(f"    {description}")


@mcp_commands.command()
@click.option("--output", default="-", help="Output file (default: stdout)")
def config_example(output):
    """Generate example MCP configuration file"""
    example_config = """# LocalVectorDB MCP Server Configuration

[mcp]
mode = "read-only"  # or "read-write"
log_level = "INFO"
log_operations = true
max_concurrent_operations = 10
operation_timeout = 300

# Optional: Customize which tools are available (defaults shown)
# read_only_tools = [
#     "list_databases",
#     "get_database_info",
#     "query_database",
#     "filter_documents",
#     "get_document",
#     "check_documents_exist",
#     "get_metadata_schema",
#     "get_system_info"
# ]

# write_tools = [
#     "create_database",
#     "delete_database",
#     "upsert_documents",
#     "update_document",
#     "delete_document",
#     "update_metadata_schema"
# ]

[databases]
# Root directory for local databases
root = "./databases"

# Map specific database names to paths or URLs
# [databases.map]
# docs = "./my_databases"
# remote_docs = "http://localhost:5000"

[defaults]
# Default parameters for database creation
embedding_provider = "ollama"
embedding_model = "nomic-embed-text"
chunk_size = 500
chunk_overlap = 1
chunking_method = "lines"
enable_fts = true
enable_gpu = false

[remote]
# Settings for remote database connections
timeout = 30
max_retries = 3
retry_delay = 1.0
"""

    if output == "-":
        click.echo(example_config)
    else:
        with open(output, "w") as f:
            f.write(example_config)
        click.echo(f"Example configuration written to {output}")


# Export the command group
__all__ = ["mcp_commands"]
