# LocalVectorDB MCP Server

The LocalVectorDB MCP (Model Context Protocol) server enables LLMs and tools like Claude Desktop to interact with LocalVectorDB vector databases through a standardized tool interface.

## Features

- **Unified Interface**: Works with both local (SQLite+FAISS) and remote (HTTP) databases
- **Security Modes**: Read-only (default) and read-write modes for safety
- **Auto-Detection**: Factory pattern automatically chooses local or remote connections
- **Full Functionality**: Query, filter, upsert, and manage vector databases
- **Async Support**: Native async operations for better performance

## Installation

```bash
pip install localvectordb[mcp]
```

Or from source:
```bash
pip install -e ".[mcp]"
```

## Quick Start

### 1. Start MCP Server (for testing)

```bash
# Start in read-only mode (default)
lvdb mcp serve

# Start in read-write mode
lvdb mcp serve --mode read-write

# With custom database root
lvdb mcp serve --databases-root ./my_databases

# With database mappings (mix of local and remote)
lvdb mcp serve --databases-map '{"local_db": "./databases", "remote_db": "http://localhost:5000"}'
```

### 2. Configure for Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "localvectordb": {
      "command": "lvdb",
      "args": ["mcp", "serve"],
      "env": {
        "LVDB_MCP_MODE": "read-only",
        "LVDB_MCP_DATABASES_ROOT": "/path/to/databases"
      }
    }
  }
}
```

Or with Python directly:

```json
{
  "mcpServers": {
    "localvectordb": {
      "command": "python",
      "args": ["-m", "localvectordb_server.mcp.server"],
      "env": {
        "LVDB_MCP_MODE": "read-write",
        "LVDB_MCP_DATABASES_ROOT": "/path/to/databases",
        "LVDB_MCP_EMBEDDING_PROVIDER": "ollama",
        "LVDB_MCP_EMBEDDING_MODEL": "nomic-embed-text"
      }
    }
  }
}
```

## Configuration

### Environment Variables

```bash
# Mode configuration
LVDB_MCP_MODE="read-only"              # or "read-write"
LVDB_MCP_LOG_LEVEL="INFO"              # DEBUG, INFO, WARNING, ERROR

# Database settings
LVDB_MCP_DATABASES_ROOT="./databases"  # Default root for local databases
LVDB_MCP_DATABASES_MAP="db1=./path1,db2=http://remote:5000"  # Name mappings

# Default database parameters
LVDB_MCP_EMBEDDING_PROVIDER="ollama"   # Embedding provider
LVDB_MCP_EMBEDDING_MODEL="nomic-embed-text"  # Model name
LVDB_MCP_CHUNK_SIZE="500"              # Chunk size for documents

# Configuration file
LVDB_MCP_CONFIG="/path/to/config.toml" # Path to config file
```

### Configuration File (TOML)

Create a configuration file:

```bash
lvdb mcp config-example --output mcp_config.toml
```

Example `mcp_config.toml`:

```toml
[mcp]
mode = "read-only"  # or "read-write"
log_level = "INFO"
max_concurrent_operations = 10
operation_timeout = 300

[databases]
root = "./databases"  # Default root for local databases

# Optional: Map specific databases to paths or URLs
[databases.map]
local_docs = "./my_databases"
remote_docs = "http://localhost:5000"

[defaults]
# Default parameters for database creation
embedding_provider = "ollama"
embedding_model = "nomic-embed-text"
chunk_size = 500
chunk_overlap = 50
chunking_method = "sentences"
enable_fts = true
enable_gpu = false

[remote]
# Settings for remote database connections
timeout = 30
max_retries = 3
retry_delay = 1.0
```

## Tool Configuration

The MCP server allows you to customize which tools are available through configuration. This is useful for:

- **Security**: Limiting available functionality in untrusted environments
- **Performance**: Reducing tool registration overhead for unused features  
- **Specialization**: Creating task-specific MCP servers

### Configuring Available Tools

In your configuration file:

```toml
[mcp]
mode = "read-only"

# Override default read-only tools (optional)
read_only_tools = [
    "list_databases",
    "query_database",
    "get_document"
]

# Override default write tools (optional, only used in read-write mode)
write_tools = [
    "create_database", 
    "upsert_documents"
]
```

If not specified, all appropriate tools for the mode are enabled.

## Available Tools

### Read-Only Tools (available in both modes)

- `list_databases` - List all available databases
- `get_database_info` - Get database statistics and configuration
- `query_database` - Search using vector, keyword, or hybrid search
- `filter_documents` - Filter documents by metadata
- `get_document` - Retrieve a specific document
- `check_documents_exist` - Check if documents exist
- `get_metadata_schema` - Get database metadata schema
- `get_system_info` - Get system information

### Write Tools (read-write mode only)

- `create_database` - Create a new database
- `delete_database` - Delete a database
- `upsert_documents` - Insert or update documents
- `update_document` - Update document content/metadata
- `delete_document` - Delete a document
- `update_metadata_schema` - Update database schema
- `get_embeddings` - Generate embeddings

## CLI Commands

```bash
# Start server
lvdb mcp serve [--mode MODE] [--config CONFIG]

# Check status
lvdb mcp status [--config CONFIG]

# Test functionality
lvdb mcp test [--mode MODE] [--config CONFIG]

# List available tools
lvdb mcp tools

# Generate example config
lvdb mcp config-example [--output FILE]
```

## Examples

### Using with Local Databases

```toml
[databases]
root = "./my_vector_databases"

[defaults]
embedding_provider = "ollama"
embedding_model = "nomic-embed-text"
```

### Using with Remote Server

```toml
[databases.map]
remote_db = "http://localhost:5000"

[remote]
timeout = 30
# Add API key if needed via environment:
# LVDB_API_KEY="your-api-key"
```

### Mixed Local and Remote

```toml
[databases]
root = "./local_dbs"

[databases.map]
# Local databases use the root path
local_docs = "./local_dbs"
# Remote databases use URLs
remote_docs = "http://vectordb-server:5000"
shared_knowledge = "https://api.example.com/vectordb"
```

## Security

### Read-Only Mode (Default)

- Cannot create or delete databases
- Cannot modify documents
- Cannot change schemas
- Safe for untrusted environments

### Read-Write Mode

- Full database management capabilities
- Document creation and modification
- Schema updates
- Use only in trusted environments

## Architecture

The MCP server leverages LocalVectorDB's factory pattern:

1. **VectorDB Factory**: Automatically detects local vs remote databases
2. **Unified API**: Same interface for local and remote operations
3. **Native Async**: Uses async operations where available
4. **Simple Configuration**: Direct parameter mapping to LocalVectorDB

## Troubleshooting

### Common Issues

1. **Module not found**: Install with `pip install localvectordb[mcp]`
2. **Database not found**: Check `LVDB_MCP_DATABASES_ROOT` path
3. **Permission denied**: Server is in read-only mode, use `--mode read-write`
4. **Connection failed**: For remote databases, ensure server is running

### Debug Mode

Enable debug logging:

```bash
lvdb mcp serve --log-level DEBUG
```

Or in configuration:

```toml
[mcp]
log_level = "DEBUG"
```

## Development

### Running from Source

```bash
# Install dependencies
pip install -e ".[mcp]"

# Run directly
python -m localvectordb_server.mcp.server
```

### Testing

```bash
# Test MCP functionality
lvdb mcp test --mode read-write

# Check configuration
lvdb mcp status
```

## License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
Contact: thomas.villani@gmail.com