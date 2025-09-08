================
MCP Server
================

The LocalVectorDB MCP (Model Context Protocol) server enables seamless integration with LLMs and tools like Claude Desktop and Claude Code. It provides a standardized interface for AI tools to interact with LocalVectorDB databases through a rich set of tools for querying, managing, and manipulating vector data.

Features
========

* **Unified Interface**: Works with both local (SQLite+FAISS) and remote (HTTP) databases
* **Security Modes**: Read-only (default) and read-write modes for safety
* **Auto-Detection**: Factory pattern automatically chooses local or remote connections
* **Full Functionality**: Query, filter, upsert, and manage vector databases
* **Native Async Support**: Async operations for better performance
* **Configurable Tool Set**: Customize which tools are available based on your needs

Installation
============

Install LocalVectorDB with MCP support:

.. code-block:: bash

   pip install localvectordb[mcp]

Or from source:

.. code-block:: bash

   pip install -e ".[mcp]"

Recommended Usage Pattern
==========================

The most effective way to use the LocalVectorDB MCP server is to create specialized knowledge bases and configure minimal, focused tool sets:

**Step 1: Create Knowledge Bases with CLI**

Use the ``lvdb`` CLI to create and populate specialized databases:

.. code-block:: bash

   # Create a specialized knowledge base
   lvdb create technical_docs --embedding-model nomic-embed-text

   # Add documents using the CLI
   lvdb db technical_docs add-file ./docs/api_reference.pdf
   lvdb db technical_docs add-file ./docs/user_guide.md
   lvdb db technical_docs add-text "Key troubleshooting steps..." --metadata '{"type": "troubleshooting"}'

**Step 2: Configure Minimal Tool Set**

Configure the MCP server to expose only the tools needed for querying:

.. code-block:: toml

   [mcp]
   mode = "read-only"  # Safe for production
   
   # Minimal tool set for knowledge assistant
   read_only_tools = [
       "query_database",
       "get_document", 
       "filter_documents"
   ]

   [databases]
   root = "./knowledge_bases"

**Step 3: Create Specialized Assistant**

Configure Claude Desktop with your focused knowledge assistant:

.. code-block:: json

   {
     "mcpServers": {
       "technical_support": {
         "command": "lvdb",
         "args": ["mcp", "serve", "--config", "./technical_assistant_config.toml"],
         "env": {
           "LVDB_MCP_DATABASES_ROOT": "./knowledge_bases"
         }
       }
     }
   }

This approach provides:

* **Focused Functionality**: Only the tools you need, reducing complexity
* **Enhanced Security**: Read-only mode prevents accidental data modification
* **Better Performance**: Smaller tool set means faster initialization
* **Specialized Purpose**: Each MCP server can serve a specific domain (technical docs, company policies, research papers, etc.)
* **Easy Maintenance**: Knowledge bases are managed separately from AI tool configuration

**Example Configurations by Use Case:**

*Research Assistant:*
.. code-block:: toml

   read_only_tools = ["query_database", "filter_documents", "get_metadata_schema"]

*Document Retrieval:*
.. code-block:: toml

   read_only_tools = ["query_database", "get_document"]

*Data Explorer:*
.. code-block:: toml

   read_only_tools = ["list_databases", "get_database_info", "query_database", "filter_documents"]

Quick Start
===========

Starting the MCP Server
------------------------

For testing and development, you can start the MCP server directly:

.. code-block:: bash

   # Start in read-only mode (default)
   lvdb mcp serve

   # Start in read-write mode
   lvdb mcp serve --mode read-write

   # With custom database root
   lvdb mcp serve --databases-root ./my_databases

   # With database mappings (mix of local and remote)
   lvdb mcp serve --databases-map '{"local_db": "./databases", "remote_db": "http://localhost:5000"}'

Configuration for AI Tools
===========================

Claude Desktop
--------------

Add to your Claude Desktop configuration file (``claude_desktop_config.json``):

.. code-block:: json

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

For read-write access with custom embedding settings:

.. code-block:: json

   {
     "mcpServers": {
       "localvectordb": {
         "command": "lvdb",
         "args": ["mcp", "serve", "--mode", "read-write"],
         "env": {
           "LVDB_MCP_DATABASES_ROOT": "/path/to/databases",
           "LVDB_MCP_EMBEDDING_PROVIDER": "ollama",
           "LVDB_MCP_EMBEDDING_MODEL": "nomic-embed-text"
         }
       }
     }
   }

Using Python Directly
----------------------

You can also run the MCP server using Python directly:

.. code-block:: json

   {
     "mcpServers": {
       "localvectordb": {
         "command": "python",
         "args": ["-m", "localvectordb_server.mcp.server"],
         "env": {
           "LVDB_MCP_MODE": "read-write",
           "LVDB_MCP_DATABASES_ROOT": "/path/to/databases",
           "LVDB_MCP_CONFIG": "/path/to/config.toml"
         }
       }
     }
   }

Claude Code
-----------

Claude Code automatically detects and uses MCP servers configured in your Claude Desktop settings. No additional configuration is required.

Configuration
=============

Environment Variables
---------------------

Configure the MCP server using environment variables:

.. code-block:: bash

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

Configuration File
------------------

Generate an example configuration file:

.. code-block:: bash

   lvdb mcp config-example --output mcp_config.toml

Example ``mcp_config.toml``:

.. code-block:: toml

   [mcp]
   mode = "read-only"  # or "read-write"
   log_level = "INFO"
   max_concurrent_operations = 10
   operation_timeout = 300

   # Optional: Customize which tools are available
   read_only_tools = [
       "list_databases",
       "get_database_info", 
       "query_database",
       "filter_documents",
       "get_document",
       "check_documents_exist",
       "get_metadata_schema",
       "get_system_info"
   ]

   write_tools = [
       "create_database",
       "delete_database",
       "upsert_documents", 
       "update_document",
       "delete_document",
       "update_metadata_schema"
   ]

   [databases]
   # Root directory for local databases
   root = "./databases"

   # Map specific database names to paths or URLs
   [databases.map]
   docs = "./my_databases"
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

Available Tools
===============

The MCP server provides a comprehensive set of tools for interacting with LocalVectorDB databases. Tools are categorized into read-only (safe for all modes) and write tools (read-write mode only).

Read-Only Tools
---------------

Available in both read-only and read-write modes:

**list_databases**
   List all available databases with count and mode information.

**get_database_info**
   Get detailed information about a specific database, including statistics, configuration, and metadata schema.

**query_database**
   Search using vector, keyword, or hybrid search with configurable parameters:
   
   * ``search_type``: "vector", "keyword", or "hybrid"
   * ``return_type``: "documents", "chunks", or "context"
   * ``k``: Number of results to return
   * ``score_threshold``: Minimum score threshold
   * ``filters``: MongoDB-style metadata filters
   * ``vector_weight``: Weight for vector search in hybrid mode

**filter_documents**
   Filter documents by metadata using MongoDB-style query syntax with limit and offset support.

**get_document**
   Retrieve a specific document by ID, including content, metadata, and timestamps.

**check_documents_exist**
   Check if multiple documents exist in the database, returning existence mapping.

**get_metadata_schema**
   Get the metadata schema definition for a database, showing field types and constraints.

**get_system_info**
   Get system information including version, configuration, available providers, and enabled tools.

Write Tools
-----------

Available only in read-write mode:

**create_database**
   Create a new vector database with optional parameters:
   
   * ``metadata_schema``: Schema for document metadata
   * ``embedding_provider``: Provider for embeddings (e.g., "ollama", "openai")
   * ``embedding_model``: Model name for embeddings
   * ``chunking_method``: Method for chunking documents
   * ``chunk_size`` and ``chunk_overlap``: Chunking parameters

**delete_database**
   Delete a vector database and all its data (use with caution).

**upsert_documents**
   Insert or update documents with support for:
   
   * Single or batch document processing
   * Custom document IDs
   * Metadata association
   * Similarity threshold for duplicate detection

**update_document**
   Update a document's content and/or metadata by ID.

**delete_document**
   Delete a specific document by ID.

**update_metadata_schema**
   Update the metadata schema for a database, adding or modifying field definitions.

CLI Commands
============

The MCP server includes several CLI commands for management and testing:

.. code-block:: bash

   # Start the MCP server
   lvdb mcp serve [OPTIONS]

Options:
   * ``--mode``: Server mode (read-only, read-write)
   * ``--config``: Configuration file path (TOML format)
   * ``--databases-root``: Root directory for databases
   * ``--databases-map``: Database name to path/URL mapping (JSON format)
   * ``--log-level``: Logging level (DEBUG, INFO, WARNING, ERROR)

.. code-block:: bash

   # Check server status and configuration
   lvdb mcp status [--config CONFIG]

   # Test server functionality
   lvdb mcp test [--mode MODE] [--config CONFIG]

   # List available tools
   lvdb mcp tools

   # Generate example configuration file
   lvdb mcp config-example [--output FILE]

Usage Examples
==============

Local Databases
---------------

Configure for local database access:

.. code-block:: toml

   [databases]
   root = "./my_vector_databases"

   [defaults]
   embedding_provider = "ollama"
   embedding_model = "nomic-embed-text"

Remote Server Integration
-------------------------

Connect to a remote LocalVectorDB server:

.. code-block:: toml

   [databases.map]
   remote_db = "http://localhost:5000"

   [remote]
   timeout = 30
   # Add API key if needed via environment:
   # LVDB_API_KEY="your-api-key"

Mixed Local and Remote Setup
----------------------------

Combine local and remote databases in one configuration:

.. code-block:: toml

   [databases]
   root = "./local_dbs"

   [databases.map]
   # Local databases use the root path
   local_docs = "./local_dbs"
   # Remote databases use URLs
   remote_docs = "http://vectordb-server:5000"
   shared_knowledge = "https://api.example.com/vectordb"

Security Considerations
=======================

Read-Only Mode (Default)
-------------------------

The default read-only mode provides maximum security by preventing:

* Database creation or deletion
* Document modification or deletion
* Schema changes
* Any write operations

This mode is recommended for:

* Production environments
* Untrusted clients
* Public-facing integrations
* Exploratory data analysis

Read-Write Mode
---------------

Read-write mode enables full functionality but should be used carefully:

* **Full database management**: Create, delete, modify databases
* **Document operations**: Add, update, delete documents
* **Schema modifications**: Update metadata schemas
* **Use in trusted environments only**

Best Practices:

* Use read-write mode only when necessary
* Implement additional access controls at the infrastructure level
* Monitor operations through logging
* Consider backup strategies before enabling write access

Architecture
============

The MCP server leverages LocalVectorDB's factory pattern for seamless operation:

Database Factory Pattern
------------------------

1. **Automatic Detection**: VectorDB factory automatically detects whether a database path is local or remote
2. **Unified API**: Same interface works for both local SQLite+FAISS and remote HTTP databases
3. **Configuration Mapping**: Database names map to either file paths or HTTP URLs
4. **Parameter Inheritance**: Default settings apply to all databases with per-database overrides

Connection Management
---------------------

* **Lazy Loading**: Databases are connected only when first accessed
* **Connection Caching**: Database connections are cached for reuse
* **Async Operations**: Native async support where available with sync fallback
* **Error Handling**: Comprehensive error handling with appropriate error codes

Tool Registration
------------------

The MCP server uses dynamic tool registration:

* Tools are registered based on configuration and mode
* Read-only mode automatically excludes write tools
* Custom tool sets can be defined in configuration
* Runtime tool availability checking

Troubleshooting
===============

Common Issues
-------------

**Module not found**
   Install with MCP support: ``pip install localvectordb[mcp]``

**Database not found**
   Check the ``LVDB_MCP_DATABASES_ROOT`` path exists and contains ``.sqlite`` files

**Permission denied**
   Server is in read-only mode; use ``--mode read-write`` for write operations

**Connection failed (remote databases)**
   Ensure the remote LocalVectorDB server is running and accessible

**Tool not available**
   Check tool configuration in your TOML config file or verify mode settings

Debug Mode
----------

Enable debug logging for detailed troubleshooting:

.. code-block:: bash

   lvdb mcp serve --log-level DEBUG

Or in configuration:

.. code-block:: toml

   [mcp]
   log_level = "DEBUG"

Debug logging includes:

* Tool registration details
* Database connection attempts
* Configuration loading
* Request/response details
* Error stack traces

Testing
=======

Functionality Testing
---------------------

Test the MCP server without connecting external tools:

.. code-block:: bash

   # Test basic functionality
   lvdb mcp test

   # Test with specific mode
   lvdb mcp test --mode read-write

   # Test with custom configuration
   lvdb mcp test --config ./mcp_config.toml

Configuration Validation
-------------------------

Check your configuration is valid:

.. code-block:: bash

   # Display current configuration and status
   lvdb mcp status

   # Check with specific config file
   lvdb mcp status --config ./mcp_config.toml

Integration Testing
-------------------

For testing with Claude Desktop or other MCP clients:

1. Configure your MCP client with the LocalVectorDB server
2. Enable debug logging to monitor connections
3. Start with read-only mode for safety
4. Test basic operations like listing databases
5. Verify tool availability matches your configuration

Development
===========

Running from Source
-------------------

For development and testing:

.. code-block:: bash

   # Install in development mode
   pip install -e ".[mcp]"

   # Run directly with Python
   python -m localvectordb_server.mcp.server

   # Set environment variables for testing
   export LVDB_MCP_MODE=read-write
   export LVDB_MCP_DATABASES_ROOT=./test_databases
   python -m localvectordb_server.mcp.server

Custom Tool Development
-----------------------

The MCP server's modular design allows for extension:

.. code-block:: python

   from localvectordb_server.mcp.server import register_tool

   @register_tool("my_custom_tool", read_only=True)
   async def my_custom_tool(database_name: str, param: str) -> Dict[str, Any]:
       """Custom tool implementation"""
       # Your implementation here
       pass

Contributing
------------

When contributing to MCP server functionality:

1. Follow the existing tool patterns
2. Add comprehensive error handling
3. Include both sync and async support where possible
4. Add appropriate logging
5. Update tool lists in configuration examples
6. Add tests for new functionality

API Reference
=============

For detailed API information, see the module documentation:

* :mod:`localvectordb_server.mcp.server` - Main MCP server implementation
* :mod:`localvectordb_server.mcp.config` - Configuration management
* :mod:`localvectordb_server.cli._mcp` - CLI command implementation