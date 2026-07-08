Command-Line Interface
======================

The LocalVectorDB CLI provides comprehensive tools for database management, document operations, and server administration.

Exit Codes
----------

Commands exit ``0`` on success and a non-zero code on failure. The semantic
codes deliberately skip ``2``, which Click reserves for its own usage errors
(unknown option, bad argument, missing value), so a configuration problem can
never be confused with a usage mistake.

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Code
     - Meaning
   * - ``0``
     - Success
   * - ``1``
     - Generic runtime error (operation failed, not found, invalid input)
   * - ``2``
     - Usage error — emitted by Click for unknown options, bad arguments, or
       missing values (not raised by the commands themselves)
   * - ``3``
     - Ollama check failed (``lvdb serve`` could not find or reach Ollama)
   * - ``4``
     - Permission error (e.g. filesystem permission denied writing a config
       file or deleting a database)
   * - ``5``
     - Configuration error (config file missing, unreadable, invalid, or a bad
       ``config``/``tuning`` target)

Installation
------------

The CLI is included with the server installation:

.. code-block:: bash

   pip install localvectordb[server]

   # Verify installation
   lvdb --help

Server Management
-----------------

Starting the Server
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Basic server start
   lvdb serve

   # Custom configuration (--config is a global option: it goes BEFORE the subcommand)
   lvdb --config production.toml serve --host 0.0.0.0 --port 8080

   # Development mode
   lvdb serve --debug --log-level DEBUG

   # Specific database folder (--db-folder is also a global option)
   lvdb --db-folder /data/vector_databases serve

   # Disable Ollama check
   lvdb serve --disable-ollama-check

**Options**:

- ``--host, -H``: Interface to bind to (default: 127.0.0.1)
- ``--port, -p``: Port to bind to (default: 8000; falls back to ``server.port`` from config)
- ``--debug``: Enable debug mode
- ``--log-level, -l``: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- ``--disable-ollama-check, -x``: Skip Ollama availability check

.. note::
   ``--config, -c``, ``--db-folder, -d``, ``--verbose, -v`` and ``--quiet, -q`` are
   **global** options on the top-level ``lvdb`` group, not on ``serve``. They must be
   given **before** the subcommand, e.g. ``lvdb --config production.toml serve`` or
   ``lvdb --verbose db mydb add file.txt``. (``-h`` is reserved for ``--help``; the host
   short flag is ``-H``.)

**Global options** (given before the subcommand):

- ``--config, -c``: Path to config file (env: ``LVDB_SERVER_CONFIG``)
- ``--db-folder, -d``: Directory containing vector databases (env: ``LVDB_DATABASE_ROOT_DIR``)
- ``--verbose, -v``: Enable verbose (DEBUG) logging
- ``--quiet, -q``: Only log errors (suppress warnings/info)
- ``--version, -V``: Print the installed ``localvectordb`` version

Configuration Management
------------------------
For a complete overview of the configuration settings, see the :doc:`Configuration Parameters Documentation <server/config.params>`.

Initialize Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create default TOML configuration
   lvdb config init

   # Create with predefined schema
   lvdb config init --schema research_papers --output research.toml

View Configuration
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Show current configuration
   lvdb config show

   # Show specific section
   lvdb config show --section database

   # Show in different format
   lvdb config show --format json

   # Show from specific file (--config is global; it precedes the subcommand)
   lvdb --config /path/to/config.toml config show

**Example Output**:

.. code-block:: toml

   [database]
   root_dir = "./.lvdb"
   timeout = 300
   connection_pool_size = 10
   enable_gpu = false
   enable_fts = true
   chunk_size = 500
   chunk_overlap = 1
   chunking_method = "lines"

   [embedding]
   provider = "ollama"
   model = "nomic-embed-text"
   batch_size = 64
   timeout = 30
   max_retries = 3

   [server]
   host = "127.0.0.1"
   port = 8000
   log_level = "INFO"
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   max_request_size = 104857600

   [server.security]
   require_api_key = false
   api_key_header = "Authorization"
   auto_prune_expired_keys = false
   key_audit_logging = true
   auth_log_level = "INFO"
   warn_expiring_days = 7
   cors_enabled = true
   cors_allowed_origins = "*"
   cors_allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
   cors_allowed_headers = ["Content-Type", "Authorization"]
   cors_max_age = 86400

Get Configuration Values
^^^^^^^^^^^^^^^^^^^^^^^^

Get individual configuration values using dot notation:

.. code-block:: bash

   # Get basic configuration values
   lvdb config get server.host
   lvdb config get server.port
   lvdb config get database.chunk_size
   lvdb config get embedding.model

   # Get security settings
   lvdb config get server.security.require_api_key
   lvdb config get server.security.cors_enabled
   lvdb config get server.security.cors_allowed_origins

   # Get metadata schema fields
   lvdb config get database.metadata_schema.title
   lvdb config get database.metadata_schema.author

   # Different output formats
   lvdb config get server.port --format raw         # Just the value
   lvdb config get server.port --format json        # JSON format
   lvdb config get server.port --format pretty      # Formatted (default)

**Options**:

- ``--format, -f``: Output format (raw, json, pretty)

**Example Output**:

.. code-block:: console

   $ lvdb config get server.security.require_api_key
   Configuration: server.security.require_api_key
   ===============================================
   false

   $ lvdb config get server.security.cors_allowed_origins --format json
   ["*"]

   $ lvdb config get database.chunk_size --format raw
   500

Set Configuration Values
^^^^^^^^^^^^^^^^^^^^^^^^

Set configuration values using dot notation with intelligent type conversion:

.. code-block:: bash

   # Set basic values
   lvdb config set server.port 8080
   lvdb config set server.host "0.0.0.0"
   lvdb config set database.chunk_size 1000

   # Set boolean values
   lvdb config set server.security.require_api_key true
   lvdb config set database.enable_fts false

   # Set list values (JSON format)
   lvdb config set server.security.cors_allowed_origins '["http://localhost:3000", "https://app.example.com"]'

   # Set list values (comma-separated)
   lvdb config set server.security.cors_allowed_methods "GET,POST,PUT,DELETE"

   # Set null values
   lvdb config set server.security.key_database_path null

   # Set metadata schema fields (JSON format)
   lvdb config set database.metadata_schema.category '{"type": "text", "indexed": true, "required": false}'

   # Preview changes without saving
   lvdb config set server.port 9000 --dry-run

   # Skip confirmation prompt
   lvdb config set embedding.model "all-minilm-l6-v2" --force

**Options**:

- ``--dry-run, -n``: Show what would be changed without saving
- ``--force, -y``: Skip confirmation prompt

**Type Conversion**:

The ``set`` command automatically converts string inputs to the appropriate type:

- **Booleans**: ``true``, ``false``, ``yes``, ``no``, ``1``, ``0``, ``on``, ``off``
- **Integers**: ``8080``, ``500``, ``-1``
- **Floats**: ``0.8``, ``3.14``
- **Lists**: JSON format ``["item1", "item2"]`` or comma-separated ``item1,item2``
- **Objects**: JSON format ``{"key": "value"}``
- **Null values**: ``null``, ``none``, or empty string

**Example Session**:

.. code-block:: console

   $ lvdb config set server.security.require_api_key true
   Configuration Change:
   =====================
   Key: server.security.require_api_key
   Old value: false
   New value: true

   Apply this change? [y/N]: y

   ✓ Configuration updated and saved to server-cfg.toml

   $ lvdb config set server.security.cors_allowed_origins '["http://localhost:3000"]' --dry-run
   Configuration Change:
   =====================
   Key: server.security.cors_allowed_origins
   Old value: ["*"]
   New value: ["http://localhost:3000"]

   [DRY RUN] No changes made.

**Common Configuration Tasks**:

.. code-block:: bash

   # Enable API key authentication
   lvdb config set server.security.require_api_key true

   # Change server binding
   lvdb config set server.host "0.0.0.0"
   lvdb config set server.port 8080

   # Configure CORS for web applications
   lvdb config set server.security.cors_allowed_origins '["http://localhost:3000", "https://app.company.com"]'

   # Adjust chunking parameters
   lvdb config set database.chunk_size 750
   lvdb config set database.chunk_overlap 2

   # Change embedding model
   lvdb config set embedding.model "all-minilm-l6-v2"
   lvdb config set embedding.provider "openai"

   # Enable GPU acceleration
   lvdb config set database.enable_gpu true

   # Configure logging
   lvdb config set server.log_level "DEBUG"

**Dot Notation Reference**:

Configuration values can be accessed using dot notation:

**Database Settings** (``database.*``):

- ``database.root_dir`` - Database storage directory
- ``database.chunk_size`` - Maximum tokens per chunk
- ``database.chunk_overlap`` - Overlap between chunks
- ``database.chunking_method`` - Chunking strategy
- ``database.enable_gpu`` - GPU acceleration
- ``database.enable_fts`` - Full-text search
- ``database.connection_pool_size`` - Connection pool size
- ``database.timeout`` - Database timeout

**Embedding Settings** (``embedding.*``):

- ``embedding.provider`` - Provider (ollama, openai)
- ``embedding.model`` - Model name
- ``embedding.base_url`` - Provider base URL
- ``embedding.api_key`` - API key for providers
- ``embedding.batch_size`` - Batch processing size
- ``embedding.timeout`` - Request timeout
- ``embedding.max_retries`` - Maximum retry attempts

**Server Settings** (``server.*``):

- ``server.host`` - Server host address
- ``server.port`` - Server port
- ``server.log_level`` - Logging level
- ``server.log_format`` - Log message format
- ``server.max_request_size`` - Maximum request size

**Security Settings** (``server.security.*``):

- ``server.security.require_api_key`` - Enable API key auth
- ``server.security.api_key_header`` - Auth header name
- ``server.security.key_database_path`` - API key database path
- ``server.security.auto_prune_expired_keys`` - Auto cleanup
- ``server.security.key_audit_logging`` - Audit key usage
- ``server.security.cors_enabled`` - Enable CORS
- ``server.security.cors_allowed_origins`` - Allowed origins
- ``server.security.cors_allowed_methods`` - Allowed methods
- ``server.security.cors_allowed_headers`` - Allowed headers

**Metadata Schema** (``database.metadata_schema.*``):

- ``database.metadata_schema.<field_name>`` - Individual schema fields

**Example Metadata Schema Configuration**:

.. code-block:: bash

   # Add a new metadata field
   lvdb config set database.metadata_schema.priority '{"type": "integer", "indexed": true, "default_value": 1}'

   # Add a simple text field
   lvdb config set database.metadata_schema.department '{"type": "text", "indexed": true}'

   # Add a JSON field for tags
   lvdb config set database.metadata_schema.tags '{"type": "json"}'

**Security Configuration Example**:

.. code-block:: bash

   # Complete security setup
   lvdb config set server.security.require_api_key true
   lvdb config set server.security.cors_enabled true
   lvdb config set server.security.cors_allowed_origins '["https://app.company.com"]'
   lvdb config set server.security.key_audit_logging true
   lvdb config set server.security.auto_prune_expired_keys true

   # Verify security settings
   lvdb config get server.security.require_api_key
   lvdb config get server.security.cors_allowed_origins

Authentication and API Key Management
-------------------------------------

The LocalVectorDB CLI provides comprehensive API key management with permission-based access control for secure server access.

Permission Levels
^^^^^^^^^^^^^^^^^

API keys support two permission levels:

- **read_only**: Grants access to query, search, and read operations only. Cannot create, update, or delete any resources.
- **read_write**: Full access to all operations including database creation, document management, and administrative functions.

API keys are stored in `{root_dir}/api_keys.db` by default. To change the location of this file, set the
configuration setting ``key_database_path = "path/to/your/api_keys.db"`` in the ``[server.security]`` section.


Check Authentication Status
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Show current authentication configuration
   lvdb auth status

   # Show in JSON format
   lvdb auth status --format json

**Example Output**:

.. code-block:: console

   Authentication Status
   Configuration file: /home/user/server-cfg.toml
   API Authentication: Enabled

   Database-managed Keys:
     Status: Available
     Total keys: 5
     Active keys: 4
     Expired keys: 1
     Expiring soon (7 days): 1
     Recently used (24h): 2

   ⚠️  1 key(s) expiring within 7 days

Create API Keys
^^^^^^^^^^^^^^^

API keys support permission levels for fine-grained access control:

- **read_only**: Can query and search, but cannot create, update, or delete resources
- **read_write**: Full access to all operations (default)

.. code-block:: bash

   # Create a basic read-write API key (default)
   lvdb auth create-key

   # Create a read-only key for monitoring/analytics
   lvdb auth create-key --description "Monitoring Dashboard" --permission-level read_only

   # Create a read-write key with expiration
   lvdb auth create-key --description "Admin Access" --permission-level read_write --expires-days 30

   # Create a read-only key for public API
   lvdb auth create-key --description "Public Search API" \
                        --permission-level read_only \
                        --expires-days 365 \
                        --created-by "api-team"

   # JSON output for scripting
   lvdb auth create-key --description "Script Access" --permission-level read_only --format json

   # Key-only output for automation
   lvdb auth create-key --permission-level read_write --format key-only

**Options**:

- ``--description, -d``: Human-readable description of the key's purpose
- ``--permission-level, -p``: Permission level (read_only, read_write) - defaults to read_write
- ``--expires-days``: Number of days until key expires (omit for no expiration)
- ``--created-by``: Identifier of who is creating the key
- ``--format, -f``: Output format (table, json, key-only)

**Example Output**:

.. code-block:: console

   ✓ API Key Created Successfully

   Key Details:
     Key ID: key_20241201_abc123
     Description: Production API Access
     Permission Level: read_write
     Created: 2024-12-01 10:30:00 UTC
     Expires: Never

   API Key (save this now - it won't be shown again):
     lvdb_XyZ9k2mN7qP4wR8tL3vB5nM9kJ7hF2dS6xW1qE4yT8rL9pN3mK5

   ⚠️  Store this key securely - it cannot be retrieved again!

List API Keys
^^^^^^^^^^^^^

.. code-block:: bash

   # List all API keys
   lvdb auth list-keys

   # List only active keys
   lvdb auth list-keys --active-only

   # Exclude expired keys
   lvdb auth list-keys --no-expired

   # Show key management statistics
   lvdb auth list-keys --show-stats

   # JSON output
   lvdb auth list-keys --format json

**Example Output**:

.. code-block:: console

   Key Management Statistics:
     Total keys: 5
     Active keys: 4
     Expired keys: 1
     Expiring soon (7 days): 1
     Recently used (24h): 2

   API Keys:

   ID                   Description              Permission   Status     Created      Expires      Last Used
   -----------------------------------------------------------------------------------------------------------------------------------
   key_20241201_abc123  Production API Access    read_write   ACTIVE     2024-12-01   Never        2024-12-01
   key_20241125_def456  CI/CD Pipeline          read_write   ACTIVE     2024-11-25   2024-12-25   Never
   key_20241120_ghi789  Monitoring Dashboard    read_only    ACTIVE     2024-11-20   Never        2024-11-26
   key_20241115_jkl012  Development Access      read_write   EXPIRED    2024-11-15   2024-11-22   2024-11-21

Get Key Information
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Show detailed key information
   lvdb auth key-info key_20241201_abc123

   # JSON output
   lvdb auth key-info key_20241201_abc123 --format json

**Example Output**:

.. code-block:: console

   API Key Information: key_20241201_abc123

   Basic Information:
     ID: key_20241201_abc123
     Description: Production API Access
     Permission Level: read_write
     Created by: admin@company.com

   Status:
     Status: ACTIVE

   Dates:
     Created: 2024-12-01 10:30:00 UTC
     Expires: Never
     Last used: 2024-12-01 15:45:00 UTC

Revoke API Keys
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Revoke a key (with confirmation)
   lvdb auth revoke-key key_20241201_abc123

   # Force revoke without confirmation
   lvdb auth revoke-key key_20241201_abc123 --confirm

**Example Output**:

.. code-block:: console

   Key Details:
     ID: key_20241201_abc123
     Description: Production API Access
     Created: 2024-12-01 10:30:00 UTC

   Are you sure you want to revoke key 'key_20241201_abc123'? [y/N]: y
   ✓ Key 'key_20241201_abc123' has been revoked.

Rotate API Keys
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Rotate a key (creates new, revokes old)
   lvdb auth rotate-key key_20241201_abc123

   # JSON output for automation
   lvdb auth rotate-key key_20241201_abc123 --format json

   # Key-only output
   lvdb auth rotate-key key_20241201_abc123 --format key-only

**Example Output**:

.. code-block:: console

   ✓ API Key Rotated Successfully

   Original Key:
     ID: key_20241201_abc123 (now revoked)

   New Key Details:
     Key ID: key_20241201_xyz789
     Description: Rotated from key_20241201_abc123: Production API Access
     Created: 2024-12-01 16:00:00 UTC
     Expires: Never

   New API Key (save this now):
     lvdb_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v2W3x4Y5z6

   ⚠️  Update your applications with the new key!

Clean Up Expired Keys
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Preview what would be cleaned up
   lvdb auth prune-expired --dry-run

   # Soft delete (deactivate) expired keys
   lvdb auth prune-expired

   # Hard delete (permanently remove) expired keys
   lvdb auth prune-expired --hard-delete

   # Skip confirmation
   lvdb auth prune-expired --confirm

**Example Output**:

.. code-block:: console

   Found 2 expired key(s):

     key_20241120_ghi789: Temporary Access (expired 4 days ago)
     key_20241110_mno345: Test Key (expired 21 days ago)

   Are you sure you want to deactivate these 2 expired keys? [y/N]: y
   ✓ 2 expired key(s) deactivated.

**Options**:

- ``--soft-delete/--hard-delete``: Deactivate vs permanently remove (default: soft-delete)
- ``--dry-run, -n``: Show what would be pruned without doing it
- ``--confirm, -y``: Skip confirmation prompt

API Key Usage in Applications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have created an API key, use it in your applications:

**cURL Example**:

.. code-block:: bash

   # Using the API key in requests
   curl -H "Authorization: Bearer lvdb_XyZ9k2mN7qP4wR8tL3vB5nM9kJ7hF2dS6xW1qE4yT8rL9pN3mK5" \
        -H "Content-Type: application/json" \
        -X POST http://localhost:8000/api/v1/databases/my_database/query \
        -d '{"query": "machine learning", "search_type": "vector", "k": 5}'

**Python Example**:

.. code-block:: python

   import requests

   headers = {
       "Authorization": "Bearer lvdb_XyZ9k2mN7qP4wR8tL3vB5nM9kJ7hF2dS6xW1qE4yT8rL9pN3mK5",
       "Content-Type": "application/json"
   }

   response = requests.post(
       "http://localhost:8000/api/v1/databases/my_database/query",
       headers=headers,
       json={"query": "neural networks", "search_type": "vector", "k": 10}
   )

**Environment Variable**:

.. code-block:: bash

   # Set as environment variable
   export LVDB_API_KEY="lvdb_XyZ9k2mN7qP4wR8tL3vB5nM9kJ7hF2dS6xW1qE4yT8rL9pN3mK5"

   # Use in application
   curl -H "Authorization: Bearer $LVDB_API_KEY" \
        -X GET http://localhost:8000/api/v1/databases/my_database/info

Security Best Practices
^^^^^^^^^^^^^^^^^^^^^^^

**Key Management**:

- **Rotate keys regularly**: Use ``lvdb auth rotate-key`` for regular key rotation
- **Set expiration dates**: Use ``--expires-days`` for temporary access
- **Use descriptive names**: Always provide meaningful descriptions
- **Monitor usage**: Regularly check ``lvdb auth list-keys --show-stats``
- **Clean up expired keys**: Use ``lvdb auth prune-expired`` to maintain hygiene

**Key Storage**:

- Store keys securely (password managers, secrets management systems)
- Never commit keys to version control
- Use environment variables for applications
- Rotate immediately if a key is compromised

**Access Control**:

- Create separate keys for different applications/users
- Use short-lived keys for temporary access
- Revoke keys immediately when no longer needed
- Monitor the "Last Used" column to identify unused keys

**Example Security Workflow**:

.. code-block:: bash

   # Weekly security routine

   # 1. Check for keys expiring soon
   lvdb auth list-keys --show-stats

   # 2. Clean up expired keys
   lvdb auth prune-expired --dry-run
   lvdb auth prune-expired

   # 3. Rotate long-lived keys (quarterly)
   lvdb auth rotate-key old_production_key

   # 4. Check for unused keys
   lvdb auth list-keys | grep "Never" | awk '{print $1}'

   # 5. Review key descriptions and purposes
   lvdb auth list-keys


Database Operations
-------------------

List Databases
^^^^^^^^^^^^^^

.. code-block:: bash

   # List all databases
   lvdb list

   # List with details
   lvdb list --details

   # Specify database folder (--db-folder is a global option, before the subcommand)
   lvdb --db-folder /path/to/databases list

**Options:**

- ``--details``: Show document/chunk count, model, and chunk method for each database

The database folder and config file are selected with the **global** options
``--db-folder, -d`` and ``--config, -c`` (given before ``list``), e.g.
``lvdb --config server.toml list --details``.

**Example Output**:

.. code-block:: bash

   lvdb list --details

.. code-block:: text

   Databases in /home/user/.lvdb
   Name                     Documents Chunks    Model                   Method
   ====================================================================================
   research_papers         1250      8500      nomic-embed-text        sentences
   customer_support        845       3200      all-minilm              sentences
   code_documentation      324       2100      nomic-embed-text        code-blocks

Create Database
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Basic database creation
   lvdb create my_database

   # With custom configuration
   lvdb create research_db \
     --embedding-model nomic-embed-text \
     --chunk-size 600 \
     --chunking-method sections \
     --metadata-schema research_papers

   # With specific provider
   lvdb create openai_db \
     --embedding-provider openai \
     --embedding-model text-embedding-3-small

**Options**:

- ``--embedding-model``: Embedding model to use
- ``--embedding-provider``: Provider (ollama, openai)
- ``--chunk-size``: Maximum tokens per chunk
- ``--chunking-method``: Chunking strategy. One of ``sentences``, ``tokens``, ``characters``, ``words``, ``lines``, ``sections``
- ``--chunk-overlap``: Overlap between chunks
- ``--metadata-schema``: Predefined schema (documents, research_papers, etc.)

Delete Database
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Delete database (with confirmation)
   lvdb delete old_database

   # Force delete without confirmation
   lvdb delete old_database --confirm

Rename Database
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Rename a database
   lvdb rename old_database new_database

``rename OLD NEW`` moves the database's on-disk files rather than rewriting anything
internally: the SQLite database (``old.sqlite`` and its ``-wal``/``-shm`` companions),
the FAISS index (``old.faiss``), and any hierarchical sidecar indexes
(``old_sections.faiss``, ``old_documents.faiss``) are renamed to the new prefix.
The command aborts if ``OLD`` does not exist or if a database named ``NEW`` already exists.

Database-Specific Operations
----------------------------

All database-specific operations use the ``lvdb db <database_name>`` prefix.

Database Information
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Show database info
   lvdb db research_papers info

   # Show detailed statistics
   lvdb db research_papers stats

**Example Output**:

.. code-block:: text

   Database Info
   -------------
     Database: research_papers
     Path: /home/user/.lvdb
     Embedding model: nomic-embed-text
     Embedding provider: ollama
     Chunk size: 500
     Chunking method: sentences
     FTS search available: True
     Total Documents: 1250
     Total Chunks: 8500
     Metadata fields: 4

Document Management
^^^^^^^^^^^^^^^^^^^

Add Documents
"""""""""""""

.. code-block:: bash

   # Add single file
   lvdb db my_database add document.txt

   # Add multiple files
   lvdb db my_database add file1.txt file2.txt
   # Or with globs
   lvdb db my_database add *.txt

   # Add with metadata
   lvdb db my_database add paper.txt --metadata '{"title": "Research Paper", "author": "Dr. Smith"}'

   # Add from stdin
   cat document.txt | lvdb db my_database add -

   # Add with custom IDs
   lvdb db my_database add file1.txt file2.txt --id "doc_1,doc_2"

.. note::

   Rich document formats (PDF, DOCX, HTML, CSV, XLSX, ...) are automatically
   extracted to Markdown; plain text, Markdown, and source files are read
   directly as UTF-8. Use ``--extract``/``--no-extract`` to force or disable
   extraction. See :doc:`/file-extraction`.

.. note::

   Unless ``--id`` is given, documents added from files use the filename
   without extension as their id (matching ``db.upsert_from_file(...)``), so
   adding the same file again updates the existing document. Text and stdin
   input gets an auto-generated id.

Get Documents
"""""""""""""

.. code-block:: bash

   # Get document by ID
   lvdb db my_database get doc_1

   # Get with metadata
   lvdb db my_database get doc_1 --metadata

   # Pretty formatted output
   lvdb db my_database get doc_1 --pretty

   # Save to file
   lvdb db my_database get doc_1 --output retrieved_doc.txt

   # JSON output
   lvdb db my_database get doc_1 --format json

By default ``get`` returns the whole document. The following flags return a
**part** of the document instead and are mutually exclusive (only one may be
given per invocation). They compose with ``--format, -f``, ``--pretty``,
``--metadata`` and ``--output``.

.. code-block:: bash

   # A single chunk (0-based index) or an inclusive chunk range, as indexed
   lvdb db my_database get doc_1 --chunk 3
   lvdb db my_database get doc_1 --chunk 2:5

   # A character slice (0-based, end-exclusive — like Python's content[M:N])
   lvdb db my_database get doc_1 --range 0:200

   # A line range (1-based, inclusive)
   lvdb db my_database get doc_1 --lines 10:20

   # A section by its Markdown heading (case-insensitive)
   lvdb db my_database get doc_1 --section "Installation"

   # The document's section outline (headings, levels, start lines)
   lvdb db my_database get doc_1 --outline

**Selection options**:

- ``--chunk M[:N]``: Return the persisted chunk at 0-based index ``M`` (as it
  was stored at ingest), or the inclusive range ``M:N``. Open-ended forms
  ``M:``/``:N``/``:`` are accepted. In ``--format, -f`` mode each chunk is emitted
  with its ``index``, ``content`` and full ``position``.
- ``--range M:N``: Return the character slice ``content[M:N]`` (0-based,
  end-exclusive). Open-ended forms are accepted.
- ``--lines M:N``: Return lines ``M`` through ``N`` (1-based, inclusive).
  Open-ended forms are accepted.
- ``--section NAME``: Return the body of the section whose Markdown heading
  matches ``NAME`` (case-insensitive). Sections are detected on the fly from
  the document content (code-fence aware), so this works regardless of whether
  hierarchical embeddings were enabled at ingest. If no heading matches, the
  available headings are listed.
- ``--outline``: Print the document's section outline as an indented tree
  (or a JSON list of ``{index, heading, level, start_line, end_line}`` with
  ``--format, -f``).

Update Documents
""""""""""""""""

.. code-block:: bash

   # Update content from file
   lvdb db my_database update doc_1 new_content.txt

   # Update from stdin
   echo "New content" | lvdb db my_database update doc_1 -

   # Update metadata only
   lvdb db my_database update doc_1 --metadata '{"status": "revised"}'

Delete Documents
""""""""""""""""

.. code-block:: bash

   # Delete document
   lvdb db my_database delete doc_1

List Documents
""""""""""""""

.. code-block:: bash

   # List document IDs
   lvdb db my_database list

   # List with pagination
   lvdb db my_database list --limit 20 --offset 40

   # Save to file
   lvdb db my_database list --output doc_ids.txt

   # JSON format
   lvdb db my_database list --format json

Search Operations
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Basic vector search
   lvdb db my_database search "machine learning"

   # Keyword search
   lvdb db my_database search "neural networks" --search-type keyword

   # Hybrid search
   lvdb db my_database search "AI algorithms" --search-type hybrid --vector-weight 0.8

   # Limit results
   lvdb db my_database search "deep learning" --limit 3

   # Return chunks instead of documents
   lvdb db my_database search "optimization" --return-type chunks

   # Return matching chunks with surrounding context (2 neighbouring chunks each side)
   lvdb db my_database search "optimization" --return-type context --context-window 2

   # Size the context by a token budget instead of a chunk count, hard-truncating to fit
   lvdb db my_database search "optimization" \
     --return-type enriched --context-unit tokens --context-window 400 --context-truncate

   # Search with metadata filter
   lvdb db my_database search "research" --metadata-filter '{"journal": "Science"}'

   # Minimum score threshold
   lvdb db my_database search "neural" --score-threshold 0.8

   # Pretty output
   lvdb db my_database search "machine learning" --pretty

   # Save results
   lvdb db my_database search "AI" --output search_results.txt

   # JSON output
   lvdb db my_database search "algorithms" --format json --metadata

**Options**:

- ``--limit, -n``: Maximum number of results (default: 5)
- ``--search-type, -t``: Search method (``vector``, ``keyword``, ``hybrid``) - defaults to ``vector``
- ``--return-type, -r``: What to return (``documents``, ``chunks``, ``context``, ``enriched``, ``sections``) - defaults to ``documents``
- ``--search-level``: Which index to search (``chunks``, ``sections``, ``documents``) - defaults to ``chunks``
- ``--score-threshold``: Minimum score threshold (default: 0.0)
- ``--vector-weight``: Weight for the vector component in hybrid search (default: 0.7)
- ``--context-window``: Context size for ``--return-type context``/``enriched``, measured in ``--context-unit`` (default: 2)
- ``--context-unit``: Unit for ``--context-window`` (``chunks``, ``tokens``, ``words``, ``characters``) - defaults to ``chunks``
- ``--context-truncate``: Hard-truncate assembled context to exactly the budget (non-chunk ``--context-unit`` only)
- ``--metadata-filter``: Metadata filter in JSON format
- ``--metadata/--no-metadata``: Include metadata in output
- ``--pretty, -p``: Human-readable, titled output
- ``--format, -f``: Output format (``table`` or ``json``, default ``table``); ``-j`` is a shortcut for ``--format json``
- ``--output, -o``: Write results to a file instead of stdout

**Context Results**:

Use ``--return-type context`` or ``--return-type enriched`` to return matching chunks
together with their neighbouring chunks. ``--context-window`` sets how much context to
include, interpreted according to ``--context-unit``: a count of neighbouring chunks
(the default), or a ``tokens``/``words``/``characters`` budget. When a token/word/character
budget is used, ``--context-truncate`` trims the assembled context to exactly that budget.

**Hierarchical Search**:

.. code-block:: bash

   # Search the section-level index and return whole sections
   lvdb db my_database search "training loop" --search-level sections --return-type sections

   # Search the document-level index
   lvdb db my_database search "training loop" --search-level documents

The ``--search-level sections``/``documents`` options and the ``sections`` value for
``--return-type`` require a database created with ``hierarchical_embeddings=True``. On a
standard (chunk-only) database, only ``--search-level chunks`` is available. See
:doc:`hierarchical` for details on hierarchical embeddings.

**Example Search Output**:

.. code-block:: text

   Vector Search Results for `machine learning`: 3 Results
   =========================================================

   1. Document: doc_123 (Score: 0.8924)
   ----------------------------------------
   Machine learning algorithms have revolutionized the field of artificial intelligence.
   Recent advances in deep learning have enabled breakthroughs in computer vision,
   natural language processing, and reinforcement learning...

   ~~~~~

   Metadata: {
     "title": "ML Advances",
     "author": "Dr. Smith",
     "journal": "AI Quarterly"
   }

   -----

   2. Document: doc_456 (Score: 0.8456)
   ----------------------------------------
   The application of machine learning techniques to real-world problems requires
   careful consideration of data quality, model selection, and evaluation metrics...

Related Documents
^^^^^^^^^^^^^^^^^

Find the documents most similar to an existing one ("more like this"). Unlike
``search``, which matches against query *text*, ``related`` ranks other documents by
their similarity to a reference document's own embedding, excluding the reference
itself.

.. code-block:: bash

   # Documents most related to doc_1
   lvdb db my_database related doc_1

   # Limit the number of neighbours
   lvdb db my_database related doc_1 --limit 10

   # Only return sufficiently similar documents
   lvdb db my_database related doc_1 --score-threshold 0.6

   # Restrict candidates with a metadata filter
   lvdb db my_database related doc_1 --metadata-filter '{"category": "research"}'

   # Pretty, JSON, or file output (same conventions as search)
   lvdb db my_database related doc_1 --pretty --metadata
   lvdb db my_database related doc_1 --format json
   lvdb db my_database related doc_1 --output related.txt

**Options**:

- ``--limit, -n``: Maximum number of related documents (default: 5)
- ``--score-threshold``: Minimum similarity score to include (default: 0.0)
- ``--metadata-filter``: Metadata filter (JSON) applied to candidate documents
- ``--metadata/--no-metadata``: Include metadata in output
- ``--pretty, -p``: Human-readable, titled output
- ``--format, -f``: Output format (``table`` or ``json``, default ``table``); ``-j`` is a shortcut for ``--format json``
- ``--output, -o``: Write results to a file instead of stdout

Results are returned as ``QueryResult`` objects (``type="document"``) sorted by
descending similarity, using the same output formatting as ``search``. This command
wraps the :meth:`~localvectordb.LocalVectorDB.nearest_neighbors` API; see
:doc:`comparison` for the underlying document-similarity model.

Schema Management
^^^^^^^^^^^^^^^^^

The LocalVectorDB CLI provides comprehensive metadata schema management capabilities, allowing you to view, update, and
evolve database schemas with optional column remapping to rename existing columns while preserving data.

View Current Schema
"""""""""""""""""""

.. code-block:: bash

   # Display current schema (pretty format)
   lvdb db my_database schema show

   # Table format for overview
   lvdb db my_database schema show --format table

   # JSON format for programmatic use
   lvdb db my_database schema show --format json

   # Save schema to file
   lvdb db my_database schema show --format json --output current_schema.json

**Options**:

- ``--format, -f``: Output format (pretty, json, table)
- ``--output, -o``: Output to file instead of stdout

**Example Output (Table Format)**:

.. code-block:: text

   +------------------+----------+---------+----------+---------------+
   | Field Name       | Type     | Indexed | Required | Default Value |
   +------------------+----------+---------+----------+---------------+
   | title            | TEXT     | ✓       | ✓        | None          |
   | author           | TEXT     | ✓       | ✗        | Unknown       |
   | publication_year | INTEGER  | ✓       | ✗        | None          |
   | tags             | JSON     | ✗       | ✗        | []            |
   +------------------+----------+---------+----------+---------------+

Update Schema
"""""""""""""

Update database metadata schema from files or JSON strings, with optional column remapping for renaming existing columns:

.. code-block:: bash

   # Update schema from JSON file
   lvdb db my_database schema update --schema new_schema.json

   # Update with column remapping
   lvdb db my_database schema update \
     --schema new_schema.json \
     --mapping '{"old_author": "author", "pub_year": "publication_year"}'

   # Update from JSON string
   lvdb db my_database schema update \
     --schema '{"title": "text", "author": "text", "year": "integer"}'

   # Dry run to preview changes
   lvdb db my_database schema update --schema new_schema.json --dry-run

   # Skip confirmation prompts
   lvdb db my_database schema update --schema new_schema.json --force

   # Verbose output with detailed progress
   lvdb db my_database schema update --schema new_schema.json --verbose

   # Column mapping from file
   lvdb db my_database schema update \
     --schema new_schema.json \
     --mapping column_mappings.json

**Options**:

- ``--schema, -s``: JSON string or path to JSON file containing new schema definition
- ``--mapping``: Column mapping as JSON string or path to JSON file (old_name: new_name)
- ``--drop-columns, --drop``: Actually drop removed columns (WARNING: data loss)
- ``--dry-run, --dry``: Show what would be changed without making changes
- ``--force``: Skip confirmation prompts
- ``--verbose, -v``: Show detailed output

**Schema File Format**:

.. code-block:: json

   {
     "title": {
       "type": "text",
       "indexed": true,
       "required": true
     },
     "author": {
       "type": "text",
       "indexed": true,
       "required": false,
       "default_value": "Unknown"
     },
     "tags": {
       "type": "json",
       "indexed": false,
       "required": false,
       "default_value": []
     },
     "rating": {
       "type": "real",
       "indexed": true,
       "required": false,
       "default_value": 0.0
     }
   }

**Column Mapping Format**:

.. code-block:: json

   {
     "old_column_name": "new_column_name",
     "author_name": "author",
     "pub_year": "publication_year",
     "rating_text": "rating"
   }

**Example Update Session**:

.. code-block:: console

   $ lvdb db papers schema update --schema new_schema.json --mapping '{"old_author": "author"}'

   Planned Changes:
     New fields: author, genre, rating
     Removed fields: category
     Column remapping:
       old_author → author

   Proceed with schema update? [y/N]: y

   Applying schema update...

   Schema Update Complete!
     Added fields: author, genre, rating
     Remapped columns:
       old_author → author (150 rows transferred)
     Populated defaults:
       genre: 150 rows updated
       rating: 150 rows updated

**Supported Field Types**:

- ``text``: String values
- ``integer``: Whole numbers
- ``real``: Floating-point numbers
- ``boolean``: True/false values
- ``date``: Date strings (ISO format)
- ``json``: JSON objects and arrays

**Type Conversion During Remapping**:

The system supports safe type conversions during column remapping:

- ``TEXT`` → Any type (SQLite handles conversion)
- ``INTEGER`` → ``REAL`` (safe numeric widening)
- ``BOOLEAN`` → ``INTEGER/REAL`` (True=1, False=0)
- ``JSON`` → ``TEXT`` (already stored as text)

Export Schema
"""""""""""""

Export current metadata schema to a file for backup or editing:

.. code-block:: bash

   # Export to JSON file
   lvdb db my_database schema export --output current_schema.json

   # Export with sample data for reference
   lvdb db my_database schema export --output schema_with_samples.json --include-data

   # Export to TOML format (requires: pip install toml)
   lvdb db my_database schema export --output schema.toml --format toml

**Options**:

- ``--output, -o``: Output file path (required)
- ``--format, -f``: Output format (json, toml)
- ``--include-data, --with-data``: Include sample data for each field type

**Example Output**:

.. code-block:: console

   $ lvdb db papers schema export --output backup_schema.json
   Schema exported to backup_schema.json
   Fields exported: 4



Common Schema Evolution Patterns
""""""""""""""""""""""""""""""""

**Simple Column Renaming**:

.. code-block:: bash

   # Rename columns while preserving data
   lvdb db my_database schema update \
     --schema '{"title": "text", "author": "text", "year": "integer"}' \
     --mapping '{"doc_title": "title", "author_name": "author", "pub_year": "year"}'

**Adding New Fields with Defaults**:

.. code-block:: bash

   # Add new fields with default values
   lvdb db my_database schema update \
     --schema new_schema.json

   # Where new_schema.json contains:
   # {
   #   "title": "text",
   #   "author": "text",
   #   "category": {"type": "text", "default_value": "general"},
   #   "tags": {"type": "json", "default_value": []},
   #   "rating": {"type": "real", "default_value": 0.0}
   # }

**Type Conversion During Migration**:

.. code-block:: bash

   # Convert text fields to appropriate types
   lvdb db my_database schema update \
     --schema converted_schema.json \
     --mapping '{"rating_text": "rating", "is_published": "published_flag"}'

   # This converts:
   # rating_text (TEXT) → rating (REAL)
   # is_published (TEXT) → published_flag (BOOLEAN)

**Safe Schema Evolution Workflow**:

.. code-block:: bash

   # 1. Backup current schema
   lvdb db my_database schema export --output backup_$(date +%Y%m%d).json

   # 2. Test changes with dry run
   lvdb db my_database schema update --schema new_schema.json --mapping mappings.json --dry-run

   # 3. Apply changes if satisfied
   lvdb db my_database schema update --schema new_schema.json --mapping mappings.json --verbose

   # 4. Verify results
   lvdb db my_database schema show --format table
   lvdb db my_database search "test query" --limit 3

Best Practices
""""""""""""""

**Planning Schema Changes**:

.. code-block:: bash

   # Always export current schema before major changes
   lvdb db my_database schema export --output backup_schema.json

   # Use dry-run to preview all changes
   lvdb db my_database schema update --schema new_schema.json --dry-run

   # Test on a copy of your database first
   cp -r my_database my_database_test
   lvdb db my_database_test schema update --schema new_schema.json

**Column Mapping Guidelines**:

- Plan mappings carefully and test with sample data
- Use descriptive column names that follow consistent conventions
- Consider type compatibility when mapping between different field types
- Document schema changes for team collaboration

**Error Recovery**:

.. code-block:: bash

   # If something goes wrong, restore from backup
   lvdb db my_database schema update --schema backup_schema.json --force

   # Check for data integrity after major changes
   lvdb db my_database stats
   lvdb db my_database search "test query"

**Production Considerations**:

- Always backup schema before production changes
- Use staged deployments (dev → staging → production)
- Monitor application logs after schema updates
- Have rollback procedures ready
- Test with representative data volumes

**Example Production Workflow**:

.. code-block:: bash

   #!/bin/bash
   # Production schema update script

   DB_NAME="production_database"
   SCHEMA_FILE="new_schema_v2.json"
   MAPPING_FILE="v1_to_v2_mappings.json"

   # 1. Backup current schema
   echo "Creating backup..."
   lvdb db $DB_NAME schema export --output "backup_$(date +%Y%m%d_%H%M%S).json"

   # 2. Validate changes on test database
   echo "Testing on copy..."
   cp -r $DB_NAME "${DB_NAME}_test"
   lvdb db "${DB_NAME}_test" schema update --schema $SCHEMA_FILE --mapping $MAPPING_FILE

   if [ $? -ne 0 ]; then
       echo "Schema update failed on test database. Aborting."
       rm -rf "${DB_NAME}_test"
       exit 1
   fi

   # 3. Test search functionality on copy
   echo "Testing search functionality..."
   lvdb db "${DB_NAME}_test" search "test query" --limit 5 > /dev/null

   if [ $? -ne 0 ]; then
       echo "Search test failed. Aborting."
       rm -rf "${DB_NAME}_test"
       exit 1
   fi

   # 4. Apply to production
   echo "Applying to production..."
   lvdb db $DB_NAME schema update --schema $SCHEMA_FILE --mapping $MAPPING_FILE --force

   # 5. Verify production update
   echo "Verifying production update..."
   lvdb db $DB_NAME schema show --format table
   lvdb db $DB_NAME stats

   # 6. Cleanup
   rm -rf "${DB_NAME}_test"
   echo "Schema update completed successfully!"

This schema management system provides a safe, powerful way to evolve your database structure while preserving data integrity and maintaining application compatibility.

Interactive Shell
^^^^^^^^^^^^^^^^^

Start an interactive shell for comprehensive database operations, including document management and schema evolution:

.. code-block:: bash

   lvdb db my_database shell

The interactive shell provides a REPL (Read-Eval-Print Loop) environment for performing multiple operations without repeatedly connecting to the database. It includes full schema management capabilities alongside traditional document operations.

**Shell Startup**:

.. code-block:: console

   $ lvdb db research_papers shell
   Connected to database: research_papers
   Documents: 1250, Chunks: 8500
   Type 'help' for available commands, 'exit' to quit

   research_papers>

Available Commands
""""""""""""""""""

**Document Operations**:

.. code-block:: console

   search "<query>" [limit] [type]    - Search for documents
   get <id>                          - Get document by ID
   add <file or glob>                - Add file(s) to database
   delete <id>                       - Delete document by ID
   list [limit] [offset]             - List document IDs
   count                             - Show document count
   stats                             - Show database statistics
   info                              - Show database information

**Schema Management**:

.. code-block:: console

   schema show [format]              - Show current schema (pretty|json|table)
   schema update <file>              - Update schema from JSON file
   schema update-str <json>          - Update schema from JSON string
   schema export <file>              - Export current schema to file
   schema map <old> <new>            - Add column mapping for next update
   schema map-clear                  - Clear column mappings
   schema map-show                   - Show current column mappings

**General Commands**:

.. code-block:: console

   clear                             - Clear the console
   help                              - Show this help
   exit/quit                         - Exit shell

Document Operations Examples
""""""""""""""""""""""""""""

**Search Operations**:

.. code-block:: console

   research_papers> search "neural networks" 3 vector
   Found 3 results:
   ========================================

   1. Document: paper_001 (Score: 0.8745)
   ----------------------------------------
   Neural networks are computational models inspired by biological neural networks...

   2. Document: paper_127 (Score: 0.8234)
   ----------------------------------------
   Deep learning approaches to natural language processing...

   3. Document: paper_089 (Score: 0.7892)
   ----------------------------------------
   Reinforcement learning in autonomous systems...

   research_papers> search "machine learning" 5 hybrid
   Hybrid search for `machine learning`...
   Found 5 results combining vector and keyword search...

**Document Management**:

.. code-block:: console

   research_papers> add new_papers/*.md
   Found 3 files. Adding to database...
   Successfully added 3 documents
   Created IDs: paper_1251, paper_1252, paper_1253

   research_papers> get paper_001
   Document: paper_001
   ----------------------------------------
   Neural networks are computational models inspired by biological neural networks...

   research_papers> delete paper_old_001
   Delete document 'paper_old_001'? [y/N]: y
   Successfully deleted document: paper_old_001

   research_papers> list 5 10
   paper_011
   paper_012
   paper_013
   paper_014
   paper_015

   research_papers> count
   Total documents: 1253

**Database Information**:

.. code-block:: console

   research_papers> stats
   Database Statistics:
   Documents: 1253, Chunks: 8521, Avg chunks/doc: 6.80
   Embedding model: nomic-embed-text
   Provider: ollama
   Chunk size: 500, Overlap: 50
   FTS enabled: True

   research_papers> info
   Database Info
   -------------
     Database: research_papers
     Embedding model: nomic-embed-text
     Total Documents: 1253
     Total Chunks: 8521
     Schema fields: 6

Schema Management Examples
""""""""""""""""""""""""""

**View Current Schema**:

.. code-block:: console

   research_papers> schema show table
   +------------------+----------+---------+----------+---------------+
   | Field Name       | Type     | Indexed | Required | Default Value |
   +------------------+----------+---------+----------+---------------+
   | title            | TEXT     | ✓       | ✓        | None          |
   | authors          | JSON     | ✗       | ✗        | []            |
   | journal          | TEXT     | ✓       | ✗        | None          |
   | publication_year | INTEGER  | ✓       | ✗        | None          |
   | keywords         | JSON     | ✗       | ✗        | []            |
   | citation_count   | INTEGER  | ✓       | ✗        | 0             |
   +------------------+----------+---------+----------+---------------+

   research_papers> schema show json
   {
     "title": {
       "type": "text",
       "indexed": true,
       "required": true,
       "default_value": null
     },
     "authors": {
       "type": "json",
       "indexed": false,
       "required": false,
       "default_value": []
     }
   }

**Schema Export and Backup**:

.. code-block:: console

   research_papers> schema export backup_schema.json
   Schema exported to backup_schema.json
   Fields exported: 6

   research_papers> schema export schema_$(date +%Y%m%d).json
   Schema exported to schema_20241201.json
   Fields exported: 6

**Column Mapping and Schema Updates**:

.. code-block:: console

   research_papers> schema map paper_title title
   Added column mapping: paper_title → title

   research_papers> schema map author_list authors
   Added column mapping: author_list → authors

   research_papers> schema map pub_year publication_year
   Added column mapping: pub_year → publication_year

   research_papers> schema map-show
   Current column mappings:
     paper_title → title
     author_list → authors
     pub_year → publication_year

   research_papers> schema update new_research_schema.json

   Planned Changes:
     New fields: keywords, citation_count, impact_factor
     Removed fields: old_category
     Column remapping:
       paper_title → title
       author_list → authors
       pub_year → publication_year

   Proceed with schema update? [y/N]: y

   Applying schema update...

   Schema Update Complete!
     Added fields: keywords, citation_count, impact_factor
     Remapped columns:
       paper_title → title (1253 rows transferred)
       author_list → authors (1253 rows transferred)
       pub_year → publication_year (1253 rows transferred)
     Populated defaults:
       keywords: 1253 rows updated
       citation_count: 1253 rows updated

**Quick Schema Updates**:

.. code-block:: console

   research_papers> schema update-str '{"title": "text", "author": "text", "tags": {"type": "json", "default_value": []}}'
   Apply schema update? [y/N]: y
   Schema updated successfully!

   research_papers> schema map-clear
   Column mappings cleared

Complete Workflow Examples
""""""""""""""""""""""""""

**Research Paper Database Migration**:

.. code-block:: console

   papers> # Starting with legacy schema
   papers> schema show table
   +---------------+----------+---------+----------+---------------+
   | Field Name    | Type     | Indexed | Required | Default Value |
   +---------------+----------+---------+----------+---------------+
   | paper_title   | TEXT     | ✓       | ✓        | None          |
   | author_names  | TEXT     | ✗       | ✗        | None          |
   | journal_name  | TEXT     | ✓       | ✗        | None          |
   | year_pub      | INTEGER  | ✓       | ✗        | None          |
   +---------------+----------+---------+----------+---------------+

   papers> # Export backup before changes
   papers> schema export legacy_backup.json
   Schema exported to legacy_backup.json
   Fields exported: 4

   papers> # Set up column mappings for modernization
   papers> schema map paper_title title
   Added column mapping: paper_title → title

   papers> schema map author_names authors
   Added column mapping: author_names → authors

   papers> schema map journal_name journal
   Added column mapping: journal_name → journal

   papers> schema map year_pub publication_year
   Added column mapping: year_pub → publication_year

   papers> # Review planned mappings
   papers> schema map-show
   Current column mappings:
     paper_title → title
     author_names → authors
     journal_name → journal
     year_pub → publication_year

   papers> # Apply modern research schema
   papers> schema update modern_research_schema.json

   Planned Changes:
     New fields: title, authors, journal, publication_year, keywords, doi, abstract
     Removed fields: (none - all mapped)
     Column remapping:
       paper_title → title
       author_names → authors
       journal_name → journal
       year_pub → publication_year

   Proceed with schema update? [y/N]: y

   Applying schema update...

   Schema Update Complete!
     Added fields: title, authors, journal, publication_year, keywords, doi, abstract
     Remapped columns:
       paper_title → title (892 rows transferred)
       author_names → authors (892 rows transferred)
       journal_name → journal (856 rows transferred)
       year_pub → publication_year (892 rows transferred)
     Populated defaults:
       keywords: 892 rows updated
       doi: 892 rows updated
       abstract: 892 rows updated

   papers> # Verify the migration worked
   papers> schema show table
   +------------------+----------+---------+----------+---------------+
   | Field Name       | Type     | Indexed | Required | Default Value |
   +------------------+----------+---------+----------+---------------+
   | title            | TEXT     | ✓       | ✓        | None          |
   | authors          | JSON     | ✗       | ✗        | []            |
   | journal          | TEXT     | ✓       | ✗        | None          |
   | publication_year | INTEGER  | ✓       | ✗        | None          |
   | keywords         | JSON     | ✗       | ✗        | []            |
   | doi              | TEXT     | ✓       | ✗        | None          |
   | abstract         | TEXT     | ✓       | ✗        | None          |
   +------------------+----------+---------+----------+---------------+

   papers> # Test search with new schema
   papers> search "machine learning" 3
   Found 3 results:
   ========================================

   1. Document: paper_045 (Score: 0.9123)
   ----------------------------------------
   Machine learning techniques for automated research discovery...

   papers> # Test document retrieval
   papers> get paper_045
   Document: paper_045
   ----------------------------------------
   Machine learning techniques for automated research discovery...

   Metadata:
   {
     "title": "ML for Research Discovery",
     "authors": ["Chen, L.", "Rodriguez, M."],
     "journal": "Nature Machine Intelligence",
     "publication_year": 2023,
     "keywords": ["machine learning", "research", "automation"],
     "doi": "10.1038/s42256-023-00123-4",
     "abstract": null
   }

   papers> # Schema migration successful!
   papers> count
   Total documents: 892

**Development and Testing Workflow**:

.. code-block:: console

   dev_db> # Quick schema prototyping
   dev_db> schema update-str '{"name": "text", "category": "text", "priority": {"type": "integer", "default_value": 1}}'
   Apply schema update? [y/N]: y
   Schema updated successfully!

   dev_db> # Add test data
   dev_db> add test_documents/*.txt
   Successfully added 5 documents

   dev_db> # Test search functionality
   dev_db> search "test query" 3
   Found 3 results...

   dev_db> # Export schema for production use
   dev_db> schema export production_ready_schema.json
   Schema exported to production_ready_schema.json
   Fields exported: 3

   dev_db> # Clear screen and continue working
   dev_db> clear

   dev_db> # Final verification
   dev_db> stats
   Database Statistics:
   Documents: 5, Chunks: 23, Avg chunks/doc: 4.60


Shell vs CLI Command Comparison
"""""""""""""""""""""""""""""""

**Interactive Shell Advantages**:

- **Persistent Connection**: No reconnection overhead between commands
- **Command History**: Easy to repeat and modify previous commands
- **Context Awareness**: Schema mappings persist across commands
- **Real-time Feedback**: Immediate results and error handling
- **Workflow Continuity**: Schema updates → test search → verify data

**When to Use Shell vs CLI**:

**Use Interactive Shell for**:

.. code-block:: bash

   # Data exploration and analysis
   lvdb db research shell
   # > search "topic A" 10
   # > search "topic B" 10
   # > search "topic C" 10

   # Schema development and testing
   # > schema show
   # > schema update-str '{"test": "text"}'
   # > add test.txt
   # > search "test"

   # Complex migrations
   # > schema export backup.json
   # > schema map old new
   # > schema update new.json
   # > search "verify"

**Use CLI Commands for**:

.. code-block:: bash

   # Scripting and automation
   lvdb db prod search "query" --format json > results.json

   # One-time operations
   lvdb db prod add important_doc.txt

   # Production deployments
   lvdb db prod schema update --schema prod_schema.json --force

This enhanced interactive shell provides a powerful environment for both day-to-day database operations and complex
schema evolution tasks, making it easy to iterate, test, and deploy changes safely.


Standalone Chunking
-------------------

``lvdb chunk`` runs LocalVectorDB's position-aware chunkers on their own — no database,
embedding provider, or configuration required. It reads text from files, globs, direct
arguments, or ``-`` (stdin), chunks each input, and writes one JSON object per chunk
(JSONL) to stdout or ``--output``. This is handy for inspecting how a document will be
split before ingesting it, or for feeding chunks to another tool.

.. code-block:: bash

   # Chunk a Markdown file to stdout (JSONL)
   lvdb chunk notes.md

   # Chunk a PDF (extracted to Markdown first), 300 tokens per chunk
   lvdb chunk report.pdf --method sentences --max-tokens 300

   # Chunk every Markdown file matching a glob into a file
   lvdb chunk "docs/*.md" --method paragraphs -o chunks.jsonl

   # Chunk text from stdin with word chunks and a 20-token overlap
   echo "some long text..." | lvdb chunk - --method words --overlap 20

Rich file formats (PDF, DOCX, HTML, ...) are extracted to Markdown first, exactly as
``lvdb db <name> add`` does. Use ``--extract``/``--no-extract`` to force or disable
extraction (the default is auto-detection per file).

**Options**:

- ``--method, -M``: Chunking strategy. One of ``sentences``, ``tokens``, ``words``, ``lines``, ``characters``, ``paragraphs``, ``sections``, ``code-blocks`` (default: ``sentences``)
- ``--max-tokens, --chunk-size, -s``: Maximum tokens per chunk (default: 500)
- ``--overlap``: Token overlap between consecutive chunks; ignored by some strategies (default: 0)
- ``--output, -o``: Write JSONL to this file instead of stdout
- ``--extract/--no-extract``: Force or disable text extraction for file inputs (default: auto)

**Output Format**:

Each output line is a JSON object with ``content``, ``index`` (position within its source),
``tokens``, and ``position`` (a dict with ``start``, ``end``, ``line``, ``column``,
``end_line``, ``end_column``). When more than one input is chunked, each record also
includes a ``source`` key naming its file. A summary of how many chunks were written is
printed to stderr, so it does not pollute the JSONL on stdout.

.. code-block:: console

   $ echo "First sentence. Second sentence." | lvdb chunk - --method sentences
   {"content": "First sentence.", "index": 0, "tokens": 2, "position": {"start": 0, "end": 15, "line": 1, "column": 0, "end_line": 1, "end_column": 15}}
   {"content": "Second sentence.", "index": 1, "tokens": 2, "position": {"start": 16, "end": 32, "line": 1, "column": 16, "end_line": 1, "end_column": 32}}
   Wrote 2 chunk(s) from 1 input(s)

Version
-------

.. code-block:: bash

   # Print the installed localvectordb version
   lvdb version

   # The top-level --version / -V flag prints the same value
   lvdb --version

Advanced Usage Examples
-----------------------

Bulk Operations
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Add all markdown files in a directory tree
   lvdb db papers add "documents/**/*.md"

   # Add with structured metadata from JSON
   lvdb db papers add papers.txt --metadata metadata.json --id ids.txt

   # Search and save results for further processing
   lvdb db papers search "deep learning" --format json --output dl_papers.json

   # Batch process search results
   for query in "neural networks" "machine learning" "computer vision"; do
     lvdb db papers search "$query" --limit 10 --output "results_${query// /_}.txt"
   done

Configuration and Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create production configuration
   lvdb config init --format toml --output production.toml

   # Start production server (--config is global; it precedes the subcommand)
   lvdb --config production.toml serve --host 0.0.0.0 --port 8080

   # Create backup-friendly database (--db-folder is global; it precedes the subcommand)
   lvdb --db-folder /backup/vector_dbs create backup_db

   # Monitor database growth
   watch 'lvdb list --details'

Pipeline Integration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   #!/bin/bash
   # Document processing pipeline

   # Create database if not exists.
   # NOTE: `lvdb create --metadata-schema` accepts: documents, research_papers,
   # code_repository, customer_support. The file-oriented `files` schema
   # (file_path, created_at, last_modified, file_size_bytes, ...) is only available
   # via `lvdb config init --schema files`; create a config with it first if you need
   # those fields, since metadata not present in the schema is ignored on insert.
   lvdb create research_pipeline --embedding-model nomic-embed-text --metadata-schema documents

   # Process incoming documents
   find /incoming/documents -name "*.txt" -newer /tmp/last_processed | while IFS= read -r file; do
      echo "Processing: $file"

      # Extract file metadata using stat
      last_modified=$(stat -c %Y "$file")
      file_size_bytes=$(stat -c %s "$file")
      created_at=$(stat -c %w "$file")

      # If creation time is unavailable (returns '-'), use the last modification time
      if [ "$created_at" = "-" ]; then
        created_at="$last_modified"
      fi

      # Construct JSON metadata string
      metadata=$(printf '{"file_path": "%s", "last_modified": "%s", "file_size_bytes": "%s", "created_at": "%s"}' \
        "$file" "$last_modified" "$file_size_bytes" "$created_at")

      # Add record using lvdb add command with metadata
      lvdb db research_pipeline add "$file" --metadata "$metadata"
   done

   # Update timestamp
   touch /tmp/last_processed

   # Search and generate report
   lvdb db research_pipeline search "quarterly report" --format json > quarterly_matches.json

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Database not found**:

.. code-block:: bash

   # Check available databases
   lvdb list

   # Verify database folder
   ls -la ~/.lvdb/

   # Create database if missing
   lvdb create missing_database

**Ollama connection errors**:

.. code-block:: bash

   # Check Ollama status
   ollama list

   # Start Ollama if needed
   ollama serve

   # Test with different model
   lvdb create test_db --embedding-model all-minilm

**Permission errors**:

.. code-block:: bash

   # Check database folder permissions
   ls -la ~/.lvdb/

   # Fix permissions
   chmod 755 ~/.lvdb/
   chmod 644 ~/.lvdb/*.sqlite

**Configuration issues**:

.. code-block:: bash

   # Validate configuration
   lvdb config show

   # Reset to defaults
   mv server-cfg.toml server-cfg.toml.backup
   lvdb config init

Debug Mode
^^^^^^^^^^

.. code-block:: bash

   # Run server in debug mode
   lvdb serve --debug --log-level DEBUG

   # Test with verbose output (--verbose is a global option, before the subcommand)
   lvdb --verbose db test_database add test.txt

   # Check server logs
   tail -f server.log

Backup and Restore Operations
-----------------------------

The LocalVectorDB CLI provides comprehensive backup and restore capabilities for database protection and disaster recovery.

Create Backups
^^^^^^^^^^^^^^

.. code-block:: bash

   # Create full backup
   lvdb backup create my_database --type full

   # Create incremental backup (requires the parent backup ID)
   lvdb backup create my_database --type incremental --parent backup-abc123

   # Create backup with compression
   lvdb backup create my_database --type full --compression gzip

   # Create backup with custom location
   lvdb backup create my_database --type full --location /backups/localvectordb

   # Create backup, skipping the FAISS index and integrity check
   lvdb backup create my_database --type full --exclude-faiss --no-verify

**Options**:

- ``--type, -t``: Backup type (full, incremental; default ``full``)
- ``--parent, -p``: Parent backup ID (required for incremental backups)
- ``--location, -l``: Backup storage location (default ``./backups``)
- ``--compression, -c``: Compression algorithm (none, gzip, lzma, bzip2; default ``gzip``)
- ``--no-verify``: Skip backup integrity verification after creation
- ``--exclude-faiss``: Exclude the FAISS index from the backup
- ``--format, -f``: Output format (``table`` (default) or ``json``); ``-j`` is a shortcut for ``--format json``

List Backups
^^^^^^^^^^^^^

.. code-block:: bash

   # List all backups
   lvdb backup list

   # List backups for specific database
   lvdb backup list --database my_database

   # Filter by backup type and limit the number shown
   lvdb backup list --type full --limit 10

   # Scan a specific backup location, in JSON format
   lvdb backup list --location /backups/localvectordb --format json

**Options**:

- ``--database, -d``: Filter backups for a specific database
- ``--type, -t``: Filter by backup type (full, incremental)
- ``--limit, -n``: Limit the number of backups shown
- ``--location, -l``: Backup storage location to scan (default ``./backups``)
- ``--format, -f``: Output format (``table`` (default) or ``json``); ``-j`` is a shortcut for ``--format json``

**Example Output**:

.. code-block:: console

   Backups for database: my_database
   =====================================

   ID                    Type        Created              Size      Status    Description
   -----------------------------------------------------------------------------------------
   backup_20241201_001   FULL        2024-12-01 10:30:00  125.4 MB  VALID     Pre-migration backup
   backup_20241201_002   INCREMENTAL 2024-12-01 15:45:00  8.2 MB    VALID     After document updates
   backup_20241202_001   FULL        2024-12-02 09:00:00  127.1 MB  VALID     Daily backup

Restore Backups
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Restore a backup to its original location (partial backup IDs are accepted)
   lvdb backup restore abc12345

   # Restore into a specific directory
   lvdb backup restore backup_20241201_001 --to-location ./restored_database

   # Overwrite existing files without confirmation
   lvdb backup restore backup_20241201_001 --to-location ./restored --overwrite

   # Restore from a specific backup store, JSON output
   lvdb backup restore abc12345 --location /backups/localvectordb --format json

**Options**:

- ``--to-location, -t``: Directory to restore to (default: the backup's original location)
- ``--overwrite``: Overwrite existing files without confirmation
- ``--location, -l``: Backup storage location to read from (default ``./backups``)
- ``--format, -f``: Output format (``table`` (default) or ``json``); ``-j`` is a shortcut for ``--format json``

The ``BACKUP_ID`` argument accepts a partial ID; the first matching backup file is used.
For incremental backups the full backup chain is located and applied automatically.

Verify Backups
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Verify a specific backup (partial backup IDs are accepted)
   lvdb backup verify abc12345

   # Verify a backup in a specific store, JSON output
   lvdb backup verify backup_20241201_001 --location /backups/localvectordb --format json

**Options**:

- ``--location, -l``: Backup storage location to read from (default ``./backups``)
- ``--format, -f``: Output format (``table`` (default) or ``json``); ``-j`` is a shortcut for ``--format json``

**Example Output**:

.. code-block:: console

   Verifying backup: backup_20241201_001
   =====================================

   ✓ Backup metadata valid
   ✓ File integrity checksums match
   ✓ Database schema valid
   ✓ FAISS index accessible
   ✓ SQLite database accessible

   Backup verification: PASSED

Clean Up Backups
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Clean up backups older than 30 days (the default), keeping at least 3 full backups
   lvdb backup cleanup

   # Delete backups older than 7 days
   lvdb backup cleanup --older-than 7

   # Keep at least 5 full backups, previewing what would be deleted
   lvdb backup cleanup --keep-full 5 --dry-run

   # Clean up a specific backup store, JSON output
   lvdb backup cleanup --location /backups/localvectordb --format json

**Options**:

- ``--older-than``: Delete backups older than N days (default ``30``)
- ``--keep-full``: Minimum number of full backups to keep (default ``3``)
- ``--location, -l``: Backup storage location (default ``./backups``)
- ``--dry-run``: Show what would be deleted without actually deleting
- ``--format, -f``: Output format (``table`` (default) or ``json``); ``-j`` is a shortcut for ``--format json``

``cleanup`` operates on all backups in the store (it takes no database argument) while
maintaining backup-chain integrity and keeping a minimum number of full backups.

Point-in-Time Recovery
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Restore database to a specific point in time (--to-location is required)
   lvdb backup pitr "2024-12-01 14:30:00" --to-location ./pitr_restored

   # ISO-8601 timestamp with a wider search tolerance, validating without restoring
   lvdb backup pitr "2024-12-01T14:30:00Z" --to-location ./restored --tolerance 120 --dry-run

   # Read from a specific backup store, JSON output
   lvdb backup pitr "2024-12-01 14:30:00" --to-location ./restored --location /backups/localvectordb --format json

**Options**:

- ``--to-location, -t``: Directory to restore to (**required**)
- ``--tolerance``: Tolerance in minutes for finding a recovery point (default ``60``)
- ``--location, -l``: Backup storage location (default ``./backups``)
- ``--dry-run``: Validate recovery without actually restoring
- ``--format, -f``: Output format (``table`` (default) or ``json``); ``-j`` is a shortcut for ``--format json``

The ``TIMESTAMP`` argument accepts ``YYYY-MM-DD HH:MM:SS`` or ISO-8601 format. The database
name is read from the backup store, so it is not passed on the command line.

**Example PITR Session**:

.. code-block:: console

   $ lvdb backup pitr "2024-12-01 14:30:00" --to-location ./recovered

   Point-in-Time Recovery
   ======================
   Target time: 2024-12-01 14:30:00 UTC
   Database: research_papers
   Restore location: ./recovered

   Recovery plan:
     Base backup: backup_20241201_001 (2024-12-01 10:00:00)
     Incremental backups: 2
       - backup_20241201_inc_001 (2024-12-01 12:00:00)
       - backup_20241201_inc_002 (2024-12-01 14:15:00)

   Proceed with recovery? [y/N]: y

   Restoring base backup...
   Applying incremental backup 1...
   Applying incremental backup 2...
   Stopping at target time: 2024-12-01 14:30:00

   ✓ Point-in-time recovery completed successfully!

Database Migration and Schema Evolution
----------------------------------------

The LocalVectorDB CLI provides powerful migration capabilities for evolving database schemas and managing version upgrades.

Migration Status
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Check migration status for database
   lvdb migrate status my_database

   # Use a custom migrations directory
   lvdb migrate status my_database --migrations-dir ./custom_migrations

   # JSON output for automation
   lvdb migrate status my_database --format json

**Options**:

- ``--migrations-dir, -m``: Directory containing migration files (default ``./migrations``)
- ``--format, -f``: Output format (``table`` (default) or ``json``); ``-j`` is a shortcut for ``--format json``

**Example Output**:

.. code-block:: console

   Migration Status: my_database
   =============================

   Current Version: 1.1.0
   Target Version: 1.2.0
   Pending Migrations: 2

   Available Migrations:
   ---------------------
   Version   Status    Description                    Created
   ---------------------------------------------------------------
   1.1.1     PENDING   Add citation_count field       2024-11-15
   1.2.0     PENDING   Add full-text search indices   2024-11-20

   Migration Path:
     1.1.0 → 1.1.1 → 1.2.0

Apply Migrations
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Apply all pending migrations (a backup is created first by default)
   lvdb migrate apply my_database

   # Apply migrations up to a specific version
   lvdb migrate apply my_database --to-version 1.2.0

   # Apply without creating a pre-migration backup
   lvdb migrate apply my_database --no-backup

   # Dry run to preview changes without a backup
   lvdb migrate apply my_database --dry-run --no-backup

   # Store the pre-migration backup in a custom location
   lvdb migrate apply my_database --backup-location /backups/premigration

**Options**:

- ``--to-version``: Target version to migrate to (default: latest)
- ``--migrations-dir, -m``: Directory containing migration files (default ``./migrations``)
- ``--backup/--no-backup``: Create a backup before migrating (default: enabled)
- ``--backup-location, -b``: Backup storage location (default ``./backups``)
- ``--dry-run``: Validate migrations without applying them
- ``--format, -f``: Output format (``table`` (default) or ``json``); ``-j`` is a shortcut for ``--format json``

**Example Migration Session**:

.. code-block:: console

   $ lvdb migrate apply research_papers --backup

   Migration Plan: research_papers
   ===============================

   Current Version: 1.1.0
   Target Version: 1.2.0

   Migrations to Apply:
     1. v1.1.1: Add citation_count field
     2. v1.2.0: Add full-text search indices

   Creating backup before migration...
   ✓ Backup created: backup_20241201_premigration

   Apply these migrations? [y/N]: y

   Applying migration v1.1.1...
   ✓ Added citation_count field (default: 0)
   ✓ Updated 1,247 existing records

   Applying migration v1.2.0...
   ✓ Created full-text search indices
   ✓ Rebuilt search index

   ✓ Migration completed successfully!
   Database version: 1.1.0 → 1.2.0

Rollback Migrations
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Rollback to a specific version (both database name and target version are required)
   lvdb migrate rollback my_database 1.1.0

   # Rollback without creating a pre-rollback backup
   lvdb migrate rollback my_database 1.0.0 --no-backup

   # Validate a rollback without applying it
   lvdb migrate rollback my_database 1.1.0 --dry-run

**Options**:

- ``--migrations-dir, -m``: Directory containing migration files (default ``./migrations``)
- ``--backup/--no-backup``: Create a backup before rolling back (default: enabled)
- ``--backup-location, -b``: Backup storage location (default ``./backups``)
- ``--dry-run``: Validate the rollback without applying it
- ``--format, -f``: Output format (``table`` (default) or ``json``); ``-j`` is a shortcut for ``--format json``

Both the ``DATABASE_NAME`` and ``TARGET_VERSION`` positional arguments are required.

Create Migration Templates
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create a new migration template (the description is the positional argument)
   lvdb migrate create "add priority field" --version 1.3.0

   # Create a schema-change template
   lvdb migrate create "add user fields" --version 1.3.1 --template schema

   # Create a data-transformation template in a specific directory
   lvdb migrate create "migrate old data format" --version 1.4.0 --template data --migrations-dir ./migrations

**Options**:

- ``--version``: Version number for the migration, e.g. ``1.2.0`` (**required**)
- ``--migrations-dir, -m``: Directory to create the migration in (default ``./migrations``)
- ``--template, -t``: Template type — ``basic`` (default), ``schema``, or ``data``
- ``--format, -f``: Output format (``table`` (default) or ``json``); ``-j`` is a shortcut for ``--format json``

**Example Generated Migration**:

Migrations subclass :class:`~localvectordb.migration.Migration` and describe schema
changes by returning a ``new_schema`` mapping from ``get_schema_changes()`` /
``get_rollback_changes()`` (there are no ``up``/``down`` methods). The ``schema`` template
produces:

.. code-block:: python

   # Migration: add user fields
   # Version: 1.3.1
   # Created: 2024-12-01 10:30:00

   from typing import Dict, Any
   from localvectordb.migration import Migration
   from localvectordb.core import MetadataField, MetadataFieldType


   class Migration_1_3_1(Migration):
       """
       add user fields
       """

       version = "1.3.1"
       description = "add user fields"
       dependencies = []  # Add version dependencies here

       def get_schema_changes(self) -> Dict[str, Any]:
           """Get schema changes to apply in forward migration."""

           # Define the new complete metadata schema after this migration
           new_schema = {
               'user_id': MetadataField(
                   type=MetadataFieldType.TEXT,
                   indexed=True,
                   required=False,
                   default_value=None,
               ),
               'priority': MetadataField(
                   type=MetadataFieldType.INTEGER,
                   indexed=True,
                   required=False,
                   default_value=0,
               ),
               # Add existing fields that should remain...
           }

           return {
               'new_schema': new_schema,
               'column_mapping': {},  # Optional: rename columns {'old_name': 'new_name'}
               'drop_columns': False,  # Whether to drop unused columns
           }

       def get_rollback_changes(self) -> Dict[str, Any]:
           """Get schema changes to apply for rollback."""

           rollback_schema = {
               # Define schema without the changes from this migration
           }

           return {
               'new_schema': rollback_schema,
               'column_mapping': {},
               'drop_columns': False,
           }

List Migrations
^^^^^^^^^^^^^^^

.. code-block:: bash

   # List all discovered migration files
   lvdb migrate list

   # Show each migration's dependencies, JSON output
   lvdb migrate list --show-dependencies --format json

   # List migrations from a custom directory
   lvdb migrate list --migrations-dir ./custom_migrations

**Options**:

- ``--migrations-dir, -m``: Directory containing migration files (default ``./migrations``)
- ``--show-dependencies, -d``: Show migration dependencies
- ``--format, -f``: Output format (``table`` (default) or ``json``); ``-j`` is a shortcut for ``--format json``

``list`` scans the migrations directory itself, so it takes no database argument.

**Example Output**:

.. code-block:: console

   Available Migrations in migrations:

   Version   Description                     File                        Dependencies
   -------------------------------------------------------------------------------------
   1.1.1     Add citation_count field        v1_1_1_add_citation.py      -
   1.2.0     Add full-text search indices    v1_2_0_add_fts.py           1.1.1
   1.3.0     Add priority field              v1_3_0_add_priority.py      1.2.0

   Total: 3 migration(s)

SQLite Performance Tuning
-------------------------

The ``lvdb tuning`` command group manages SQLite performance profiles and pragma
settings for individual databases. Unlike most commands, ``tuning`` does not require a
configuration file; point it at your databases with the global ``--db-folder, -d``
option (given before the subcommand), e.g. ``lvdb --db-folder ./data tuning get mydb``.

List Tuning Profiles
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # List available SQLite tuning profiles with descriptions
   lvdb tuning list

``list`` takes no arguments or options.

Show Tuning Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Show the current tuning profile, overrides, and active pragmas for a database
   lvdb --db-folder ./data tuning get my_database

   # JSON output
   lvdb --db-folder ./data tuning get my_database --format json

**Options**:

- ``--format, -f``: Output format (``table`` (default), ``json``)

Apply a Tuning Profile
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Apply a named profile to a database
   lvdb --db-folder ./data tuning set my_database read_optimized

   # Apply a profile with pragma overrides (repeat --override per pragma)
   lvdb --db-folder ./data tuning set my_database balanced \
     --override cache_size=-64000 --override mmap_size=268435456

   # Apply without persisting the settings to the database
   lvdb --db-folder ./data tuning set my_database read_optimized --no-persist

   # Preview what would be applied without changing anything
   lvdb --db-folder ./data tuning set my_database read_optimized --dry-run

**Options**:

- ``--override, -o KEY=VALUE``: Pragma override (may be given multiple times)
- ``--no-persist``: Do not persist the settings to the database
- ``--dry-run``: Show what would be applied without applying it

Set a Single Pragma
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Set a single pragma override, keeping the current profile
   lvdb --db-folder ./data tuning set-pragma my_database cache_size -64000

**Options**:

- ``--no-persist``: Do not persist the setting to the database

``set-pragma`` takes three positional arguments: ``DATABASE``, ``PRAGMA_KEY`` and
``PRAGMA_VALUE``.

Auto-Tuning Recommendations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Get a recommendation from an interactive workload interview
   lvdb --db-folder ./data tuning auto my_database --interactive

   # Get a recommendation from explicit workload parameters
   lvdb --db-folder ./data tuning auto my_database \
     --workload-type read_heavy --memory-constraint generous --durability normal

   # Apply the recommended settings immediately, JSON output
   lvdb --db-folder ./data tuning auto my_database --workload-type balanced --apply --format json

**Options**:

- ``--interactive, -i``: Run an interactive workload interview
- ``--workload-type``: Workload type (``read_heavy``, ``write_heavy``, ``balanced``,
  ``batch_ingest``, ``real_time``) — skips the interview
- ``--memory-constraint``: Memory availability (``generous``, ``moderate``, ``limited``)
- ``--durability``: Data durability importance (``critical``, ``high``, ``normal``, ``low``)
- ``--apply``: Apply the recommended settings immediately
- ``--format, -f``: Output format (``table`` (default), ``json``)

Database Maintenance
--------------------

The ``lvdb maintenance`` command group runs SQLite maintenance operations on a database.
Like ``tuning``, it does not require a configuration file; select the database directory
with the global ``--db-folder, -d`` option before the subcommand.

WAL Checkpoint
^^^^^^^^^^^^^^

.. code-block:: bash

   # Run a WAL checkpoint (default mode PASSIVE)
   lvdb --db-folder ./data maintenance checkpoint my_database

   # Run a TRUNCATE checkpoint
   lvdb --db-folder ./data maintenance checkpoint my_database --mode TRUNCATE

**Options**:

- ``--mode, -m``: Checkpoint mode (``PASSIVE`` (default), ``FULL``, ``RESTART``, ``TRUNCATE``)

Optimize
^^^^^^^^

.. code-block:: bash

   # Run PRAGMA optimize on a database
   lvdb --db-folder ./data maintenance optimize my_database

``optimize`` takes only the ``DATABASE`` argument.

Vacuum
^^^^^^

.. code-block:: bash

   # Full VACUUM (prompts for confirmation; requires exclusive access)
   lvdb --db-folder ./data maintenance vacuum my_database

   # Skip the confirmation prompt
   lvdb --db-folder ./data maintenance vacuum my_database --confirm

   # Incremental vacuum, reclaiming a given number of pages
   lvdb --db-folder ./data maintenance vacuum my_database --incremental --pages 5000

**Options**:

- ``--incremental, -i``: Run an incremental vacuum instead of a full VACUUM
- ``--pages``: Pages to reclaim (incremental only; default ``2000``)
- ``--confirm``: Skip the confirmation prompt (full VACUUM only)

Analyze System Resources
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Analyze system RAM, CPU, and disk for tuning recommendations
   lvdb maintenance analyze-system

   # JSON output
   lvdb maintenance analyze-system --format json

**Options**:

- ``--format, -f``: Output format (``table`` (default), ``json``)

``analyze-system`` inspects the host machine and takes no database argument.

MCP Server Integration
----------------------

The LocalVectorDB CLI provides Model Context Protocol (MCP) server capabilities for integration with AI assistants and tools.

Start MCP Server
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Start MCP server in read-only mode (default)
   lvdb mcp serve

   # Start in read-write mode
   lvdb mcp serve --mode read-write

   # Start with specific configuration
   lvdb mcp serve --config /path/to/mcp-config.toml

   # Start with custom database root
   lvdb mcp serve --databases-root /data/vector_databases

   # Start with database mapping
   lvdb mcp serve --databases-map '{"papers": "/data/research", "docs": "http://remote-server:8000/docs"}'

   # Start with debug logging
   lvdb mcp serve --log-level DEBUG

**Options**:

- ``--mode``: Server mode (read-only, read-write)
- ``--config``: Path to MCP configuration file (TOML format)
- ``--databases-root``: Root directory for local databases
- ``--databases-map``: JSON mapping of database names to paths/URLs
- ``--log-level``: Logging level (DEBUG, INFO, WARNING, ERROR)

**MCP Configuration File Example**:

Server-level settings (including the read-only/read-write ``mode``) live under
``[mcp]``. Databases are configured under ``[databases]`` with a ``root`` directory
and an optional ``[databases.map]`` table mapping database names to specific local
paths or remote URLs. Run ``lvdb mcp config-example`` to print a fully-commented
template. See :doc:`mcp` for the complete configuration reference.

.. code-block:: toml

   # mcp-config.toml
   [mcp]
   mode = "read-write"   # or "read-only" (read-only is a whole-server setting)
   log_level = "INFO"

   [databases]
   # Root directory searched for local databases
   root = "/data/databases"

   # Optional: map specific names to local paths or remote URLs
   [databases.map]
   research = "/data/research_papers"
   documentation = "/data/docs"
   shared_knowledge = "http://knowledge-server:8000/shared"

**Claude Desktop Integration**:

To use with Claude Desktop, add this to your Claude Desktop configuration:

.. code-block:: json

   {
     "mcpServers": {
       "localvectordb": {
         "command": "lvdb",
         "args": ["mcp", "serve", "--mode", "read-write"],
         "env": {
           "LVDB_MCP_CONFIG": "/path/to/your/mcp-config.toml"
         }
       }
     }
   }

**Available MCP Tools**:

When running as an MCP server, LocalVectorDB provides these tools to AI assistants.
Read-only tools are available in every mode; write tools require ``--mode read-write``.
See :doc:`mcp` for the full parameter reference.

Read-only tools:

- ``list_databases``: List available databases
- ``get_database_info``: Get database statistics and configuration
- ``query_database``: Search with vector, keyword, or hybrid search
- ``find_related_documents``: Find documents related to a given document (nearest neighbours)
- ``filter_documents``: Filter documents by metadata (with limit/offset)
- ``get_document``: Retrieve a document by ID, or a portion of it (chunk/range/lines/section/outline)
- ``check_documents_exist``: Check whether documents exist
- ``get_metadata_schema``: Get a database's metadata schema
- ``get_system_info``: Get version, configuration, and enabled tools

Write tools (read-write mode only):

- ``create_database``: Create a new database
- ``delete_database``: Delete a database and its data
- ``upsert_documents``: Insert or update documents
- ``update_document``: Update a document's content and/or metadata
- ``delete_document``: Delete a document by ID
- ``update_metadata_schema``: Add or modify metadata schema fields
