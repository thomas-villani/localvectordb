Command-Line Interface
======================

The LocalVectorDB CLI provides comprehensive tools for database management, document operations, and server administration.

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

   # Custom configuration
   lvdb serve --host 0.0.0.0 --port 8080 --config production.toml

   # Development mode
   lvdb serve --debug --log-level DEBUG

   # Specific database folder
   lvdb serve --db-folder /data/vector_databases

   # Disable Ollama check
   lvdb serve --disable-ollama-check

**Options**:

- ``--host, -h``: Interface to bind to (default: 127.0.0.1)
- ``--port, -p``: Port to bind to (default: 5000)
- ``--debug``: Enable Flask debug mode
- ``--config, -c``: Path to configuration file
- ``--db-folder, -d``: Database directory path
- ``--log-level, -l``: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- ``--disable-ollama-check, -x``: Skip Ollama availability check

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

   # Show from specific file
   lvdb config show --config /path/to/config.toml

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
   port = 5000
   log_level = "INFO"
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   max_request_size = 104857600
   request_timeout = 300

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
- ``--force, -f``: Skip confirmation prompt

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
- ``server.request_timeout`` - Request timeout

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
   lvdb auth status --output json

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
   lvdb auth create-key --description "Script Access" --permission-level read_only --output json

   # Key-only output for automation
   lvdb auth create-key --permission-level read_write --output key-only

**Options**:

- ``--description, -d``: Human-readable description of the key's purpose
- ``--permission-level, -p``: Permission level (read_only, read_write) - defaults to read_write
- ``--expires-days``: Number of days until key expires (omit for no expiration)
- ``--created-by``: Identifier of who is creating the key
- ``--output, -o``: Output format (table, json, key-only)

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
   lvdb auth list-keys --output json

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
   lvdb auth key-info key_20241201_abc123 --output json

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
   lvdb auth rotate-key key_20241201_abc123 --output json

   # Key-only output
   lvdb auth rotate-key key_20241201_abc123 --output key-only

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
        -X POST http://localhost:5000/api/v1/my_database/query \
        -d '{"query": "machine learning", "search_type": "vector", "k": 5}'

**Python Example**:

.. code-block:: python

   import requests

   headers = {
       "Authorization": "Bearer lvdb_XyZ9k2mN7qP4wR8tL3vB5nM9kJ7hF2dS6xW1qE4yT8rL9pN3mK5",
       "Content-Type": "application/json"
   }

   response = requests.post(
       "http://localhost:5000/api/v1/my_database/query",
       headers=headers,
       json={"query": "neural networks", "search_type": "vector", "k": 10}
   )

**Environment Variable**:

.. code-block:: bash

   # Set as environment variable
   export LVDB_API_KEY="lvdb_XyZ9k2mN7qP4wR8tL3vB5nM9kJ7hF2dS6xW1qE4yT8rL9pN3mK5"

   # Use in application
   curl -H "Authorization: Bearer $LVDB_API_KEY" \
        -X GET http://localhost:5000/api/v1/my_database/info

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

   # Specify database folder
   lvdb list --folder /path/to/databases

**Options:**
- ``-c, --config``: Path to the config file for the server
- ``-f, --folder``: Optionally provide the explicit path containing databases
- ``-d, --details``: Show document/chunk count, model, and chunk method for each database

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
     --chunking-method paragraphs \
     --metadata-schema research_papers

   # With specific provider
   lvdb create openai_db \
     --embedding-provider openai \
     --embedding-model text-embedding-3-small

**Options**:

- ``--embedding-model``: Embedding model to use
- ``--embedding-provider``: Provider (ollama, openai)
- ``--chunk-size``: Maximum tokens per chunk
- ``--chunking-method``: Method (sentences, paragraphs, tokens, etc.)
- ``--chunk-overlap``: Overlap between chunks
- ``--metadata-schema``: Predefined schema (documents, research_papers, etc.)

Delete Database
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Delete database (with confirmation)
   lvdb delete old_database

   # Force delete without confirmation
   lvdb delete old_database --confirm

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
   lvdb db my_database get doc_1 --json

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
   lvdb db my_database list --json

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

   # Search with metadata filter
   lvdb db my_database search "research" --metadata-filter '{"journal": "Science"}'

   # Minimum score threshold
   lvdb db my_database search "neural" --score-threshold 0.8

   # Pretty output
   lvdb db my_database search "machine learning" --pretty

   # Save results
   lvdb db my_database search "AI" --output search_results.txt

   # JSON output
   lvdb db my_database search "algorithms" --json --metadata

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
     --schema-string '{"title": "text", "author": "text", "year": "integer"}'

   # Dry run to preview changes
   lvdb db my_database schema update --schema new_schema.json --dry-run

   # Skip confirmation prompts
   lvdb db my_database schema update --schema new_schema.json --force

   # Verbose output with detailed progress
   lvdb db my_database schema update --schema new_schema.json --verbose

   # Column mapping from file
   lvdb db my_database schema update \
     --schema new_schema.json \
     --mapping-file column_mappings.json

**Options**:

- ``--schema, -s``: JSON string or path to JSON file containing new schema definition
- ``--mapping, -m``: Column mapping as JSON string or path to JSON file (old_name: new_name)
- ``--drop-columns, --drop``: Actually drop removed columns (WARNING: data loss)
- ``--dry-run, --dry``: Show what would be changed without making changes
- ``--force, -f``: Skip confirmation prompts
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
     --schema-string '{"title": "text", "author": "text", "year": "integer"}' \
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
   lvdb db my_database schema update --schema new_schema.json --mapping-file mappings.json --dry-run

   # 3. Apply changes if satisfied
   lvdb db my_database schema update --schema new_schema.json --mapping-file mappings.json --verbose

   # 4. Verify results
   lvdb db my_database schema show table
   lvdb db my_database search "test query" 3

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
   lvdb db "${DB_NAME}_test" search "test query" 5 > /dev/null

   if [ $? -ne 0 ]; then
       echo "Search test failed. Aborting."
       rm -rf "${DB_NAME}_test"
       exit 1
   fi

   # 4. Apply to production
   echo "Applying to production..."
   lvdb db $DB_NAME schema update --schema $SCHEMA_FILE --mapping-file $MAPPING_FILE --force

   # 5. Verify production update
   echo "Verifying production update..."
   lvdb db $DB_NAME schema show table
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
   lvdb db prod search "query" --json > results.json

   # One-time operations
   lvdb db prod add important_doc.txt

   # Production deployments
   lvdb db prod schema update --schema prod_schema.json --force

This enhanced interactive shell provides a powerful environment for both day-to-day database operations and complex
schema evolution tasks, making it easy to iterate, test, and deploy changes safely.


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
   lvdb db papers search "deep learning" --json --output dl_papers.json

   # Batch process search results
   for query in "neural networks" "machine learning" "computer vision"; do
     lvdb db papers search "$query" --limit 10 --output "results_${query// /_}.txt"
   done

Configuration and Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create production configuration
   lvdb config init --format toml --output production.toml

   # Start production server
   lvdb serve --config production.toml --host 0.0.0.0 --port 8080

   # Create backup-friendly database
   lvdb create backup_db --db-folder /backup/vector_dbs

   # Monitor database growth
   watch 'lvdb list --details'

Pipeline Integration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   #!/bin/bash
   # Document processing pipeline

   # Create database if not exists. Add 'documents' metadata schema so we can add file info
   lvdb create research_pipeline --embedding-model nomic-embed-text --metadata-schema files

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
   lvdb db research_pipeline search "quarterly report" --json > quarterly_matches.json

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

   # Test with verbose output
   lvdb db test_database add test.txt --verbose

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

   # Create incremental backup
   lvdb backup create my_database --type incremental

   # Create backup with compression
   lvdb backup create my_database --type full --compression gzip

   # Create backup with custom location
   lvdb backup create my_database --type full --output-dir /backups/localvectordb

   # Create backup with description
   lvdb backup create my_database --type full --description "Pre-migration backup"

**Options**:

- ``--type, -t``: Backup type (full, incremental)
- ``--compression, -c``: Compression algorithm (none, gzip, lzma, zstd)
- ``--output-dir, -o``: Directory to store backup files
- ``--description, -d``: Description for the backup
- ``--verify``: Verify backup integrity after creation

List Backups
^^^^^^^^^^^^^

.. code-block:: bash

   # List all backups
   lvdb backup list

   # List backups for specific database
   lvdb backup list --database my_database

   # List with detailed information
   lvdb backup list --details

   # List in JSON format
   lvdb backup list --format json

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

   # Restore specific backup
   lvdb backup restore backup_20241201_001 --to-location ./restored_database

   # Restore latest backup for database
   lvdb backup restore --database my_database --latest --to-location ./restored

   # Restore with verification
   lvdb backup restore backup_20241201_001 --to-location ./restored --verify

   # Force restore (overwrite existing)
   lvdb backup restore backup_20241201_001 --to-location ./restored --force

**Options**:

- ``--to-location, -l``: Directory where database should be restored
- ``--database, -d``: Database name (when using --latest)
- ``--latest``: Restore the most recent backup
- ``--verify``: Verify backup integrity before restoration
- ``--force, -f``: Overwrite existing database at restore location

Verify Backups
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Verify specific backup
   lvdb backup verify backup_20241201_001

   # Verify all backups for database
   lvdb backup verify --database my_database

   # Verify and repair if possible
   lvdb backup verify backup_20241201_001 --repair

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

   # Clean up old backups (interactive)
   lvdb backup cleanup --database my_database

   # Keep only last 5 backups
   lvdb backup cleanup --database my_database --keep 5

   # Clean up backups older than 30 days
   lvdb backup cleanup --database my_database --older-than 30

   # Clean up with confirmation
   lvdb backup cleanup --database my_database --keep 3 --confirm

**Options**:

- ``--database, -d``: Database name to clean up backups for
- ``--keep, -k``: Number of most recent backups to keep
- ``--older-than, -o``: Remove backups older than specified days
- ``--confirm, -y``: Skip confirmation prompts
- ``--dry-run, -n``: Show what would be removed without deleting

Point-in-Time Recovery
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Restore database to specific point in time
   lvdb backup pitr "2024-12-01 14:30:00" --database my_database --to-location ./pitr_restored

   # Point-in-time recovery with timezone
   lvdb backup pitr "2024-12-01 14:30:00 UTC" --database my_database --to-location ./restored

   # List available recovery points
   lvdb backup pitr --list --database my_database

**Options**:

- ``--database, -d``: Database name for recovery
- ``--to-location, -l``: Directory where database should be restored
- ``--list``: Show available recovery points
- ``--verify``: Verify backup chain before recovery

**Example PITR Session**:

.. code-block:: console

   $ lvdb backup pitr "2024-12-01 14:30:00" --database research_papers --to-location ./recovered

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

   # Show detailed migration information
   lvdb migrate status my_database --details

   # JSON output for automation
   lvdb migrate status my_database --format json

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

   # Apply all pending migrations
   lvdb migrate apply my_database

   # Apply migrations to specific version
   lvdb migrate apply my_database --to-version 1.2.0

   # Apply with automatic backup
   lvdb migrate apply my_database --backup

   # Dry run to preview changes
   lvdb migrate apply my_database --dry-run

   # Force apply without confirmations
   lvdb migrate apply my_database --force

**Options**:

- ``--to-version, -v``: Target version to migrate to
- ``--backup, -b``: Create backup before applying migrations
- ``--dry-run, -n``: Show what would be changed without applying
- ``--force, -f``: Skip confirmation prompts
- ``--rollback-on-error``: Automatically rollback on migration failure

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

   # Rollback to previous version
   lvdb migrate rollback my_database

   # Rollback to specific version
   lvdb migrate rollback my_database 1.1.0

   # Rollback with backup
   lvdb migrate rollback my_database 1.1.0 --backup

   # Force rollback without confirmation
   lvdb migrate rollback my_database 1.1.0 --force

**Options**:

- ``--backup, -b``: Create backup before rollback
- ``--force, -f``: Skip confirmation prompts
- ``--verify``: Verify database integrity after rollback

Create Migration Templates
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create new migration template
   lvdb migrate create "add priority field" --version 1.3.0

   # Create migration with description
   lvdb migrate create "optimize search performance" --version 1.3.1 --description "Add database indices for faster queries"

   # Create migration in specific directory
   lvdb migrate create "schema update" --version 1.4.0 --output-dir ./migrations

**Example Generated Migration**:

.. code-block:: python

   """
   Migration: Add priority field
   Version: 1.3.0
   Created: 2024-12-01 10:30:00
   """

   from localvectordb.migration import Migration, MigrationStep
   from localvectordb.core import MetadataField

   class AddPriorityFieldMigration(Migration):
       version = "1.3.0"
       description = "Add priority field"
       
       def up(self, db):
           # Add priority field
           priority_field = MetadataField(
               name="priority",
               field_type="integer",
               indexed=True,
               default_value=1
           )
           db.schema.add_field(priority_field)
       
       def down(self, db):
           # Remove priority field
           db.schema.remove_field("priority")

List Migrations
^^^^^^^^^^^^^^^

.. code-block:: bash

   # List all available migrations
   lvdb migrate list

   # List migrations for specific database
   lvdb migrate list --database my_database

   # Show migration details
   lvdb migrate list --details

**Example Output**:

.. code-block:: console

   Available Migrations
   ====================

   Version   Status      Database        Description                     Created
   -------------------------------------------------------------------------------
   1.1.1     APPLIED     research_papers Add citation_count field        2024-11-15
   1.2.0     APPLIED     research_papers Add full-text search indices    2024-11-20
   1.2.1     PENDING     research_papers Fix metadata indexing           2024-11-25
   1.3.0     PENDING     -               Add priority field              2024-12-01

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
   lvdb mcp serve --databases-map '{"papers": "/data/research", "docs": "http://remote-server:5000/docs"}'

   # Start with debug logging
   lvdb mcp serve --log-level DEBUG

**Options**:

- ``--mode``: Server mode (read-only, read-write)
- ``--config``: Path to MCP configuration file (TOML format)
- ``--databases-root``: Root directory for local databases
- ``--databases-map``: JSON mapping of database names to paths/URLs
- ``--log-level``: Logging level (DEBUG, INFO, WARNING, ERROR)

**MCP Configuration File Example**:

.. code-block:: toml

   # mcp-config.toml
   mode = "read-write"
   log_level = "INFO"
   
   [databases]
   # Local databases
   research = "/data/research_papers"
   documentation = "/data/docs"
   
   # Remote databases
   shared_knowledge = "http://knowledge-server:5000/shared"
   
   [security]
   read_only_databases = ["shared_knowledge"]
   require_auth = true

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

When running as an MCP server, LocalVectorDB provides these tools to AI assistants:

- ``search_database``: Search documents in a database
- ``get_document``: Retrieve specific document by ID
- ``list_documents``: List document IDs in a database  
- ``add_document``: Add new document (read-write mode only)
- ``update_document``: Update existing document (read-write mode only)
- ``delete_document``: Remove document (read-write mode only)
- ``get_database_info``: Get database statistics and configuration
- ``list_databases``: List available databases
