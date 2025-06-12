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

The LocalVectorDB CLI provides comprehensive API key management for secure server access.

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

.. code-block:: bash

   # Create a basic API key
   lvdb auth create-key

   # Create with description
   lvdb auth create-key --description "Production API Access"

   # Create with expiration
   lvdb auth create-key --description "Temporary Access" --expires-days 30

   # Create with creator info
   lvdb auth create-key --description "CI/CD Pipeline" --created-by "admin@company.com"

   # JSON output for scripting
   lvdb auth create-key --description "Script Access" --output json

   # Key-only output for automation
   lvdb auth create-key --output key-only

**Options**:

- ``--description, -d``: Human-readable description of the key's purpose
- ``--expires-days``: Number of days until key expires (omit for no expiration)
- ``--created-by``: Identifier of who is creating the key
- ``--output, -o``: Output format (table, json, key-only)

**Example Output**:

.. code-block:: console

   ✓ API Key Created Successfully

   Key Details:
     Key ID: key_20241201_abc123
     Description: Production API Access
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

   ID                   Description                    Status     Created      Expires      Last Used
   ------------------------------------------------------------------------------------------------------------------------
   key_20241201_abc123  Production API Access          ACTIVE     2024-12-01   Never        2024-12-01
   key_20241125_def456  CI/CD Pipeline                 ACTIVE     2024-11-25   2024-12-25   Never
   key_20241120_ghi789  Temporary Access               EXPIRED    2024-11-20   2024-11-27   2024-11-26
   key_20241115_jkl012  Development Access             ACTIVE     2024-11-15   Never        2024-11-30

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

Interactive Shell
^^^^^^^^^^^^^^^^^

Start an interactive shell for database operations:

.. code-block:: bash

   lvdb db my_database shell

**Shell Commands**:

.. code-block:: console

   my_database> help
   Available commands:
     search "<query>" [limit] [type] - Search for documents
     get <id>                       - Get document by ID
     add <file or glob>             - Add file(s) to database
     delete <id>                    - Delete document by ID
     list [limit] [offset]          - List document IDs
     count                          - Show document count
     stats                          - Show database statistics
     info                           - Show database information
     clear                          - Clear the console
     exit/quit                      - Exit shell

   my_database> search "neural networks" 5 vector
   Vector search for `neural networks`...

   Results:
   ========

   1. doc_789 (Score: 0.9123):
      Neural networks are computational models inspired by biological neural networks...
      -----

   my_database> add *.txt
   Found 5 files. Adding to database...
   Successfully added 5 documents
   Created IDs: doc_890, doc_891, doc_892, doc_893, doc_894

   my_database> count
   Document count: 1255, Chunk count: 8505

   my_database> exit
   Database connection closed.


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
