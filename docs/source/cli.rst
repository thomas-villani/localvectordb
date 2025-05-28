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

Initialize Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create default TOML configuration
   lvdb config init

   # Create YAML configuration
   lvdb config init --format yaml --output server.yaml

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

   # LocalVectorDB Server Configuration v2.0

   [database]
   root_dir = "./.lvdb"
   timeout = 300
   connection_pool_size = 10
   enable_gpu = false
   enable_fts = true
   chunk_size = 500
   chunk_overlap = 1
   embedding_model = "nomic-embed-text"
   provider = "ollama"
   chunking_method = "sentences"

   [embedding]
   provider = "ollama"
   model = "nomic-embed-text"
   batch_size = 64
   timeout = 30

   [server]
   host = "127.0.0.1"
   port = 5000
   log_level = "INFO"
   require_api_key = false
   cors_enabled = true

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
   lvdb list --db-folder /path/to/databases

**Example Output**:

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
   lvdb db my_database add *.txt

   # Add with metadata
   lvdb db my_database add paper.pdf --metadata '{"title": "Research Paper", "author": "Dr. Smith"}'

   # Add from stdin
   cat document.txt | lvdb db my_database add -

   # Add with auto metadata
   lvdb db my_database add *.pdf --metadata auto

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

   my_database> add *.pdf
   Found 5 files. Adding to database...
   Successfully added 5 documents
   Created IDs: doc_890, doc_891, doc_892, doc_893, doc_894

   my_database> count
   Document count: 1255, Chunk count: 8505

   my_database> exit
   Database connection closed.

Authentication Management
-------------------------

Check Authentication Status
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Show current auth configuration
   lvdb auth status

**Example Output**:

.. code-block:: console

   Configuration file: /home/user/server-cfg.toml
   API Authentication: Enabled
   API Keys configured: 3

Global Search
-------------

.. code-block:: bash

   # Global search across all databases
   lvdb search "artificial intelligence" --search-type vector --limit 5

   # Search specific databases
   lvdb search "machine learning" --databases "research_papers,tech_docs"

   # Hybrid search across databases
   lvdb search "neural networks" --search-type hybrid --vector-weight 0.7

Advanced Usage Examples
-----------------------

Bulk Operations
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Add all PDFs in a directory tree
   lvdb db papers add "documents/**/*.pdf" --metadata auto

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

   # Create database if not exists
   lvdb create research_pipeline --embedding-model nomic-embed-text

   # Process incoming documents
   find /incoming/documents -name "*.txt" -newer /tmp/last_processed | while read file; do
     echo "Processing: $file"
     lvdb db research_pipeline add "$file" --metadata auto
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
