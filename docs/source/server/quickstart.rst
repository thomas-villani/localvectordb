Server Quick Start
------------------

Starting the Server
^^^^^^^^^^^^^^^^^^^
The easiest way to get started with the server is through the CLI.

.. code-block:: bash

   # Start with default settings
   lvdb serve

   # Start with custom configuration
   # (--config is a global option, so it comes BEFORE the `serve` subcommand)
   lvdb --config production.toml serve --host 0.0.0.0 --port 8080

   # Start with specific database folder (--db-folder is also a global option)
   lvdb --db-folder /path/to/databases serve --log-level DEBUG

Basic API Usage
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Check server health
   curl http://localhost:5000/api/v1/health

   # Create a database
   curl -X POST http://localhost:5000/api/v1/databases \
     -H "Content-Type: application/json" \
     -d '{
       "name": "my_database",
       "embedding_model": "nomic-embed-text",
       "chunk_size": 500
     }'

   # Add documents
   curl -X POST http://localhost:5000/api/v1/my_database/documents \
     -H "Content-Type: application/json" \
     -d '{
       "documents": ["This is my first document", "This is my second document"],
       "metadata": [{"category": "test"}, {"category": "example"}]
     }'

   # Search documents
   curl -X POST http://localhost:5000/api/v1/my_database/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "search text",
       "search_type": "vector",
       "k": 5
     }'

Production Setup with Authentication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For production use, enable API key authentication:

.. code-block:: bash

   # Enable authentication in server config
   lvdb config set server.security.require_api_key true

   # Create API keys with appropriate permissions
   # Read-write key for admin operations
   lvdb auth create-key --description "Admin API" --permission-level read_write --expires-days 30

   # Read-only key for public search
   lvdb auth create-key --description "Public Search" --permission-level read_only --expires-days 365

   # Use API keys in requests
   curl -H "Authorization: Bearer your_api_key_here" \
        http://localhost:5000/api/v1/databases

Learn More
^^^^^^^^^^

- :doc:`Command Line Interface (CLI) Documentation <../cli>`
- :doc:`routes`
- :doc:`config`
- :doc:`config.params`
- :doc:`advanced`