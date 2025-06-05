Server
======

LocalVectorDB Server provides a production-ready HTTP API for managing multiple vector databases. It features comprehensive authentication, configuration management, and a powerful CLI for administration.

Overview
--------

The LocalVectorDB Server is built on Flask and provides:

- *RESTful API*: Complete HTTP API for all database operations
- *Multi-Database Management*: Handle multiple databases simultaneously
- *Authentication & Security*: API key authentication with CORS support
- *Configuration Management*: Flexible TOML/YAML/JSON configuration
- *CLI Administration*: Comprehensive command-line tools
- *Production Features*: Logging, metrics, rate limiting

Quick Start
-----------

Starting the Server
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Start with default settings
   lvdb serve

   # Start with custom configuration
   lvdb serve --config production.toml --host 0.0.0.0 --port 8080

   # Start with specific database folder
   lvdb serve --db-folder /path/to/databases --log-level DEBUG

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


Learn More
----------
- :doc:`Server Routes API <routes>`
- :doc:`Server Configuration <config>`
- :doc:`Command Line Interface (CLI) Overview <cli>`
- :doc:`Advanced Server Usage <server.advanced>`

