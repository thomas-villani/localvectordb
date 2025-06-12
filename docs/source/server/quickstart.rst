Server Quick Start
------------------

Starting the Server
^^^^^^^^^^^^^^^^^^^
The easiest way to get started with the server is through the CLI.

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
^^^^^^^^^^

- :doc:`Command Line Interface (CLI) Documentation <../cli>`
- :doc:`routes`
- :doc:`config`
- :doc:`config.params`
- :doc:`advanced`