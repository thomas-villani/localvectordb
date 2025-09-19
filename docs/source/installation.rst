Installation
============

Requirements
------------
- **Python**: 3.12 or higher
- **Operating System**: Linux, macOS, Windows
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: SSD recommended for optimal performance

Installation Options
--------------------
Basic Installation
^^^^^^^^^^^^^^^^^^
For local vector database functionality:

.. code-block:: bash

   pip install localvectordb

This includes:

- Core LocalVectorDB library
- SQLite and FAISS dependencies
- Basic embedding providers
- Chunking and search functionality

Server Installation
^^^^^^^^^^^^^^^^^^^
For running the LocalVectorDB HTTP server:

.. code-block:: bash

   pip install localvectordb[server]

Additional dependencies:

- Flask web framework
- HTTP client libraries
- Configuration management
- CLI tools

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^
For contributing or advanced usage:

.. code-block:: bash

   pip install localvectordb[dev]

Includes:

- Testing frameworks
- Documentation tools
- Code quality tools
- All optional dependencies

.. todo: update github link

From Source
^^^^^^^^^^^
.. code-block:: bash

   git clone https://github.com/your-org/localvectordb.git
   cd localvectordb
   pip install -e .[dev]

System Dependencies
-------------------
Ollama (Recommended)
^^^^^^^^^^^^^^^^^^^^
For local embeddings without API keys:

.. code-block:: bash

   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh

   # Windows
   # Download from https://ollama.ai/download

   # Pull embedding model
   ollama pull nomic-embed-text

FAISS Installation
^^^^^^^^^^^^^^^^^^
FAISS is automatically installed, but for GPU support:

.. code-block:: bash

   # CPU version (default)
   pip install faiss-cpu

   # GPU version (if you have CUDA)
   pip install faiss-gpu

SQLite FTS5
^^^^^^^^^^^
Most Python installations include FTS5 support. To verify:

.. code-block:: python

   import sqlite3
   conn = sqlite3.connect(':memory:')
   cursor = conn.execute("PRAGMA compile_options")
   options = [row[0] for row in cursor.fetchall()]
   has_fts5 = 'ENABLE_FTS5' in options
   print(f"FTS5 available: {has_fts5}")

Configuration
-------------
Environment Variables
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Ollama configuration
   export OLLAMA_HOST=http://localhost:11434

   # OpenAI configuration
   export OPENAI_API_KEY=your_api_key_here

   # LocalVectorDB server
   export LVDB_SERVER_CONFIG=/path/to/config.toml
   export LVDB_DATABASE_ROOT_DIR=/path/to/databases

First-Time Setup
^^^^^^^^^^^^^^^^
#. **Verify Installation**:

   .. code-block:: python

      import localvectordb
      print(localvectordb.__version__)

      # Test basic functionality
      from localvectordb import VectorDB
      db = VectorDB("test", ":memory:")
      print("LocalVectorDB installed successfully!")

#. **Test Embedding Provider**:

   .. code-block:: python

      from localvectordb.embeddings import EmbeddingRegistry

      # List available providers
      providers = EmbeddingRegistry.list()
      print(f"Available providers: {providers}")

      # Test Ollama
      try:
          provider = EmbeddingRegistry.create_provider("ollama", "nomic-embed-text")
          if provider.validate_model():
              print("Ollama setup successful!")
      except Exception as e:
          print(f"Ollama setup failed: {e}")

#. **Initialize Configuration** (for server):

   .. code-block:: bash

      lvdb config init --format toml --schema documents

Troubleshooting
---------------
Common Issues
^^^^^^^^^^^^^
**ImportError: No module named ’faiss’**

.. code-block:: bash

   pip install faiss-cpu

**Ollama connection errors**

.. code-block:: bash

   # Check if Ollama is running
   ollama list

   # Start Ollama service
   ollama serve

**SQLite FTS5 not available**

- Upgrade Python to a newer version
- Or compile SQLite with FTS5 support

**Permission errors on database files**

.. code-block:: bash

   # Ensure proper permissions
   chmod 755 /path/to/database/directory


.. todo: Update this

Getting Help
^^^^^^^^^^^^
- **Documentation**: https://localvectordb.readthedocs.io
- **GitHub Issues**: https://github.com/your-org/localvectordb/issues