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

LocalVectorDB is published on PyPI and installs with either `uv
<https://docs.astral.sh/uv/>`_ (recommended) or pip. Every ``pip install
localvectordb[...]`` command below has a direct equivalent:

- **uv (project):** ``uv add "localvectordb[...]"``
- **uv (pip interface):** ``uv pip install "localvectordb[...]"``
- **CLI without installing:** ``uvx --from "localvectordb[server]" lvdb serve``

The examples use ``pip`` for brevity; substitute your preferred command.

Basic Installation
^^^^^^^^^^^^^^^^^^
For local vector database functionality:

.. code-block:: bash

   # uv (recommended)
   uv add localvectordb

   # ...or pip
   pip install localvectordb

This includes:

- Core LocalVectorDB library
- SQLite and FAISS dependencies
- Basic embedding providers
- Chunking and search functionality

CLI Installation
^^^^^^^^^^^^^^^^
For the ``lvdb`` command-line tool without the HTTP server:

.. code-block:: bash

   pip install "localvectordb[cli]"

This is a light install (click, tomli-w, bcrypt) that runs every ``lvdb``
command except ``lvdb serve`` — create, inspect, search, chunk, back up,
migrate, manage auth keys, and configure databases. ``lvdb serve`` prints an
actionable hint if only this extra is installed; add the ``[server]`` extra to
run the server.

Server Installation
^^^^^^^^^^^^^^^^^^^
For running the LocalVectorDB HTTP server (includes the CLI):

.. code-block:: bash

   pip install "localvectordb[server]"

Additional dependencies:

- FastAPI web framework with Uvicorn ASGI server
- HTTP client libraries
- Configuration management
- The ``lvdb`` CLI (via the ``[cli]`` extra)

SentenceTransformers Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For local inference with SentenceTransformer models:

.. code-block:: bash

   pip install "localvectordb[sentence-transformers]"

Local Embeddings Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For local inference with HuggingFace transformers models:

.. code-block:: bash

   pip install "localvectordb[local-embeddings]"

File Extraction Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Common document formats (PDF, DOCX, PPTX, XLSX, HTML, Markdown, …) are extracted
with the base install. To add all2md's extended/niche format parsers (archives,
LaTeX, Outlook ``.msg``, FictionBook, wikitext, and more):

.. code-block:: bash

   pip install "localvectordb[file-extraction]"

For OCR of scanned PDFs and images (requires the Tesseract binary — see
*System Dependencies* below):

.. code-block:: bash

   pip install "localvectordb[file-extraction-ocr]"

Extracted content is returned as **Markdown**, preserving headings, tables, and
lists. See :doc:`/file-extraction` for the full format list and security
options.

.. note::

   The heavier ``pdf-layout`` (Polyform Noncommercial license) and EasyOCR
   (PyTorch) extras are deliberately **not** exposed, to keep LocalVectorDB MIT
   and its install footprint small.

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^
For contributing or advanced usage, install the development dependency group with
uv (add ``--extra mcp`` if you are working on the MCP server):

.. code-block:: bash

   git clone https://github.com/thomas-villani/localvectordb.git
   cd localvectordb
   uv sync --dev

Includes:

- Testing frameworks
- Documentation tools
- Code quality tools
- The server, file-extraction, and visualization extras

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

Tesseract (Optional, for OCR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Required only when using the ``file-extraction-ocr`` extra to extract text from
scanned PDFs and images:

.. code-block:: bash

   # macOS
   brew install tesseract

   # Linux (Debian/Ubuntu)
   sudo apt-get install tesseract-ocr

   # Windows
   # Install from https://github.com/UB-Mannheim/tesseract/wiki

FAISS Installation
^^^^^^^^^^^^^^^^^^
``faiss-cpu`` is a runtime dependency and is installed automatically with
LocalVectorDB, so you normally do not need to install FAISS yourself.

.. code-block:: bash

   # CPU version (this is the packaged runtime dependency)
   pip install faiss-cpu

   # GPU version (only if you have CUDA; install manually)
   pip install faiss-gpu

.. note::

   LocalVectorDB only depends on and packages ``faiss-cpu``. ``faiss-gpu`` is
   **not** a declared dependency and is not pulled in by any extra; you must
   install it yourself if you want GPU acceleration. Note that ``faiss-gpu``
   is not published on PyPI for every FAISS/Python version, so it may need to
   be installed via conda or built from source.

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

      # Test basic functionality. The "mock" provider needs no external service,
      # so this verifies the install without a running Ollama/API backend.
      from localvectordb import VectorDB
      db = VectorDB("test", ":memory:", embedding_provider="mock", embedding_model="mock")
      db.upsert(["hello world"])
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
**ImportError: No module named 'faiss'**

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


Getting Help
^^^^^^^^^^^^
- **GitHub Issues**: https://github.com/thomas-villani/localvectordb/issues