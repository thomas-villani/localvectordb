=======================================
Your First 5 Minutes with LocalVectorDB
=======================================

This tutorial gets you from zero to searching documents in under 5 minutes. No complex setup, no configuration files
- just the essentials to see LocalVectorDB in action.

What You'll Build
=================

In the next 5 minutes, you'll:

1. Install LocalVectorDB
2. Create your first database
3. Add some sample documents
4. Search and see results
5. Understand what just happened

Let's Go!
=========

Step 1: Install (30 seconds)
-----------------------------

.. code-block:: bash

   pip install localvectordb

That installs the library itself. This tutorial's default embedding provider is Ollama,
which runs locally and is installed separately (see `No Ollama? No Problem!`_ to use a hosted
provider instead, or the Ollama setup steps near the end to install it).

.. tip::

    It is best practice to isolate your environments using a virtual environment.
    Before running ``pip install ...``, try using:

    .. code-block:: console

       $ python -m venv venv
       $ source venv/bin/activate


Step 2: Create and Test (4.5 minutes)
--------------------------------------

Create a new Python file called ``quick_start.py`` and copy this code:

.. code-block:: python

   from localvectordb import VectorDB

   # Create a database (this creates files in ./my_first_db/)
   print("Creating database...")
   db = VectorDB(
       name="my_first_db",
       base_path="./my_first_db",
       embedding_provider="ollama",  # Uses local Ollama
       embedding_model="nomic-embed-text"
   )

   # Add some sample documents
   print("Adding documents...")
   documents = [
       "Python is a versatile programming language used for web development, data science, and automation.",
       "JavaScript enables interactive web pages and runs both in browsers and on servers with Node.js.",
       "Machine learning algorithms can automatically improve through experience and learn from data.",
       "LocalVectorDB is a simple vector database that makes semantic search easy to implement.",
       "Cooking pasta requires boiling water, adding salt, and cooking until al dente."
   ]

   # Insert all documents at once
   doc_ids = db.upsert(documents)
   print(f"Added {len(doc_ids)} documents with IDs: {doc_ids}")

   # Search for something
   print("\nSearching for 'programming languages'...")
   results = db.query("programming languages", k=3)

   print(f"Found {len(results)} results:")
   for i, result in enumerate(results, 1):
       print(f"\n{i}. Score: {result.score:.3f}")
       print(f"   Content: {result.content}")

   # Try another search
   print("\nSearching for 'cooking food'...")
   results = db.query("cooking food", k=2)

   print(f"Found {len(results)} results:")
   for i, result in enumerate(results, 1):
       print(f"\n{i}. Score: {result.score:.3f}")
       print(f"   Content: {result.content}")

   # Check database stats
   import json
   print(f"\nDatabase stats:\n{json.dumps(db.get_stats(), indent=2)}")

   # Clean up
   db.close()
   print("\nDone! You just created your first vector database!")

Run Your First Database
-----------------------

.. code-block:: bash

   python quick_start.py

You should see output like this:

.. code-block:: text

   Creating database...
   Adding documents...
   Added 5 documents with IDs: ['doc_1', 'doc_2', 'doc_3', 'doc_4', 'doc_5']

   Searching for 'programming languages'...
   Found 3 results:

   1. Score: 0.789
      Content: Python is a versatile programming language used for web development, data science, and automation.

   2. Score: 0.712
      Content: JavaScript enables interactive web pages and runs both in browsers and servers with Node.js.

   3. Score: 0.234
      Content: Machine learning algorithms can automatically improve through experience and learn from data.

   Searching for 'cooking food'...
   Found 2 results:

   1. Score: 0.856
      Content: Cooking pasta requires boiling water, adding salt, and cooking until al dente.

   2. Score: 0.123
      Content: LocalVectorDB is a simple vector database that makes semantic search easy to implement.

   Database stats: {'documents': 5, 'chunks': 5, 'index_vectors': 5, 'embedding_dimension': 768}

   Done! You just created your first vector database!

What Just Happened?
===================

**You created an AI-powered search engine!** Here's what LocalVectorDB did behind the scenes:

1. **Generated Embeddings**: Converted your text into numerical vectors that capture semantic meaning
2. **Built an Index**: Created a searchable index of these vectors using FAISS
3. **Performed Semantic Search**: Found documents similar in meaning, not just exact word matches

Notice how:

- "programming languages" found both Python and JavaScript docs
- "cooking food" found the pasta document, even though the query didn't contain those exact words
- The scores show relevance (higher = more similar)

Try It Yourself
===============

Modify the script to:

**Add Your Own Documents**

.. code-block:: python

   my_documents = [
       "Your first document here",
       "Add as many as you want",
       "Each will be searchable"
   ]

   doc_ids = db.upsert(my_documents)

**Try Different Searches**

.. code-block:: python

   # These will work even if the exact words aren't in your documents
   results = db.query("artificial intelligence")
   results = db.query("web development")
   results = db.query("data analysis")

**See Different Search Types**

.. code-block:: python

   # Vector search (semantic similarity)
   vector_results = db.query("machine learning", search_type="vector")

   # Keyword search (exact word matching)
   keyword_results = db.query("machine learning", search_type="keyword")

   # Hybrid search (combines both)
   hybrid_results = db.query("machine learning", search_type="hybrid")

No Ollama? No Problem!
======================

If you don't have Ollama installed, use this version instead:

.. code-block:: python

   from localvectordb import LocalVectorDB

   # Use OpenAI embeddings (requires API key)
   db = LocalVectorDB(
       name="my_first_db",
       embedding_provider="openai",
       embedding_model="text-embedding-3-small",
       embedding_config={"api_key": "your-openai-api-key"}
   )

   # Rest of the code is the same...

Or install Ollama for free local embeddings:

.. code-block:: bash

   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull the embedding model
   ollama pull nomic-embed-text

What's Next?
============

You just built a working vector database! This same pattern scales to:

- **Thousands of documents**: PDFs, articles, notes, code files
- **Advanced search**: Filter by metadata, combine search types
- **Real applications**: Chatbots, knowledge bases, recommendation systems

**Ready to go deeper?** Check out these tutorials:

- **FAQ Bot in 20 Lines**: Turn this into a question-answering system
- **Index Your Downloads Folder**: Search through your actual files
- **Building a RAG Chat Application**: Create a full chatbot with memory

**Files Created**

Your first database created these files:
- ``./my_first_db/my_first_db.sqlite`` - Document storage and metadata
- ``./my_first_db/my_first_db.faiss`` - Vector index for fast search

You can delete the ``my_first_db`` folder when you're done experimenting.

Congratulations!
================

In just 5 minutes, you've:

- Created an AI-powered database
- Added documents and performed semantic search
- Seen how LocalVectorDB works under the hood
- Built something you can immediately use and extend

Welcome to the world of vector databases!