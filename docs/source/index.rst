=============================
localvectordb documentation
=============================

*A document-first vector database. SQLite + FAISS, zero infrastructure.*

You bring documents; LocalVectorDB handles chunking, embedding, and indexing.
Search them by meaning, by keyword, or both at once — through one API that works
identically against a local file and a remote server.

.. code-block:: python

   from localvectordb import VectorDB

   db = VectorDB("my_docs", "./data")

   db.upsert([
       "Python is a programming language",
       "Machine learning with neural networks",
   ])

   for result in db.query("programming", k=5):
       print(f"{result.id}: {result.score:.3f}  {result.content}")

That is a complete, running program. No server to stand up, no index to
configure, no chunking to hand-roll.

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 4 0 0 0

   .. grid-item-card:: New here?
      :link: quickstart
      :link-type: doc

      Install it, index your first documents, and run a hybrid search in about
      five minutes.

   .. grid-item-card:: Learn by building
      :link: tutorials/index
      :link-type: doc

      End-to-end tutorials: a RAG pipeline, an FAQ bot, a downloads indexer, and
      a custom embedding provider.

   .. grid-item-card:: Understand the model
      :link: overview
      :link-type: doc

      How documents, chunks, embeddings, and metadata fit together — and why the
      API is shaped the way it is.

   .. grid-item-card:: Look something up
      :link: modules/index
      :link-type: doc

      Full API reference for the library, the HTTP server, the CLI, and the
      TypeScript SDK.


What makes it different
=======================

Plenty of libraries will embed a list of strings for you. These are the things
LocalVectorDB does that most vector stores do not.

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: Hierarchical retrieval
      :link: hierarchical
      :link-type: doc

      Search a *document → section → chunk* hierarchy, so a query can match a
      whole section instead of one stray sentence adrift in a long report.
      Section and document vectors are centroids of chunk embeddings, so this
      costs no extra embedding calls.

   .. grid-item-card:: Reverse-RAG fact-checking
      :link: factcheck
      :link-type: doc

      Point it at LLM-generated text and your corpus, and it scores each claim
      for grounding — citing a supporting excerpt, or flagging the claim as
      unsupported or contradicted.

   .. grid-item-card:: Comparison & visualization
      :link: comparison
      :link-type: doc

      Nearest neighbours and similarity matrices at whole-document level, t-SNE
      and PCA embedding maps, and synteny ribbons showing how two documents
      align chunk by chunk.


Choose your path
================

**Working in Python.** Start with :doc:`quickstart`, then :doc:`query` for the
search API and :doc:`metadata.filtering` for narrowing results by structured
fields.

**Standing up a service.** :doc:`server/index` covers the FastAPI server,
authentication, and rate limiting. :doc:`sdk/index` documents the TypeScript
client, and :doc:`cli` the ``lvdb`` command.

**Wiring it into an AI agent.** :doc:`mcp` sets up the built-in Model Context
Protocol server, which lets Claude Code, Claude Desktop, and other MCP clients
search your databases directly — read-only by default.

**Tuning for scale.** :doc:`chunking` and :doc:`embeddings` control what gets
indexed and how. :doc:`document-scoring` covers the chunk-to-document
aggregation strategies. :doc:`performance` has the benchmark numbers.


.. toctree::
   :caption: Get started
   :maxdepth: 2
   :hidden:

   quickstart
   installation
   tutorials/index

.. toctree::
   :caption: Core concepts
   :maxdepth: 2
   :hidden:

   overview
   client
   query
   embeddings
   chunking
   metadata.filtering
   querybuilder
   document-scoring
   hierarchical
   hierarchical-evaluation

.. toctree::
   :caption: Advanced features
   :maxdepth: 2
   :hidden:

   streaming
   comparison
   factcheck
   file-extraction
   backup
   migrations
   performance

.. toctree::
   :caption: Deployment
   :maxdepth: 2
   :hidden:

   server/index
   cli
   sdk/index

.. toctree::
   :caption: AI agent integration
   :maxdepth: 2
   :hidden:

   mcp
   skills

.. toctree::
   :caption: Reference
   :maxdepth: 2
   :hidden:

   recipes
   modules/index
