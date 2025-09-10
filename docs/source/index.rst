.. localvectordb documentation master file, created by
   sphinx-quickstart on Tue May 27 14:23:00 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

localvectordb documentation
===========================

LocalVectorDB is a document-first vector database that combines the simplicity of SQLite with the power of FAISS for
semantic search. Unlike traditional vector databases that require you to manage chunks manually, LocalVectorDB lets
you work with complete documents while automatically handling the chunking, embedding, and indexing behind the scenes.
Whether you're building a RAG application, document search system, or knowledge base, LocalVectorDB provides a unified
interface for vector similarity search, keyword search, and hybrid search with normalized scoring across all methods.

Built for both development and production use, LocalVectorDB features a plugin-based embedding system supporting local
providers like Ollama (free, no API keys) and cloud providers like OpenAI, structured metadata with indexed SQLite
columns for fast filtering, and position-tracking chunking that enables perfect document reconstruction and precise
highlighting. The included HTTP server and comprehensive CLI tools make it easy to deploy in any environment—from
local development with an in-memory database to production clusters with multi-database management, authentication,
and monitoring. With its identical API for local and remote databases, you can start developing locally and seamlessly
scale to distributed deployments without changing your code.



.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   installation
   overview
   query
   embeddings
   chunking
   metadata.filtering
   querybuilder
   document-scoring
   server/index
   cli
   mcp
   backup
   migrations
   recipes
   tutorials/index
   modules/index
