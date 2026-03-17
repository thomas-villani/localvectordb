Server
======

LocalVectorDB Server provides a production-ready HTTP API for managing multiple vector databases. It features
comprehensive authentication, configuration management, and a powerful CLI for administration.

The LocalVectorDB Server is built on FastAPI and provides:

- *RESTful API*: Complete HTTP API for all database operations
- *Multi-Database Management*: Handle multiple databases simultaneously
- *Authentication & Security*: API key authentication with CORS support
- *SSE Streaming*: Server-Sent Events for real-time query result streaming
- *Document Comparison*: Pairwise similarity, nearest neighbors, and similarity matrices
- *Fact-Checking*: LLM-based factual grounding against your databases
- *Configuration Management*: Flexible TOML/JSON configuration
- *CLI Administration*: :doc:`Comprehensive command-line tools <../cli>`
- *Production Features*: Structured logging, rate limiting, security headers


.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   routes
   config
   config.params
   advanced
