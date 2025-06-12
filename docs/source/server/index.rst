Server
======

LocalVectorDB Server provides a production-ready HTTP API for managing multiple vector databases. It features
comprehensive authentication, configuration management, and a powerful CLI for administration.

The LocalVectorDB Server is built on Flask and provides:

- *RESTful API*: Complete HTTP API for all database operations
- *Multi-Database Management*: Handle multiple databases simultaneously
- *Authentication & Security*: API key authentication with CORS support
- *Configuration Management*: Flexible TOML/JSON configuration
- *CLI Administration*: :doc:`Comprehensive command-line tools <../cli>`
- *Production Features*: Logging, metrics, rate limiting


.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   routes
   config
   config.params
   advanced
