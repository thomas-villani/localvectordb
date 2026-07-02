Configuration Parameters
========================

.. contents::
   :local:
   :depth: 2

Full Configuration File
-----------------------

.. code-block:: toml

   # LocalVectorDB Server Configuration v1.0
   [database]
   root_dir = "/var/www/localvectordb_server/.lvdb"  # Recommend absolute paths in production environment
   timeout = 300
   connection_pool_size = 10
   enable_gpu = false
   enable_fts = true
   chunk_size = 500
   chunk_overlap = 1
   chunking_method = "lines"

   # Faiss index settings
   faiss_index_type = "IndexFlatL2"       # Can be one of: IndexFlatL2, IndexFlatIP, IndexHNSWFlat, IndexLSH
   # faiss_index_hnsw_flat_neighbors = 0  # Only used for IndexHNSWFlat, set the number of neighbors for the graph
   # faiss_index_lsh_bits = 1536          # Only used for IndexLSH Typically set to twice the embedding dimension

   [database.default_metadata_schema]
   file_path = { type = "text", indexed = true, required = false }
   created_at = { type = "date", indexed = true, required = false }
   last_modified = { type = "date", indexed = true, required = false }
   mimetype = { type = "text", indexed = true, required = false }
   tags = { type = "json", indexed = false, required = false }
   # You can define the type directly if you don't need indexing and the field is not required
   file_size_bytes = "integer"


   [embedding]
   provider = "ollama"
   model = "nomic-embed-text"
   batch_size = 64
   timeout = 30
   max_retries = 3

   [server]
   debug = false
   environment = "development"
   host = "127.0.0.1"
   port = 5000
   log_level = "INFO"
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   file_upload_enabled = true   # enables the /api/v1/<db-name>/upload route
   max_request_size = 104857600
   enable_rate_limiting = false
   rate_limit = "100 per minute"
   rate_limit_storage_uri = "memory://"

   proxy_enabled = true
   # REQUIRED when proxy_enabled = true: list of trusted proxy IPs / CIDR blocks
   # allowed to set forwarded headers. Validation fails if this is empty.
   trusted_proxies = ["10.0.0.0/8", "127.0.0.1"]

   cache_enabled = true
   cache_ignore_errors = true
   cache_timeout = 300
   cache_type = "RedisCache"

   [server.cache_settings]  # These parameters are passed to the cache constructor. See https://cachelib.readthedocs.io/en/stable/
   host = "localhost"
   port = 6379
   db = 0
   password = "$REDIS_PASSWORD"   # '$' prefix indicates to load the environment variable
   key_prefix = "my-db-prefix-"   # Allows you to run multiple servers using single redis db.

   # IMPORTANT: if ``proxy_enabled = true``, you must set the proxy settings AND trusted_proxies.
   # The specific values depend on your server set-up. Set the number of proxies forwarding each header.
   # When behind a reverse proxy (nginx, etc.), configure trusted_hosts, trusted_proxies, and proxy settings.
   [server.proxy_settings]
   x_for = 1
   x_proto = 1
   x_host = 1
   x_prefix = 1

   [server.security]
   # API settings
   require_api_key = true
   key_database_path = "/path/to/api_keys.db"    # If not provided, located in ``{database.root_dir}/api_keys.db``
   # Recommended to set keys to expire
   default_key_expiry_days = 90
   api_key_header = "Authorization"
   auto_prune_expired_keys = false
   key_audit_logging = true
   auth_log_level = "INFO"
   warn_expiring_days = 7

   # Filter connections by hostname for best practices.
   trusted_hosts = ["your-website.com", "*.subdomain.example.com", "localhost"]

   # CORS settings
   cors_enabled = true
   # IMPORTANT: if CORS is enabled, you need to set the allowed origins!
   cors_allowed_origins = ["http://localhost:5000", "https://your-website.com"]
   cors_allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
   # If you changed the default Authorization header, make sure to put it here too
   cors_allowed_headers = ["Content-Type", "Authorization"]
   cors_max_age = 86400

   [extraction]
   # Hardened defaults for untrusted uploads; relax only for trusted content.
   allow_remote_fetch = false           # Fetch remote assets referenced by a document (SSRF risk)
   allowed_hosts = []                   # Host allowlist applied when allow_remote_fetch = true
   strip_dangerous_elements = true      # HTML: strip scripts and event handlers
   attachment_mode = "skip"             # How embedded attachments/assets are handled


.. note::

   The configuration file shown above is a representative example of a configuration that might be used in a production
   environment. You will need to modify this for your server! **Do not** copy and paste and use without modifications,
   as it almost certainly **won't work**.


Database Settings
-----------------

.. list-table::
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Default**
     - **Description**
   * - ``root_dir``
     - str
     - ``"./.lvdb"``
     - Root directory for storing database files and data.
   * - ``timeout``
     - int
     - ``300``
     - Timeout for database operations, in seconds.
   * - ``connection_pool_size``
     - int
     - ``10``
     - Number of simultaneous database connections allowed (connection pool size).
   * - ``enable_gpu``
     - bool
     - ``false``
     - Whether to enable GPU acceleration for database operations (if supported).
   * - ``enable_fts``
     - bool
     - ``true``
     - Whether to enable Full-Text Search (FTS) capabilities.
   * - ``faiss_index_type``
     - str
     - ``IndexFlatL2``
     - The type of faiss index to use: ``IndexFlatL2`` (exact L2 distance), ``IndexFlatIP`` (exact inner product), ``IndexHNSWFlat`` (graph-based approximate-nearest neighbors, fast), ``IndexLSH`` (binary hashing)
   * - faiss_index_hnsw_flat_neighbors
     - int
     - ``null``
     - The number of neighbors for the ``IndexHNSWFlat`` index. Ignored if using a different index.
   * - faiss_index_lsh_bits
     - int
     - ``null``
     - The number of bits to use for the binary hashing, defaults to 2x the number of embedding dimensions if not provided and using ``IndexLSH``, ignored otherwise.
   * - ``chunk_size``
     - int
     - ``500``
     - Default size (in tokens or lines, depending on ``chunking_method``) for document chunking.
   * - ``chunk_overlap``
     - int
     - ``1``
     - Number of tokens/lines to overlap between chunks.
   * - ``chunking_method``
     - str
     - ``"lines"``
     - Method for splitting documents into chunks (e.g., ``"lines"``).
   * - ``default_metadata_schema``
     - Dictionary (table) or null
     - ``null``
     - Default metadata schema as a table of field names to ``MetadataField`` definitions (type, indexed, required) for new databases.

Embedding Settings
------------------

.. list-table::
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Default**
     - **Description**
   * - ``provider``
     - str
     - ``"ollama"``
     - Embedding provider to use (e.g., ``"ollama"``, ``"openai"``).
   * - ``model``
     - str
     - ``"nomic-embed-text"``
     - Embedding model name to use with the provider.
   * - ``base_url``
     - str or null
     - ``null``
     - Base URL for the embedding provider's API (if required).
   * - ``api_key``
     - str or null
     - ``null``
     - API key for the embedding provider (required for some providers, e.g., OpenAI).
   * - ``batch_size``
     - int
     - ``64``
     - Number of items to embed per API request (batch size).
   * - ``timeout``
     - int
     - ``30``
     - Timeout for embedding API requests, in seconds.
   * - ``max_retries``
     - int
     - ``3``
     - Maximum number of retries for embedding API requests.
   * - ``config``
     - Dictionary (table) or null
     - ``null``
     - Additional provider-specific configuration options.

Server Settings
---------------

.. note::
   Parameters from ``require_api_key`` onward in the table below (authentication, key
   management, ``trusted_hosts``, and CORS settings) live under the ``[server.security]``
   table, **not** ``[server]``. The remaining parameters (``debug`` … ``trusted_proxies``)
   live under ``[server]``. Use the dot-notation paths accordingly, e.g.
   ``server.security.require_api_key`` and ``server.host``.

.. list-table::
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Default**
     - **Description**
   * - ``debug``
     - bool
     - ``false``
     - Whether to run the server in debug mode (enables verbose error output).
   * - ``environment``
     - str
     - ``"development"``
     - Server environment (e.g., ``"development"``, ``"production"``).
   * - ``host``
     - str
     - ``"127.0.0.1"``
     - Host address to bind the server to.
   * - ``port``
     - int
     - ``5000``
     - Port number to bind the server to.
   * - ``log_level``
     - str
     - ``"INFO"``
     - Logging level (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``).
   * - ``log_format``
     - str
     - ``'%(asctime)s - %(name)s - %(levelname)s - %(message)s'``
     - Format string for log messages.
   * - ``file_upload_enabled``
     - bool
     - ``false``
     - Whether to allow file uploads via the ``/api/v1/<db-name>/upload`` route.
   * - ``max_request_size``
     - int
     - ``104857600`` (100 MB)
     - Maximum allowed size for incoming requests, in bytes.
   * - ``enable_rate_limiting``
     - bool
     - ``false``
     - Whether to enable API rate limiting.
   * - ``rate_limit``
     - str
     - ``"100 per minute"``
     - Rate limit rule (e.g., requests per time period).
   * - ``rate_limit_storage_uri``
     - str
     - ``"memory://"``
     - Storage backend URI for rate limiting (e.g., ``"memory://"``, ``"redis://..."``).
   * - ``cache_enabled``
     - bool
     - ``false``
     - Whether to enable server-side caching.
   * - ``cache_ignore_errors``
     - bool
     - ``true``
     - If ``true``, cache errors are ignored rather than causing failures.
   * - ``cache_timeout``
     - int
     - ``300``
     - Default cache timeout, in seconds.
   * - ``cache_key_prefix``
     - str
     - ``"lvdb_cache_"``
     - Prefix for all cache keys.
   * - ``cache_type``
     - string
     - ``"SimpleCache"``
     - Type of cache backend to use, one of: ``"SimpleCache"``, ``"RedisCache"``, ``"FileSystemCache"``, ``"MemcachedCache"``, ``"UWSGICache"``, ``"DynamoDbCache"``, ``"MongoDbCache"``
   * - ``cache_settings``
     - dict or null
     - ``null``
     - Additional configuration for the cache backend (e.g. Redis cache config settings). Set as a TOML table under ``[server.cache_settings]``. See https://cachelib.readthedocs.io/en/stable/ **Deprecated** in favour of the typed backend keys below.
   * - ``redis_host`` / ``redis_port`` / ``redis_db``
     - str / int / int
     - ``"localhost"`` / ``6379`` / ``0``
     - Redis connection settings (used when ``cache_type = "RedisCache"``).
   * - ``redis_password``
     - str or null
     - ``null``
     - Redis password, if required.
   * - ``redis_url``
     - str or null
     - ``null``
     - Full Redis URL; when set, overrides the individual ``redis_*`` settings.
   * - ``redis_socket_timeout`` / ``redis_socket_connect_timeout``
     - float or null
     - ``null``
     - Redis socket read / connect timeouts in seconds.
   * - ``filesystem_cache_dir``
     - str or null
     - ``null``
     - Directory for ``FileSystemCache`` entries.
   * - ``filesystem_cache_threshold``
     - int
     - ``500``
     - Maximum number of items ``FileSystemCache`` keeps before pruning.
   * - ``memcached_servers``
     - list of str
     - ``["127.0.0.1:11211"]``
     - Memcached server addresses (used when ``cache_type = "MemcachedCache"``).
   * - ``memcached_username`` / ``memcached_password``
     - str or null
     - ``null``
     - Memcached SASL credentials, if required.
   * - ``use_single_cache``
     - bool
     - ``false``
     - Reuse the general cache for the multi-worker database registry instead of
       configuring a separate ``db_registry`` backend.
   * - ``proxy_enabled``
     - bool
     - ``false``
     - Whether to enable proxy support (for use behind reverse proxies).
   * - ``proxy_settings``
     - dict or null
     - ``null``
     - Proxy configuration settings. Required if ``proxy_enabled`` is true. Must be a dict with: x_for, x_proto, x_host, x_prefix, set to the number of proxies forwarding the ``X-Forwarded-`` headers. Set as a TOML table under ``[server.proxy_settings]``.
   * - ``trusted_proxies``
     - list of str
     - ``[]``
     - List of trusted proxy IP addresses or CIDR blocks allowed to set forwarded headers. **Required (non-empty) when** ``proxy_enabled`` **is true** — configuration validation fails otherwise. Set under ``[server]``.
   * - ``require_api_key``
     - bool
     - ``false``
     - Whether API key authentication is required for access.
   * - ``api_key_header``
     - str
     - ``"Authorization"``
     - HTTP header used to transmit the API key.
   * - ``trusted_hosts``
     - list of str or null
     - ``null``
     - List of trusted hostnames or IP addresses for incoming requests.
   * - ``key_database_path``
     - str or null
     - ``null``
     - Path to the API key database file (``null`` = in the ``database.root_dir`` folder as ``api_keys.db``).
   * - ``default_key_expiry_days``
     - int or null
     - ``null``
     - Default number of days before API keys expire (``null`` = no expiration).
   * - ``auto_prune_expired_keys``
     - bool
     - ``false``
     - Whether to automatically remove expired API keys.
   * - ``key_audit_logging``
     - bool
     - ``true``
     - Whether to log API key usage for audit purposes.
   * - ``auth_log_level``
     - str
     - ``"INFO"``
     - Logging level for authentication events.
   * - ``warn_expiring_days``
     - int
     - ``7``
     - Number of days before key expiry to issue warnings.
   * - ``cors_enabled``
     - bool
     - ``true``
     - Whether to enable Cross-Origin Resource Sharing (CORS).
   * - ``cors_allowed_origins``
     - str or list of str
     - ``"*"``
     - Allowed origins for CORS (``"*"``, or a list of origins).
   * - ``cors_allowed_methods``
     - list of str
     - ``["GET", "POST", "PUT", "DELETE", "OPTIONS"]``
     - Allowed HTTP methods for CORS.
   * - ``cors_allowed_headers``
     - list of str
     - ``["Content-Type", "Authorization"]``
     - Allowed HTTP headers for CORS.
   * - ``cors_max_age``
     - int
     - ``86400``
     - Maximum time (in seconds) for browsers to cache CORS preflight responses.
   * - ``security_headers_enabled``
     - bool
     - ``true``
     - Master switch for the security-header / CSP middleware. When ``false``, none
       of the headers below are applied.
   * - ``force_https``
     - bool
     - ``false``
     - Redirect HTTP requests to HTTPS and mark cookies secure.
   * - ``strict_transport_security``
     - bool
     - ``true``
     - Send the ``Strict-Transport-Security`` (HSTS) header.
   * - ``strict_transport_security_max_age``
     - int
     - ``31536000``
     - ``max-age`` (seconds) for the HSTS header.
   * - ``content_security_policy``
     - dict
     - see below
     - Content-Security-Policy directives as a mapping of directive to source
       expression (default allows ``'self'`` plus inline scripts/styles). Set as a
       TOML table under ``[server.security.content_security_policy]``.
   * - ``content_type_nosniff``
     - bool
     - ``true``
     - Send ``X-Content-Type-Options: nosniff``.
   * - ``x_frame_options``
     - str
     - ``"DENY"``
     - Value for the ``X-Frame-Options`` header.
   * - ``x_xss_protection``
     - bool
     - ``true``
     - Send the legacy ``X-XSS-Protection`` header.
   * - ``referrer_policy``
     - str
     - ``"strict-origin-when-cross-origin"``
     - Value for the ``Referrer-Policy`` header.

Extraction Settings
-------------------

These live under the ``[extraction]`` table and control how uploaded files are
converted to Markdown (via all2md). Defaults are hardened for untrusted uploads.
See :doc:`/file-extraction` for the full security model.

.. list-table::
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Default**
     - **Description**
   * - ``allow_remote_fetch``
     - bool
     - ``false``
     - Allow fetching remote assets referenced by a document. Leaving this off avoids an SSRF surface.
   * - ``allowed_hosts``
     - list of str or null
     - ``null``
     - Host allowlist applied when ``allow_remote_fetch`` is enabled (``null`` = all hosts).
   * - ``strip_dangerous_elements``
     - bool
     - ``true``
     - HTML only: strip ``<script>`` tags and inline event handlers.
   * - ``attachment_mode``
     - str
     - ``"skip"``
     - How embedded attachments/assets are handled during extraction.

.. note::
   All configuration parameters can also be set via environment variables with the ``LVDB_`` prefix,
   using uppercase and underscores, e.g., ``LVDB_SERVER_DEBUG``, ``LVDB_DATABASE_ROOT_DIR``,
   ``LVDB_EXTRACTION_ALLOW_REMOTE_FETCH``, etc.
