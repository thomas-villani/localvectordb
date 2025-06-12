Advanced Configuration and Deployment
=====================================

This guide covers advanced configuration options for running LocalVectorDB Server in production environments, including caching, rate limiting, authentication, logging, and multi-worker deployments.

Configuration Management
------------------------

Configuration File Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~

LocalVectorDB Server supports multiple configuration formats:

.. code-block:: bash

   # TOML (recommended)
   lvdb config init --format toml --output config.toml

   # JSON
   lvdb config init --format json --output config.json

Configuration files are loaded in the following order of precedence:

1. Explicit ``--config`` parameter
2. ``LVDB_SERVER_CONFIG`` environment variable
3. Default locations:
   - ``./.lvdb-config.toml``
   - ``./.lvdb-config.json``
   - ``./.lvdb/.lvdb-config.toml``
   - ``~/.lvdb/.lvdb-config.toml``

Configuration Sections
~~~~~~~~~~~~~~~~~~~~~~

The configuration is organized into three main sections:

**Database Settings**
   Core database and embedding configuration

**Server Settings**
   HTTP server, security, and performance settings

**Embedding Settings**
   Embedding provider configuration

Environment Variable Override
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any configuration value can be overridden using environment variables with the pattern ``LVDB_<SECTION>_<KEY>``:

.. code-block:: bash

   export LVDB_SERVER_HOST=0.0.0.0
   export LVDB_SERVER_PORT=8080
   export LVDB_DATABASE_ROOT_DIR=/data/vectordb
   export LVDB_EMBEDDING_PROVIDER=openai
   export LVDB_EMBEDDING_API_KEY=sk-...

Advanced Configuration Options
------------------------------

Caching Configuration
~~~~~~~~~~~~~~~~~~~~~

Response caching significantly improves performance for repeated queries:

.. code-block:: toml

   [server]
   cache_enabled = true
   cache_type = "RedisCache"  # or "FileSystemCache", "SimpleCache"
   cache_timeout = 300        # 5 minutes
   cache_key_prefix = "lvdb_cache_"
   cache_ignore_errors = true

   # Redis cache settings
   cache_settings = { host = "localhost", port = 6379, db = 0, password = "$REDIS_PASSWORD" }

.. note:: Indicate environment variables like ``$VARIABLE_NAME``

Cache Types
^^^^^^^^^^^

**SimpleCache** (Memory)
   - Best for: Single worker deployments
   - Pros: Fast, no external dependencies
   - Cons: Not shared between workers, memory usage

**FileSystemCache**
   - Best for: Multi-worker on single machine
   - Pros: Shared between workers, persistent
   - Cons: Disk I/O overhead

**RedisCache** (Recommended for production)
   - Best for: Distributed deployments
   - Pros: Shared, fast, scalable
   - Cons: Requires Redis server

.. code-block:: toml

   # File system cache
   [server]
   cache_type = "FileSystemCache"
   cache_settings = { cache_dir = "./.lvdb/cache" }

   # Redis cache with authentication
   [server]
   cache_type = "RedisCache"

   [server.cache_settings]
   host = "redis.example.com"
   port = 6379
   db = 0
   password = "$REDIS_PASSWORD"   # Note: load from environment variable with '$' prefix
   username = "lvdb_user"


Rate Limiting
~~~~~~~~~~~~~

Protect your server from abuse with configurable rate limiting:

.. code-block:: toml

   [server]
   enable_rate_limiting = true
   rate_limit = "100 per minute"          # or "1000 per hour", "10 per second"
   rate_limit_storage_uri = "redis://localhost:6379/1"

Rate Limit Patterns
^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   # Different rate limiting strategies
   rate_limit = "100 per minute"    # Standard API usage
   rate_limit = "1000 per hour"     # Bulk operations
   rate_limit = "10 per second"     # High-frequency applications
   rate_limit = "50 per day"        # Free tier limits

The rate limiting uses the client's IP address by default. For applications behind proxies, ensure proper proxy configuration (see Proxy Settings below).

API Key Authentication
----------------------

Database-Managed API Keys
~~~~~~~~~~~~~~~~~~~~~~~~~

LocalVectorDB Server includes a comprehensive API key management system:

.. code-block:: toml

   [server]
   require_api_key = true
   key_database_path = "./.lvdb/api_keys.db"    # Auto-determined if not set
   default_key_expiry_days = 90                 # Default expiration
   auto_prune_expired_keys = true               # Auto-cleanup
   key_audit_logging = true                     # Log key usage
   warn_expiring_days = 7                       # Warning threshold

Creating and Managing API Keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create a new API key
   lvdb auth create-key --description "Production API" --expires-days 90

   # List all keys
   lvdb auth list-keys --active-only

   # Get key information
   lvdb auth key-info key_20241201_abc123

   # Rotate a key (creates new, revokes old)
   lvdb auth rotate-key key_20241201_abc123

   # Revoke a key
   lvdb auth revoke-key key_20241201_abc123

   # Clean up expired keys
   lvdb auth prune-expired --soft-delete

Key Management Best Practices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Use descriptive names**: Include purpose and owner information
2. **Set expiration dates**: Regular rotation improves security
3. **Monitor usage**: Review audit logs for suspicious activity
4. **Automated rotation**: Script key rotation for production systems
5. **Least privilege**: Create separate keys for different applications

.. code-block:: bash

   # Example production key creation
   lvdb auth create-key \
     --description "Production Web App - user:webteam" \
     --expires-days 30 \
     --created-by "deployment-script" \
     --output key-only > /secure/api-key.txt

Security Configuration
----------------------

CORS (Cross-Origin Resource Sharing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure CORS for web applications:

.. code-block:: toml

   [server]
   cors_enabled = true
   cors_allowed_origins = ["https://myapp.com", "https://dashboard.myapp.com"]
   cors_allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
   cors_allowed_headers = ["Content-Type", "Authorization"]
   cors_max_age = 86400  # 24 hours

.. code-block:: toml

   # Development setup (**NOT** secure)
   cors_allowed_origins = "*"  # Allow all origins

   # Production setup (recommended)
   cors_allowed_origins = [
       "https://app.example.com",
       "https://dashboard.example.com"
   ]

Trusted Hosts
^^^^^^^^^^^^^

Protect against Host header attacks:

.. code-block:: toml

   [server]
   trusted_hosts = ["api.example.com", "vectordb.internal.com"]

Proxy Configuration
~~~~~~~~~~~~~~~~~~~

When running behind reverse proxies (Nginx, Apache, load balancers):

.. code-block:: toml

   [server]
   proxy_enabled = true

   # For single proxy (most common)
   proxy_settings = { x_for = 1, x_proto = 1 }

   # For multiple proxies
   proxy_settings = { x_for = 2, x_proto = 1, x_host = 1 }

Common proxy configurations:

.. code-block:: toml

   # Standard reverse proxy
   proxy_settings = { x_for = 1 }

   # Load balancer + reverse proxy
   proxy_settings = { x_for = 2, x_proto = 1 }

   # Complex proxy chain
   [server.proxy_settings]
   x_for = 3
   x_proto = 1
   x_host = 1
   x_port = 1


Nginx Example Configuration:

.. code-block:: nginx

   upstream vectordb {
       server 127.0.0.1:5000;
   }

   server {
       listen 80;
       server_name api.example.com;

       location / {
           proxy_pass http://vectordb;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }

Logging Configuration
---------------------

Structured Logging
~~~~~~~~~~~~~~~~~~

LocalVectorDB Server supports comprehensive logging with structured output:

.. code-block:: toml

   [server]
   log_level = "INFO"                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
   enable_structured_logging = true     # JSON format logs
   enable_performance_logging = true    # Performance metrics
   auth_log_level = "INFO"              # Authentication events
   security_log_level = "WARNING"       # Security events

Log Destinations
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Console only (development)
   lvdb serve --log-level DEBUG

   # File logging (production)
   lvdb serve --config production.toml  # Uses config file settings

Log file structure in production:

.. code-block:: text

   .lvdb/
   └── logs/
       ├── localvectordb.log          # Main application log
       ├── localvectordb_security.log # Security events
       ├── localvectordb_errors.log   # Error details
       └── localvectordb_performance.log # Performance metrics

Log Rotation
^^^^^^^^^^^^

Logs automatically rotate when they reach 10MB, keeping 5 backup files.

Structured Log Format
^^^^^^^^^^^^^^^^^^^^^

When structured logging is enabled, logs are in JSON format:

.. code-block:: json

   {
     "timestamp": "2024-12-01T10:30:45.123Z",
     "level": "INFO",
     "logger": "localvectordb_server.routes",
     "message": "Database operation: search completed successfully",
     "request_id": "req_abc123",
     "api_key_hash": "hash_xyz789",
     "method": "POST",
     "path": "/api/v1/mydb/search",
     "remote_addr": "10.0.1.100",
     "operation_type": "search",
     "operation": "vector_search",
     "duration_seconds": 0.245,
     "database_name": "mydb",
     "result_count": 5
   }

Performance Monitoring
^^^^^^^^^^^^^^^^^^^^^^

Enable performance logging to track operation timing:

.. code-block:: toml

   [server]
   enable_performance_logging = true

This creates detailed metrics for:

- Request/response times
- Database operation duration
- Search performance by type
- Cache hit/miss rates
- Authentication timing

Multi-Worker Deployment
-----------------------

LocalVectorDB Server supports multi-worker deployments for improved performance and reliability.

Database Registry
~~~~~~~~~~~~~~~~~

Multiple workers need to coordinate database access. Configure a shared registry:

**File-Based Registry** (Single machine)

.. code-block:: toml

   [server]
   db_registry_type = "FileSystemCache"
   db_registry_settings = { cache_dir = "./.lvdb/registry_cache" }

**Redis Registry** (Distributed)

.. code-block:: toml

   [server]
   db_registry_type = "RedisCache"

   [server.db_registry_settings]
   host = "redis.example.com"
   port = 6379
   db = 1
   password = "$REDIS_PASSWORD"


Deployment Architectures
~~~~~~~~~~~~~~~~~~~~~~~~

**Single Machine, Multiple Workers**

.. code-block:: bash

   # Use file-based registry and cache
   lvdb config init --multi-worker --enable-cache --cache-type file

**Distributed Deployment**

.. code-block:: bash

   # Use Redis for coordination
   lvdb config init \
     --redis-registry redis://redis.internal:6379/1 \
     --enable-cache --cache-type redis \
     --cache-redis-url redis://redis.internal:6379/0

Production Deployment Examples
------------------------------

Docker Deployment
~~~~~~~~~~~~~~~~~

**Dockerfile**

.. code-block:: dockerfile

   FROM python:3.12-slim

   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   # Create directories
   RUN mkdir -p /app/data /app/logs

   # Non-root user
   RUN useradd -m -u 1000 vectordb
   RUN chown -R vectordb:vectordb /app
   USER vectordb

   EXPOSE 5000

   CMD ["lvdb", "serve", "--config", "/app/config/production.toml"]

**docker-compose.yml**

.. code-block:: yaml

   version: '3.8'

   services:
     redis:
       image: redis:7-alpine
       command: redis-server --requirepass ${REDIS_PASSWORD}
       volumes:
         - redis_data:/data

     vectordb:
       build: .
       ports:
         - "5000:5000"
       environment:
         - LVDB_SERVER_CONFIG=/app/config/production.toml
         - REDIS_PASSWORD=${REDIS_PASSWORD}
       volumes:
         - ./config:/app/config:ro
         - vectordb_data:/app/data
         - vectordb_logs:/app/logs
       depends_on:
         - redis
       deploy:
         replicas: 3

     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf:ro
         - ./ssl:/etc/nginx/ssl:ro
       depends_on:
         - vectordb

   volumes:
     redis_data:
     vectordb_data:
     vectordb_logs:

Kubernetes Deployment
~~~~~~~~~~~~~~~~~~~~~

**deployment.yaml**

.. code-block:: yaml

   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: localvectordb-server
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: localvectordb-server
     template:
       metadata:
         labels:
           app: localvectordb-server
       spec:
         containers:
         - name: vectordb
           image: localvectordb-server:latest
           ports:
           - containerPort: 5000
           env:
           - name: LVDB_SERVER_CONFIG
             value: "/config/production.toml"
           - name: REDIS_PASSWORD
             valueFrom:
               secretKeyRef:
                 name: redis-secret
                 key: password
           volumeMounts:
           - name: config
             mountPath: /config
             readOnly: true
           - name: data
             mountPath: /app/data
           resources:
             limits:
               memory: "1Gi"
               cpu: "500m"
             requests:
               memory: "512Mi"
               cpu: "250m"
           livenessProbe:
             httpGet:
               path: /api/v1/health
               port: 5000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /api/v1/health
               port: 5000
             initialDelaySeconds: 5
             periodSeconds: 5
         volumes:
         - name: config
           configMap:
             name: vectordb-config
         - name: data
           persistentVolumeClaim:
             claimName: vectordb-data

Gunicorn Deployment
~~~~~~~~~~~~~~~~~~~

For production WSGI deployment:

**gunicorn.conf.py**

.. code-block:: python

   import os

   bind = "0.0.0.0:5000"
   workers = 4
   worker_class = "sync"
   worker_connections = 1000
   max_requests = 1000
   max_requests_jitter = 50
   timeout = 300
   keepalive = 5

   # Logging
   accesslog = "/app/logs/access.log"
   errorlog = "/app/logs/error.log"
   loglevel = "info"

   # Process naming
   proc_name = "localvectordb-server"

   # Worker management
   preload_app = True

   def when_ready(server):
       server.log.info("Server is ready. Spawning workers")

   def worker_int(worker):
       worker.log.info("worker received INT or QUIT signal")

**Deployment script**

.. code-block:: bash

   #!/bin/bash

   # Create application factory
   cat > wsgi.py << 'EOF'
   from localvectordb_server import create_app

   app = create_app(configuration="config/production.toml")

   if __name__ == "__main__":
       app.run()
   EOF

   # Start with Gunicorn
   gunicorn wsgi:app -c gunicorn.conf.py

Performance Tuning
------------------

Database Optimization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [database]
   connection_pool_size = 20        # Increase for high concurrency
   timeout = 600                    # Longer timeout for large operations
   chunk_size = 512                 # Optimize for your content
   chunk_overlap = 2                # Balance context vs. performance

Server Optimization
~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [server]
   max_request_size = 50_000_000    # 50MB for large document uploads
   enable_rate_limiting = true      # Prevent abuse
   rate_limit = "1000 per hour"     # Adjust based on usage patterns

Caching Strategy
~~~~~~~~~~~~~~~~

.. code-block:: toml

   [server]
   cache_enabled = true
   cache_timeout = 1800            # 30 minutes for search results
   cache_type = "RedisCache"       # Best performance for multi-worker

Monitoring and Alerting
-----------------------

Health Checks
~~~~~~~~~~~~~

The ``/api/v1/health`` endpoint provides system status:

.. code-block:: bash

   curl http://localhost:5000/api/v1/health

Response:

.. code-block:: json

   {
     "status": "healthy",
     "version": "1.0.0",
     "ollama_available": true,
     "timestamp": "2024-12-01T10:30:45Z"
   }

Key Metrics to Monitor
~~~~~~~~~~~~~~~~~~~~~~

1. **Response Times**: Monitor ``/api/v1/health`` and search endpoints
2. **Error Rates**: Track 4xx/5xx responses
3. **Database Growth**: Monitor document and chunk counts
4. **Cache Performance**: Hit/miss ratios
5. **API Key Usage**: Monitor for unusual patterns
6. **Resource Usage**: CPU, memory, disk space

Example monitoring script:

.. code-block:: bash

   #!/bin/bash

   # Check health endpoint
   curl -f http://localhost:5000/api/v1/health || exit 1

   # Check API key stats
   API_STATS=$(lvdb auth status --output json)
   EXPIRING_KEYS=$(echo "$API_STATS" | jq '.database_keys.stats.expiring_soon')

   if [ "$EXPIRING_KEYS" -gt 5 ]; then
       echo "WARNING: $EXPIRING_KEYS API keys expiring soon"
   fi

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**High Memory Usage**
   - Reduce ``chunk_size`` or ``connection_pool_size``
   - Enable disk-based caching instead of memory caching
   - Monitor for memory leaks in long-running processes

**Slow Search Performance**
   - Enable Redis caching
   - Optimize embedding model selection
   - Check database size and consider sharding

**Authentication Failures**
   - Verify API key expiration: ``lvdb auth list-keys``
   - Check proxy configuration for rate limiting
   - Review security logs for patterns

**Rate Limiting Issues**
   - Adjust rate limits for your usage patterns
   - Implement client-side retry logic with exponential backoff
   - Consider using different keys for different access patterns

Debug Mode
~~~~~~~~~~

Enable debug mode for development:

.. code-block:: bash

   lvdb serve --debug --log-level DEBUG

This provides:

- Detailed error tracebacks
- Request/response logging
- Performance timing information
- Configuration validation details

Log Analysis
~~~~~~~~~~~~

Query logs for specific patterns:

.. code-block:: bash

   # Find authentication failures
   grep "authentication.*failed" /app/logs/localvectordb_security.log

   # Monitor slow operations
   jq 'select(.duration_seconds > 5)' /app/logs/localvectordb_performance.log

   # Check error patterns
   jq 'select(.level == "ERROR")' /app/logs/localvectordb.log | head -10

This comprehensive guide covers the advanced configuration and deployment options for LocalVectorDB Server.
For basic setup and API usage, refer to the :doc:`Quick Start <../quickstart>` and :doc:`API Documentation <../modules/index>` sections.
