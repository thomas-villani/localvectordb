Server Configuration
====================

LocalVectorDB Server supports comprehensive configuration management for production deployments, including security,
performance tuning, and operational considerations.

Configuration Files
-------------------

Supported Formats
^^^^^^^^^^^^^^^^^

LocalVectorDB supports multiple configuration formats:

* **TOML** (recommended): Human-readable, supports comments
* **JSON**: Machine-readable, no comments

Configuration File Locations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configuration files are loaded in order of precedence:

1. **Explicit path**: ``--config /path/to/.lvdb-config.toml``
2. **Environment variable**: ``LVDB_SERVER_CONFIG``
3. **Current directory**: ``./.lvdb-config.toml``
4. **Instance directory**: ``./instance/.lvdb-config.toml``
5. **Home directory**: ``~/localvectordb_server/.lvdb-config.toml``

Creating Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create default configuration
   lvdb config init --format toml

   # Create with predefined schema
   lvdb config init --schema research_papers --output research.toml


Configuration Sections
----------------------

For a complete overview of the configuration settings, see the :doc:`Configuration Parameters Documentation <config.params>`.

Database Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [database]
   root_dir = "./.lvdb"
   timeout = 300
   connection_pool_size = 10
   enable_gpu = false
   enable_fts = true
   chunk_size = 500
   chunk_overlap = 1
   chunking_method = "lines"

   # Faiss index parameters (optional, for advanced users only)
   faiss_index_type = "IndexFlatL2"
   # faiss_index_hnsw_flat_neighbors = 16
   # faiss_index_lsh_bits = 1536

Embedding Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [embedding]
   # Primary embedding provider
   provider = "ollama"                  # ollama, openai
   model = "nomic-embed-text"           # Model name
   base_url = "http://localhost:11434"  # Provider-specific URL
   api_key = "api-key-here"             # API key for cloud providers
   batch_size = 64                      # Batch size for embedding generation
   timeout = 30                         # Request timeout in seconds
   max_retries = 3                      # Number of retry attempts

   # Provider-specific configuration
   [embedding.config]
   # Custom provider settings go here

Server Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [server]
   # Basic server settings
   host = "127.0.0.1"             # Interface to bind to
   port = 8000                    # Port to listen on
   log_level = "INFO"             # DEBUG, INFO, WARNING, ERROR, CRITICAL
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

   # Performance settings
   max_request_size = 104857600   # 100MB max request size


Security Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [server.security]
   # API-key authentication
   require_api_key = false
   key_database_path = "path/to/key/store.db"   # Provide a path for the key store database, otherwise it will be in the database `root_dir`
   api_key_header = "Authorization"             # Optionally use a different header for the api key
   auto_prune_expired_keys = false
   key_audit_logging = true
   auth_log_level = "INFO"
   warn_expiring_days = 7

   # CORS settings for web applications
   cors_enabled = true
   cors_allowed_origins = [
       "https://myapp.example.com",
       "https://admin.example.com"
   ]
   cors_allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
   cors_allowed_headers = ["Content-Type", "Authorization"]
   cors_max_age = 86400


Extraction Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Controls how uploaded files are converted to text (via all2md). Defaults are
hardened for untrusted uploads; relax them only for trusted content. See
:doc:`/file-extraction` for the full security model.

.. code-block:: toml

   [extraction]
   allow_remote_fetch = false           # Fetch remote assets referenced by a document (SSRF risk)
   allowed_hosts = []                   # Host allowlist applied when allow_remote_fetch = true
   strip_dangerous_elements = true      # HTML: strip scripts and event handlers
   attachment_mode = "skip"             # How embedded attachments/assets are handled


Environment Variables
---------------------

All configuration options can be overridden with environment variables using the ``LVDB_`` prefix:

.. code-block:: bash

   # Database settings
   export LVDB_DATABASE_ROOT_DIR="/data/vector_databases"
   export LVDB_DATABASE_TIMEOUT=600
   export LVDB_DATABASE_ENABLE_GPU=true

   # Embedding settings
   export LVDB_EMBEDDING_PROVIDER="openai"
   export LVDB_EMBEDDING_MODEL="text-embedding-3-small"
   export LVDB_EMBEDDING_API_KEY="your_openai_key"

   # Server settings
   export LVDB_SERVER_HOST="0.0.0.0"
   export LVDB_SERVER_PORT=8080
   export LVDB_SERVER_LOG_LEVEL="DEBUG"

   # Security settings
   export LVDB_SERVER_REQUIRE_API_KEY=true

   # Extraction settings
   export LVDB_EXTRACTION_ALLOW_REMOTE_FETCH=false
   export LVDB_EXTRACTION_STRIP_DANGEROUS_ELEMENTS=true
   export LVDB_EXTRACTION_ATTACHMENT_MODE="skip"

   # Start server with environment overrides
   lvdb serve

Production Configuration Examples
---------------------------------

High-Performance Setup
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   # production-high-perf.toml
   [database]
   root_dir = "/data/vector_databases"
   connection_pool_size = 50
   enable_gpu = true
   enable_fts = true

   # Optimized chunking
   chunk_size = 400
   chunk_overlap = 2
   chunking_method = "sentences"

   [embedding]
   provider = "ollama"
   model = "nomic-embed-text"
   base_url = "http://ollama-cluster:11434"
   batch_size = 128
   timeout = 60
   max_retries = 5

   [server]
   host = "0.0.0.0"
   port = 8080
   log_level = "INFO"
   max_request_size = 209715200  # 200MB
   enable_performance_logging = true

   [server.security]
   # Security for production
   require_api_key = true
   api_key_header = "Authorization"
   auto_prune_expired_keys = false
   key_audit_logging = true
   auth_log_level = "INFO"
   warn_expiring_days = 7

   # CORS for web applications
   cors_enabled = true
   cors_allowed_origins = [
       "https://app.yourdomain.com",
       "https://admin.yourdomain.com"
   ]



Development Setup
^^^^^^^^^^^^^^^^^

.. code-block:: toml

   # development.toml
   [database]
   root_dir = "./dev_databases"
   connection_pool_size = 5
   enable_gpu = false
   enable_fts = true

   [embedding]
   provider = "ollama"
   model = "all-minilm"  # Faster model for development
   base_url = "http://localhost:11434"
   batch_size = 32

   [server]
   host = "127.0.0.1"
   port = 8000
   log_level = "DEBUG"
   log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

   [server.security]
   # Relaxed security for development
   require_api_key = false
   cors_enabled = true
   cors_allowed_origins = "*"


Security Considerations
-----------------------

API Key Management
^^^^^^^^^^^^^^^^^^

LocalVectorDB Server includes a comprehensive key management system with SQLite-based storage, bcrypt hashing, full audit trails, and permission-based access control.

Permission Levels
~~~~~~~~~~~~~~~~~

API keys now support two permission levels:

* **read_only** - Can query databases, search documents, and retrieve data. Cannot create, update, or delete any resources.
* **read_write** - Full access to all operations including creating databases, adding documents, and deleting resources.

Creating API Keys
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create a read-write API key (default)
   lvdb auth create-key --description "Production API access"

   # Create a read-only key for monitoring/analytics
   lvdb auth create-key --description "Monitoring dashboard" --permission-level read_only

   # Create a read-write key with expiration
   lvdb auth create-key --description "Admin access" --permission-level read_write --expires-days 30

   # Create a read-only key for CI/CD testing
   lvdb auth create-key --description "CI/CD Pipeline" --permission-level read_only --created-by "admin" --expires-days 90

   # Output just the key for scripting
   lvdb auth create-key --description "Script access" --permission-level read_only --format key-only

   # Output as JSON for automation
   lvdb auth create-key --description "API integration" --format json

Managing API Keys
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # List all API keys
   lvdb auth list-keys

   # List only active keys
   lvdb auth list-keys --active-only

   # Show detailed statistics
   lvdb auth list-keys --show-stats

   # Get detailed information about a specific key
   lvdb auth key-info key_20241201_abc123

   # Check overall authentication status
   lvdb auth status

Key Rotation and Security
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Rotate a key (creates new key, deactivates old one)
   lvdb auth rotate-key key_20241201_abc123

   # Revoke a key immediately
   lvdb auth revoke-key key_20241201_abc123

   # Remove expired keys (soft delete - deactivates)
   lvdb auth prune-expired

   # Permanently delete expired keys
   lvdb auth prune-expired --hard-delete

   # Preview what would be pruned
   lvdb auth prune-expired --dry-run

Configuration
~~~~~~~~~~~~~

Configure key management in your server configuration:

.. code-block:: toml

   [server.security]
   # Enable API key authentication
   require_api_key = true

   # Key database location (optional, defaults to <root_dir>/api_keys.db)
   key_database_path = "/secure/path/api_keys.db"

   # API key header name
   api_key_header = "Authorization"

   # Automatically remove expired keys
   auto_prune_expired_keys = false

   # Enable audit logging
   key_audit_logging = true
   auth_log_level = "INFO"

   # Warn about keys expiring soon
   warn_expiring_days = 7

Using API Keys
~~~~~~~~~~~~~~

API keys must be sent in the Authorization header as Bearer tokens:

.. code-block:: bash

   # Using curl
   curl -H "Authorization: Bearer lvdb_your_api_key_here" \
        http://localhost:8080/api/v1/databases

.. code-block:: python

   # Using Python requests
   import requests

   headers = {"Authorization": "Bearer lvdb_your_api_key_here"}
   response = requests.get("http://localhost:8080/api/v1/health", headers=headers)

Permission Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the principle of least privilege when assigning API key permissions:

* **Use read-only keys** for:
  
  - Monitoring and analytics dashboards
  - Public-facing search interfaces
  - CI/CD test runners that only validate functionality
  - Backup verification scripts
  - Report generation tools

* **Use read-write keys** for:
  
  - Administrative interfaces
  - Data ingestion pipelines
  - Content management systems
  - Database maintenance scripts
  - Development environments (with short expiration)

* **Security recommendations**:
  
  - Always set expiration dates for read-write keys
  - Rotate keys regularly, especially after personnel changes
  - Use descriptive names to track key usage
  - Monitor key usage through audit logs
  - Revoke unused keys promptly

Security Features
~~~~~~~~~~~~~~~~~

The key management system provides enterprise-grade security:

* **Secure Storage**: Keys are hashed with bcrypt before storage
* **Expiration Support**: Keys can have automatic expiration dates
* **Audit Logging**: All key usage is logged for security monitoring
* **Key Rotation**: Seamlessly rotate keys without service interruption
* **Usage Tracking**: Monitor when keys were last used
* **Soft Deletion**: Revoked keys are deactivated, not deleted (for audit trails)

Automation and CI/CD
~~~~~~~~~~~~~~~~~~~~

For automated deployments and CI/CD pipelines:

.. code-block:: bash

   # Create a key for automation (outputs only the key)
   API_KEY=$(lvdb auth create-key --description "CI/CD Pipeline" --expires-days 365 --format key-only)

   # Use in scripts
   export LVDB_API_KEY="$API_KEY"

   # Rotate keys programmatically
   NEW_KEY=$(lvdb auth rotate-key $OLD_KEY_ID --format key-only)

Monitoring and Maintenance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Monitor key usage and expiration
   lvdb auth list-keys --show-stats

   # Check for keys expiring soon
   lvdb auth list-keys | grep -E "(EXPIRING|EXPIRED)"

   # Set up automated cleanup (add to cron)
   0 2 * * * /usr/local/bin/lvdb auth prune-expired --confirm

HTTPS Configuration
^^^^^^^^^^^^^^^^^^^

LocalVectorDB server runs on HTTP by default. For HTTPS, use a reverse proxy:

Nginx Configuration:

.. code-block:: nginx

   server {
       listen 443 ssl;
       server_name vectordb.yourdomain.com;

       ssl_certificate /path/to/certificate.crt;
       ssl_certificate_key /path/to/private.key;

       location / {
           proxy_pass http://127.0.0.1:8080;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forward_for;
           proxy_set_header X-Forwarded-Proto $scheme;

           # Handle large requests
           client_max_body_size 200M;
           proxy_read_timeout 300s;
           proxy_send_timeout 300s;
       }
   }

Network Security
^^^^^^^^^^^^^^^^

.. code-block:: toml

   # Restrict access to specific networks
   [server]
   host = "127.0.0.1"  # Local access only

   # Or bind to specific interface
   host = "10.0.1.100"  # Internal network only

.. code-block:: bash

   # Use firewall rules
   iptables -A INPUT -p tcp --dport 8080 -s 10.0.0.0/8 -j ACCEPT
   iptables -A INPUT -p tcp --dport 8080 -j DROP

Performance Tuning
------------------

Hardware Recommendations
^^^^^^^^^^^^^^^^^^^^^^^^

* **CPU**:
  - Minimum: 4 cores
  - Recommended: 8+ cores for high-throughput
  - Consider ARM64 for efficiency (M1/M2 Macs, AWS Graviton)
* **Memory**:
  - Minimum: 8GB RAM
  - Recommended: 16GB+ for large databases
  - Rule of thumb: 2–4 GB per million documents
* **Storage**:
  - SSD strongly recommended for database files
  - NVMe SSD for high-performance setups
  - Separate storage for databases and logs
* **GPU** (Optional):
  - NVIDIA GPU with CUDA support for FAISS acceleration
  - Minimum 8GB VRAM for large embeddings
  - Multi-GPU setups supported

Performance Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [database]
   # Optimize connection pooling
   connection_pool_size = 20        # Increase for high concurrency

   # Longer timeout for large operations
   timeout = 600

   # GPU acceleration
   enable_gpu = true                # If NVIDIA GPU available

   [embedding]
   # Optimize batch sizes
   batch_size = 128                 # Larger batches for throughput
   timeout = 120                    # Longer timeout for large batches

   [server]
   # Optimize request handling
   max_request_size = 209715200     # 200MB for large document uploads


Deployment Strategies
---------------------

Docker Deployment
^^^^^^^^^^^^^^^^^

.. code-block:: dockerfile

   FROM python:3.11-slim

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       curl \
       && rm -rf /var/lib/apt/lists/*

   # Install Ollama
   RUN curl -fsSL https://ollama.ai/install.sh | sh

   # Install LocalVectorDB
   RUN pip install "localvectordb[server]"

   # Create app directory
   WORKDIR /app

   # Copy configuration
   COPY production.toml /app/.lvdb-config.toml

   # Create data directory
   RUN mkdir -p /data/vector_databases
   VOLUME /data/vector_databases

   # Expose port
   EXPOSE 8080

   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
       CMD curl -f http://localhost:8080/api/v1/health || exit 1

   # Start server
   CMD ["lvdb", "--config", "/app/.lvdb-config.toml", "serve", "--host", "0.0.0.0", "--port", "8080"]

.. code-block:: yaml

   version: "3.12"

   services:
     localvectordb:
       build: .
       ports:
         - "8080:8080"
       volumes:
         - vector_data:/data/vector_databases
         - ./production.toml:/app/.lvdb-config.toml:ro
       environment:
         - LVDB_EMBEDDING_PROVIDER=ollama
         - LVDB_EMBEDDING_BASE_URL=http://ollama:11434
       depends_on:
         - ollama
       restart: unless-stopped

     ollama:
       image: ollama/ollama:latest
       ports:
         - "11434:11434"
       volumes:
         - ollama_data:/root/.ollama
       restart: unless-stopped

     nginx:
       image: nginx:alpine
       ports:
         - "443:443"
         - "80:80"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf:ro
         - ./ssl:/etc/ssl:ro
       depends_on:
         - localvectordb
       restart: unless-stopped

   volumes:
     vector_data:
     ollama_data:

Kubernetes Deployment
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: localvectordb
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: localvectordb
     template:
       metadata:
         labels:
           app: localvectordb
       spec:
         containers:
         - name: localvectordb
           image: localvectordb:latest
           ports:
           - containerPort: 8080
           env:
           - name: LVDB_DATABASE_ROOT_DIR
             value: "/data/vector_databases"
           - name: LVDB_EMBEDDING_PROVIDER
             value: "ollama"
           - name: LVDB_EMBEDDING_BASE_URL
             value: "http://ollama-service:11434"
           volumeMounts:
           - name: vector-data
             mountPath: /data/vector_databases
           - name: config
             mountPath: /app/.lvdb-config.toml
             subPath: .lvdb-config.toml
           livenessProbe:
             httpGet:
               path: /api/v1/health
               port: 8080
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /api/v1/health
               port: 8080
             initialDelaySeconds: 5
             periodSeconds: 5
         volumes:
         - name: vector-data
           persistentVolumeClaim:
             claimName: vector-data-pvc
         - name: config
           configMap:
             name: localvectordb-config
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: localvectordb-service
   spec:
     selector:
       app: localvectordb
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8080
     type: LoadBalancer

Systemd Service
^^^^^^^^^^^^^^^

.. code-block:: ini

   # /etc/systemd/system/localvectordb.service
   [Unit]
   Description=LocalVectorDB Server
   After=network.target

   [Service]
   Type=simple
   User=vectordb
   Group=vectordb
   WorkingDirectory=/opt/localvectordb
   Environment=LVDB_SERVER_CONFIG=/etc/localvectordb/production.toml
   ExecStart=/opt/localvectordb/venv/bin/lvdb --config /etc/localvectordb/production.toml serve
   Restart=always
   RestartSec=5
   StandardOutput=journal
   StandardError=journal

   # Security settings
   NoNewPrivileges=true
   PrivateTmp=true
   ProtectSystem=strict
   ProtectHome=true
   ReadWritePaths=/data/vector_databases /var/log/localvectordb

   [Install]
   WantedBy=multi-user.target

.. code-block:: bash

   # Create user
   sudo useradd -r -s /bin/false vectordb

   # Create directories
   sudo mkdir -p /opt/localvectordb /etc/localvectordb /data/vector_databases
   sudo chown vectordb:vectordb /data/vector_databases

   # Install service
   sudo systemctl daemon-reload
   sudo systemctl enable localvectordb
   sudo systemctl start localvectordb

   # Check status
   sudo systemctl status localvectordb

Backup and Recovery
-------------------

Database Backup
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Backup database files
   cp -r /data/vector_databases /backup/vector_databases_$(date +%Y%m%d)

   # Backup with compression
   tar -czf /backup/vectordb_backup_$(date +%Y%m%d).tar.gz /data/vector_databases

   # Automated backup script
   #!/bin/bash
   BACKUP_DIR="/backup/vectordb"
   DATE=$(date +%Y%m%d_%H%M%S)
   mkdir -p "$BACKUP_DIR"

   # Stop server gracefully
   systemctl stop localvectordb

   # Create backup
   tar -czf "$BACKUP_DIR/vectordb_backup_$DATE.tar.gz" /data/vector_databases

   # Keep only last 7 days of backups
   find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete

   # Restart server
   systemctl start localvectordb

Configuration Backup
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Backup configuration
   cp /etc/localvectordb/production.toml /backup/config_$(date +%Y%m%d).toml

   # Export database metadata
   lvdb db my_database list --format json > /backup/my_database_docs_$(date +%Y%m%d).json

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

High Memory Usage:

.. code-block:: toml

   # Reduce memory usage
   [database]
   connection_pool_size = 5        # Reduce connections

   [embedding]
   batch_size = 32                 # Smaller batches

Slow Performance:

.. code-block:: toml

   # Performance tuning
   [database]
   enable_gpu = true               # Use GPU if available
   connection_pool_size = 20       # More connections

   [server]
   enable_performance_logging = true  # Surface slow operations in logs

Connection Errors:

.. code-block:: bash

   # Verify configuration
   lvdb config show

   # Check logs
   journalctl -u localvectordb -f

Monitoring Tools
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Server monitoring
   htop  # Monitor CPU/memory usage
   iotop # Monitor disk I/O
   netstat -tlnp | grep 8080  # Check port binding

   # Application monitoring
   curl http://localhost:8080/api/v1/health
   lvdb list --details

   # Log analysis
   tail -f /var/log/localvectordb/server.log | jq .
