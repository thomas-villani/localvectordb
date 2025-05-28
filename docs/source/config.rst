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
* **YAML**: Popular for DevOps, hierarchical structure
* **JSON**: Machine-readable, no comments
* **INI**: Legacy format, limited nesting

Configuration File Locations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configuration files are loaded in order of precedence:

1. **Explicit path**: ``--config /path/to/config.toml``
2. **Environment variable**: ``LVDB_SERVER_CONFIG``
3. **Current directory**: ``./server-cfg.toml``
4. **Instance directory**: ``./instance/server-cfg.toml``
5. **Home directory**: ``~/localvectordb_server/server-cfg.toml``

Creating Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create default configuration
   lvdb config init --format toml

   # Create with predefined schema
   lvdb config init --schema research_papers --output research.toml

   # Create YAML configuration
   lvdb config init --format yaml --output server.yaml

Configuration Sections
----------------------

Database Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [database]
   # Storage location
   root_dir = "./.lvdb"

   # Connection settings
   timeout = 300                    # Connection timeout in seconds
   connection_pool_size = 10        # Number of database connections

   # Performance settings
   enable_gpu = false               # Use GPU for FAISS if available
   enable_fts = true                # Enable full-text search (FTS5)
   auto_save_interval = 300         # Auto-save interval (0 = disabled)

   # Default settings for new databases
   chunk_size = 500                 # Maximum tokens per chunk
   chunk_overlap = 1                # Overlap between chunks
   chunking_method = "sentences"    # Default chunking method
   embedding_model = "nomic-embed-text"  # Default embedding model
   provider = "ollama"              # Default embedding provider

   # Migration settings
   migration_auto_detect = true
   migration_backup_on_migrate = true
   migration_backup_dir = "./backups"

   # Default metadata schema for new databases
   [database.metadata_schema]
   title = {type = "text", indexed = true}
   author = {type = "text", indexed = true}
   date = {type = "date", indexed = true}
   tags = {type = "json"}

Embedding Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [embedding]
   # Primary embedding provider
   provider = "ollama"              # ollama, openai
   model = "nomic-embed-text"       # Model name
   base_url = "http://localhost:11434"  # Provider-specific URL
   api_key = ""                     # API key for cloud providers
   batch_size = 64                  # Batch size for embedding generation
   timeout = 30                     # Request timeout in seconds
   max_retries = 3                  # Number of retry attempts

   # Provider-specific configuration
   [embedding.config]
   # Custom provider settings go here

Server Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [server]
   # Basic server settings
   host = "127.0.0.1"             # Interface to bind to
   port = 5000                    # Port to listen on
   log_level = "INFO"             # DEBUG, INFO, WARNING, ERROR, CRITICAL
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

   # Performance settings
   max_request_size = 104857600   # 100MB max request size
   request_timeout = 300          # Request timeout in seconds

   # Security settings
   require_api_key = false
   authorized_api_keys = []
   api_key_header = "Authorization"

   # CORS settings
   cors_enabled = true
   cors_allowed_origins = "*"
   cors_allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
   cors_allowed_headers = ["Content-Type", "Authorization"]
   cors_max_age = 86400

   # Rate limiting
   rate_limit_enabled = false
   rate_limit_requests_per_minute = 100
   rate_limit_burst = 20

.. TODO: update with new config options for key management

Security Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [server.security]
   # API key authentication
   require_api_key = true
   authorized_api_keys = [
       "sk-1234567890abcdef1234567890abcdef",
       "sk-abcdef1234567890abcdef1234567890"
   ]
   api_key_header = "Authorization"

   # CORS configuration for web applications
   cors_enabled = true
   cors_allowed_origins = [
       "https://myapp.example.com",
       "https://admin.example.com"
   ]
   cors_allowed_methods = ["GET", "POST", "PUT", "DELETE"]
   cors_allowed_headers = ["Content-Type", "Authorization", "X-Requested-With"]
   cors_max_age = 3600

   # Rate limiting (requires Redis for distributed setups)
   rate_limit_enabled = true
   rate_limit_requests_per_minute = 1000
   rate_limit_burst = 100

Migration Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [migration]
   # Automatic detection and migration of v1.x databases
   auto_detect = true
   backup_on_migrate = true
   backup_dir = "./backups"
   preserve_v1_metadata = true
   default_v2_schema = "documents"
   migration_batch_size = 1000
   verify_migration = true

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
   export LVDB_SERVER_AUTHORIZED_API_KEYS='["key1", "key2"]'

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
   auto_save_interval = 600

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
   request_timeout = 600
   worker_count = 8
   enable_async_processing = true
   enable_performance_metrics = true

   # Security for production
   require_api_key = true
   authorized_api_keys = [
       "sk-prod-key-1234567890abcdef",
       "sk-admin-key-abcdef1234567890"
   ]

   # CORS for web applications
   cors_enabled = true
   cors_allowed_origins = [
       "https://app.yourdomain.com",
       "https://admin.yourdomain.com"
   ]

   # Rate limiting
   rate_limit_enabled = true
   rate_limit_requests_per_minute = 10000
   rate_limit_burst = 1000

Multi-Tenant Setup
^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   # multi-tenant.toml
   [database]
   root_dir = "/data/tenant_databases"
   connection_pool_size = 100
   timeout = 300

   # Separate databases per tenant with consistent settings
   chunk_size = 500
   chunking_method = "sentences"
   chunk_overlap = 1

   [embedding]
   provider = "openai"  # Consistent provider across tenants
   model = "text-embedding-3-small"
   api_key = "${OPENAI_API_KEY}"
   batch_size = 100

   [server]
   host = "0.0.0.0"
   port = 8080
   log_level = "INFO"

   # Strict security for multi-tenant
   require_api_key = true
   authorized_api_keys = [
       "tenant-a-key-1234567890",
       "tenant-b-key-abcdef1234",
       "admin-key-9876543210"
   ]

   # Rate limiting to prevent tenant abuse
   rate_limit_enabled = true
   rate_limit_requests_per_minute = 1000
   rate_limit_burst = 200

   # Logging for audit trails
   enable_request_logging = true
   enable_performance_metrics = true

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
   port = 5000
   log_level = "DEBUG"
   log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

   # Relaxed security for development
   require_api_key = false
   cors_enabled = true
   cors_allowed_origins = "*"

   # No rate limiting in development
   rate_limit_enabled = false

Security Considerations
-----------------------

API Key Management
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate secure API keys
   python -c "import secrets; print('sk-' + secrets.token_hex(32))"

   # Store in environment variables
   export LVDB_API_KEYS='["sk-prod-key", "sk-admin-key"]'

   # Or use external key management
   export LVDB_API_KEYS_FILE="/etc/localvectordb/api_keys.json"

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

   # Batch processing optimization
   auto_save_interval = 1800        # Less frequent saves for performance

   # GPU acceleration
   enable_gpu = true                # If NVIDIA GPU available

   [embedding]
   # Optimize batch sizes
   batch_size = 128                 # Larger batches for throughput
   timeout = 120                    # Longer timeout for large batches

   [server]
   # Increase worker processes
   worker_count = 16                # Match CPU cores

   # Optimize request handling
   max_request_size = 209715200     # 200MB for large document uploads
   request_timeout = 600            # Longer timeout for large operations
   enable_async_processing = true   # Async processing where possible

Monitoring and Metrics
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [server]
   enable_performance_metrics = true
   enable_request_logging = true

   # Custom log format for structured logging
   log_format = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'

.. code-block:: bash

   # Enable structured logging
   export LVDB_SERVER_LOG_FORMAT='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'

   # Log to file
   lvdb serve --config production.toml 2>&1 | tee server.log

   # Monitor with external tools
   tail -f server.log | jq '.message'

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
   RUN pip install localvectordb[server]

   # Create app directory
   WORKDIR /app

   # Copy configuration
   COPY production.toml /app/server-cfg.toml

   # Create data directory
   RUN mkdir -p /data/vector_databases
   VOLUME /data/vector_databases

   # Expose port
   EXPOSE 8080

   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
       CMD curl -f http://localhost:8080/api/v1/health || exit 1

   # Start server
   CMD ["lvdb", "serve", "--config", "/app/server-cfg.toml", "--host", "0.0.0.0", "--port", "8080"]

.. code-block:: yaml

   version: "3.8"

   services:
     localvectordb:
       build: .
       ports:
         - "8080:8080"
       volumes:
         - vector_data:/data/vector_databases
         - ./production.toml:/app/server-cfg.toml:ro
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
             mountPath: /app/server-cfg.toml
             subPath: server-cfg.toml
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
   ExecStart=/opt/localvectordb/venv/bin/lvdb serve --config /etc/localvectordb/production.toml
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
   lvdb db my_database list --json > /backup/my_database_docs_$(date +%Y%m%d).json

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

High Memory Usage:

.. code-block:: toml

   # Reduce memory usage
   [database]
   connection_pool_size = 5        # Reduce connections
   auto_save_interval = 300        # More frequent saves

   [embedding]
   batch_size = 32                 # Smaller batches

Slow Performance:

.. code-block:: toml

   # Performance tuning
   [database]
   enable_gpu = true               # Use GPU if available
   connection_pool_size = 20       # More connections

   [server]
   worker_count = 16               # More workers
   enable_async_processing = true  # Async processing

Connection Errors:

.. code-block:: bash

   # Check server status
   lvdb health

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
