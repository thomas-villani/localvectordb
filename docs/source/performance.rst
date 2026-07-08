=====================================
Performance Tuning Guide
=====================================

LocalVectorDB provides comprehensive SQLite performance tuning capabilities to optimize your database for different workloads and system configurations. This guide covers how to use the built-in tuning profiles, auto-tuner, and maintenance commands to achieve optimal performance.

Overview
========

LocalVectorDB uses SQLite with WAL (Write-Ahead Logging) mode and FAISS for vector operations. Performance can be significantly improved by tuning SQLite pragma settings based on your:

- **System resources** (RAM, CPU, disk type)
- **Workload patterns** (read-heavy, write-heavy, mixed)
- **Data durability requirements** (critical, normal, temporary)
- **Memory constraints** (generous, moderate, limited)

Quick Start
===========

The fastest way to optimize your database is using the auto-tuner:

.. code-block:: bash

   # Analyze system and get recommendations
   lvdb tuning auto mydatabase
   
   # Apply recommendations immediately
   lvdb tuning auto mydatabase --apply

For more control, you can manually select profiles:

.. code-block:: bash

   # List available profiles
   lvdb tuning list
   
   # Apply a specific profile
   lvdb tuning set mydatabase read_optimized

Tuning Profiles
===============

LocalVectorDB includes five pre-configured tuning profiles optimized for different scenarios:

Balanced (Default)
------------------

**Best for:** Mixed workloads with balanced read/write operations

.. code-block:: python

   db = LocalVectorDB(name="mydatabase", sqlite_profile="balanced")

**Key settings:**
- 64MB cache per connection
- 256MB memory mapping
- Normal synchronization
- 1000-page WAL checkpoint threshold

**Use when:**
- General-purpose applications
- Unknown or mixed workload patterns
- Starting point for new deployments

Fast Ingest
-----------

**Best for:** High-throughput data ingestion and bulk imports

.. code-block:: python

   db = LocalVectorDB(name="mydatabase", sqlite_profile="fast_ingest")

**Key settings:**
- 256MB cache per connection
- Larger WAL (4000 pages before checkpoint)
- Optimized for write throughput
- 10-second busy timeout for concurrency

**Use when:**
- Batch document processing
- Initial database population
- ETL operations
- Import/migration scenarios

Read Optimized
--------------

**Best for:** Query-heavy applications with frequent searches

.. code-block:: python

   db = LocalVectorDB(name="mydatabase", sqlite_profile="read_optimized")

**Key settings:**
- 128MB cache per connection
- 512MB memory mapping (on suitable systems)
- Optimized query planner statistics
- Reduced busy timeout (3 seconds)

**Use when:**
- Search and retrieval applications
- Production databases with heavy query load
- Real-time search systems
- Analytics and reporting

Durable
-------

**Best for:** Mission-critical data requiring maximum safety

.. code-block:: python

   db = LocalVectorDB(name="mydatabase", sqlite_profile="durable")

**Key settings:**
- FULL synchronization mode
- Frequent WAL checkpoints (100 pages)
- File-based temporary storage
- Enhanced data integrity checks

**Use when:**
- Financial or medical data
- Regulatory compliance requirements
- Critical business data
- Systems where data loss is unacceptable

Memory Saver
------------

**Best for:** Resource-constrained environments

.. code-block:: python

   db = LocalVectorDB(name="mydatabase", sqlite_profile="memory_saver")

**Key settings:**
- 8MB cache per connection
- Memory mapping disabled
- File-based temporary storage
- Frequent WAL checkpoints (500 pages)

**Use when:**
- Edge computing devices
- Docker containers with memory limits
- Embedded systems
- Development environments

Auto-Tuning
===========

The auto-tuner analyzes your system resources and workload to recommend optimal settings.

System Analysis
---------------

The auto-tuner automatically detects:

.. code-block:: bash

   # View system analysis
   lvdb maintenance analyze-system

**Detected information:**
- Total and available RAM
- CPU core count
- Disk type (SSD vs HDD)
- Free disk space
- Operating system

Interactive Mode
----------------

For the most accurate recommendations, use interactive mode:

.. code-block:: bash

   lvdb tuning auto mydatabase --interactive

The interactive interview asks about:

- **Primary use case:** Search/retrieval, bulk ingestion, balanced, batch imports, real-time processing
- **Document size:** Small (<1KB), medium (1-10KB), large (>10KB)
- **Concurrent users:** Single user, small team (2-5), medium (6-20), large (20+)
- **Data durability:** Critical, high, normal, low
- **Memory availability:** Generous, moderate, limited

Non-Interactive Mode
--------------------

You can also specify workload parameters directly:

.. code-block:: bash

   # Specify workload characteristics
   lvdb tuning auto mydatabase \
     --workload-type read_heavy \
     --memory-constraint moderate \
     --durability normal \
     --apply

**Available options:**
- ``--workload-type``: read_heavy, write_heavy, balanced, batch_ingest, real_time
- ``--memory-constraint``: generous, moderate, limited  
- ``--durability``: critical, high, normal, low

Custom Pragma Overrides
========================

You can fine-tune any profile with custom pragma overrides:

.. code-block:: python

   # Python API
   db = LocalVectorDB(
       name="mydatabase",
       sqlite_profile="read_optimized",
       sqlite_pragma_overrides={
           "cache_size": -131072,  # 128MB cache
           "mmap_size": 1073741824,  # 1GB memory mapping
           "busy_timeout": 5000  # 5-second timeout
       }
   )

.. code-block:: bash

   # CLI commands
   lvdb tuning set mydatabase read_optimized \
     --override cache_size=-131072 \
     --override mmap_size=1073741824

Common Pragma Settings
----------------------

**cache_size**
  Memory cache size in KB (negative values) or pages (positive values)
  
  - ``-65536`` = 64MB cache
  - ``-131072`` = 128MB cache
  - ``-262144`` = 256MB cache

**mmap_size**
  Memory-mapped I/O size in bytes
  
  - ``268435456`` = 256MB (good for most systems)
  - ``536870912`` = 512MB (high-memory systems)
  - ``0`` = Disable (memory-constrained systems)

**synchronous**
  Data durability vs performance trade-off
  
  - ``FULL`` = Maximum safety, slower writes
  - ``NORMAL`` = Good balance (recommended)
  - ``OFF`` = Fastest, risk of corruption on crash

**wal_autocheckpoint**
  Pages written before automatic WAL checkpoint
  
  - ``100`` = Frequent checkpoints (durable)
  - ``1000`` = Balanced (default)
  - ``4000`` = Infrequent checkpoints (fast ingest)

Runtime Tuning Changes
======================

You can change tuning settings without restarting your application:

.. code-block:: python

   # Change profile at runtime
   db.set_sqlite_tuning("fast_ingest")
   
   # Add pragma overrides
   db.set_sqlite_tuning("balanced", {"cache_size": -131072})
   
   # Check current settings
   config = db.get_sqlite_tuning()
   print(f"Current profile: {config['profile']}")

CLI Management
==============

The CLI provides comprehensive tuning management:

Viewing Configuration
---------------------

.. code-block:: bash

   # Show current tuning settings
   lvdb tuning get mydatabase
   
   # JSON format for scripting
   lvdb tuning get mydatabase --format json

Applying Settings
-----------------

.. code-block:: bash

   # Apply a profile
   lvdb tuning set mydatabase read_optimized
   
   # Test changes first (dry run)
   lvdb tuning set mydatabase fast_ingest --dry-run
   
   # Set specific pragma values
   lvdb tuning set-pragma mydatabase cache_size -131072

Database Maintenance
====================

Regular maintenance operations help maintain optimal performance:

WAL Checkpoint
--------------

.. code-block:: bash

   # Passive checkpoint (non-blocking)
   lvdb maintenance checkpoint mydatabase
   
   # Truncate WAL file (requires brief exclusive lock)
   lvdb maintenance checkpoint mydatabase --mode TRUNCATE

**Checkpoint modes:**
- ``PASSIVE`` = Non-blocking, best effort
- ``FULL`` = Complete checkpoint, may block briefly
- ``RESTART`` = Restart WAL, requires exclusive access
- ``TRUNCATE`` = Truncate WAL file after checkpoint

Query Optimizer Statistics
---------------------------

.. code-block:: bash

   # Update query planner statistics
   lvdb maintenance optimize mydatabase

Run this after:
- Large data imports
- Significant schema changes
- Index modifications
- Periodic maintenance (weekly/monthly)

Database Compaction
-------------------

.. code-block:: bash

   # Full vacuum (requires exclusive access)
   lvdb maintenance vacuum mydatabase
   
   # Incremental vacuum (less disruptive)
   lvdb maintenance vacuum mydatabase --incremental --pages 1000

**Vacuum operations:**
- **Full VACUUM:** Rebuilds entire database, reclaims all free space
- **Incremental VACUUM:** Reclaims specified number of pages, less disruptive

Performance Monitoring
======================

Monitor these key metrics to evaluate tuning effectiveness:

Database Statistics
-------------------

.. code-block:: python

   # Get database statistics
   stats = db.get_stats()
   
   print(f"Total documents: {stats['documents']}")
   print(f"Total chunks: {stats['chunks']}")
   print(f"Index vectors: {stats['index_vectors']}")

Connection Pool Metrics
-----------------------

.. code-block:: python

   # Monitor connection pool performance
   pool_stats = db.connection_pool.stats
   
   print(f"Pool size: {pool_stats['pool_size']}")
   print(f"Active connections: {pool_stats['created_connections']}")

System Resource Usage
---------------------

.. code-block:: bash

   # Analyze current system resources
   lvdb maintenance analyze-system

Best Practices
==============

Profile Selection Guidelines
----------------------------

1. **Start with auto-tuning** to get baseline recommendations
2. **Use balanced profile** for unknown workloads
3. **Switch to specialized profiles** as workload patterns emerge
4. **Test profile changes** in development before production
5. **Monitor performance metrics** after tuning changes

Development vs Production
-------------------------

**Development:**
- Use ``balanced`` or ``memory_saver`` profiles
- Lower cache sizes to conserve resources
- More frequent checkpoints for safety

**Production:**
- Use workload-specific profiles (``read_optimized``, ``fast_ingest``)
- Larger cache sizes if RAM permits
- Monitor and adjust based on actual usage patterns

Deployment Strategies
---------------------

**Initial deployment:**

.. code-block:: python

   # Start with auto-tuned settings
   db = LocalVectorDB(name="myapp")
   
   # Apply auto-tuning
   recommendation = db.auto_tune(apply=True)
   
   # Log the applied settings
   logging.info(f"Applied profile: {recommendation['profile_name']}")

**Production optimization:**

.. code-block:: bash

   # Weekly maintenance routine
   lvdb maintenance optimize mydatabase
   lvdb maintenance checkpoint mydatabase --mode TRUNCATE
   
   # Monthly system analysis
   lvdb maintenance analyze-system
   lvdb tuning auto mydatabase  # Review recommendations

Troubleshooting
===============

Common Performance Issues
-------------------------

**Slow queries:**
1. Check if query planner statistics are current: ``lvdb maintenance optimize``
2. Increase cache size: ``--override cache_size=-131072``
3. Enable memory mapping on SSD systems: ``--override mmap_size=536870912``

**Slow writes:**
1. Switch to ``fast_ingest`` profile during bulk operations
2. Increase WAL checkpoint threshold: ``--override wal_autocheckpoint=4000``
3. Consider ``synchronous=OFF`` for non-critical data (with caution)

**High memory usage:**
1. Switch to ``memory_saver`` profile
2. Reduce cache size: ``--override cache_size=-32768``
3. Disable memory mapping: ``--override mmap_size=0``

**Lock timeouts:**
1. Increase busy timeout: ``--override busy_timeout=10000``
2. Run WAL checkpoint: ``lvdb maintenance checkpoint mydatabase``
3. Check for long-running transactions

Profile Comparison
==================

.. list-table:: Profile Comparison
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Setting
     - Balanced
     - Fast Ingest
     - Read Optimized
     - Durable
     - Memory Saver
   * - Cache Size
     - 64MB
     - 256MB
     - 128MB
     - 64MB
     - 8MB
   * - Memory Mapping
     - 256MB
     - 256MB
     - 512MB
     - 128MB
     - Disabled
   * - Synchronous
     - NORMAL
     - NORMAL
     - NORMAL
     - FULL
     - NORMAL
   * - WAL Checkpoint
     - 1000 pages
     - 4000 pages
     - 1000 pages
     - 100 pages
     - 500 pages
   * - Best For
     - General use
     - Bulk imports
     - Query-heavy
     - Critical data
     - Low memory

Advanced Topics
===============

Custom Profiles
---------------

You can create custom profiles by extending the existing ones:

.. code-block:: python

   # Create a custom high-performance profile
   custom_overrides = {
       "cache_size": -524288,  # 512MB cache
       "mmap_size": 2147483648,  # 2GB memory mapping
       "wal_autocheckpoint": 10000,  # Large WAL
       "temp_store": "MEMORY",
       "synchronous": "NORMAL"
   }
   
   db.set_sqlite_tuning("read_optimized", custom_overrides)

Backup Integration
------------------

The tuning system integrates with backup operations:

- Current pragma settings are stored in backup metadata
- Backup operations use optimized pragma settings temporarily
- Original settings are restored after backup completion

.. code-block:: python

   # Backup operations automatically optimize performance
   from localvectordb.backup import BackupManager
   
   backup_manager = BackupManager(db.db_path)
   backup_id = backup_manager.create_backup()  # Uses optimized settings

Remote Database Tuning
-----------------------

LocalVectorDB provides full remote tuning capabilities through the HTTP API, allowing administrators to optimize database performance on remote servers without direct server access. This is particularly valuable for production deployments where database servers are managed centrally.

RemoteVectorDB Client Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RemoteVectorDB`` client inherits all tuning capabilities from the ``TuningMixin``, providing the same interface as local databases:

.. code-block:: python

   from localvectordb import RemoteVectorDB

   # Connect to remote database with initial tuning profile
   remote_db = RemoteVectorDB(
       name="mydatabase",
       base_url="http://server:8000",
       api_key="your_api_key",
       sqlite_profile="read_optimized"
   )

   # Same API as local databases
   remote_db.set_sqlite_tuning("fast_ingest")
   config = remote_db.get_sqlite_tuning()

   print(f"Current profile: {config['profile']}")
   print(f"Active pragmas: {config['pragmas']}")

Real-Time Performance Adjustments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remote tuning allows dynamic performance optimization based on changing workload patterns:

.. code-block:: python

   # Switch to fast ingestion during bulk data import
   remote_db.set_sqlite_tuning("fast_ingest", {
       "cache_size": -262144,  # 256MB cache for large operations
       "wal_autocheckpoint": 10000  # Larger WAL for bulk operations
   })

   # Perform bulk import operations
   documents = load_large_dataset()
   remote_db.upsert_documents(documents)

   # Switch back to read-optimized for normal operations
   remote_db.set_sqlite_tuning("read_optimized")

Auto-Tuning Remote Databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The auto-tuner works seamlessly with remote databases, analyzing the server's system resources:

.. code-block:: python

   # Get auto-tuning recommendations for remote server
   recommendation = remote_db.auto_tune(
       workload={
           "workload_type": "read_heavy",
           "document_size": "large",
           "concurrent_users": 50,
           "durability_level": "normal",
           "memory_constraint": "generous"
       }
   )

   print(f"Recommended profile: {recommendation['profile_name']}")
   print(f"Reasoning: {recommendation['reasoning']}")
   print(f"Estimated memory usage: {recommendation['estimated_memory_mb']} MB")

   # Apply recommendations if suitable
   if recommendation['profile_name'] != remote_db.get_sqlite_tuning()['profile']:
       remote_db.set_sqlite_tuning(
           recommendation['profile_name'],
           recommendation['pragma_overrides'],
           persist=True
       )

System Resource Analysis
^^^^^^^^^^^^^^^^^^^^^^^^

Analyze remote server resources to inform tuning decisions:

.. code-block:: python

   # Get remote server system information
   system_info = remote_db.analyze_system_resources()

   print(f"Server RAM: {system_info['total_ram_mb']} MB")
   print(f"Available RAM: {system_info['available_ram_mb']} MB")
   print(f"CPU cores: {system_info['cpu_cores']}")
   print(f"Disk type: {system_info['disk_type']}")
   print(f"OS: {system_info['os_type']}")

   # Use system info to make informed tuning decisions
   if system_info['disk_type'] == 'SSD' and system_info['total_ram_mb'] > 8192:
       # High-performance configuration for well-equipped servers
       remote_db.set_sqlite_tuning("read_optimized", {
           "cache_size": -524288,  # 512MB cache
           "mmap_size": 2147483648,  # 2GB memory mapping
       })

Remote Maintenance Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Perform database maintenance operations on remote servers:

.. code-block:: python

   # Run maintenance operations remotely
   remote_db.sqlite_checkpoint("TRUNCATE")  # Checkpoint and truncate WAL
   remote_db.sqlite_optimize()  # Update query planner statistics
   remote_db.sqlite_incremental_vacuum(5000)  # Reclaim space

   # Check if WAL checkpoint is needed
   if remote_db.checkpoint_if_wal_large(wal_mb_threshold=256):
       print("Large WAL file was checkpointed")

Multi-Database Remote Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Manage tuning across multiple remote databases:

.. code-block:: python

   databases = [
       ("analytics_db", "read_optimized"),
       ("staging_db", "balanced"),
       ("ingest_db", "fast_ingest")
   ]

   for db_name, profile in databases:
       remote_db = RemoteVectorDB(
           name=db_name,
           base_url="http://server:8000",
           api_key="admin_key"
       )

       # Apply appropriate tuning profile
       remote_db.set_sqlite_tuning(profile)

       # Get current configuration for monitoring
       config = remote_db.get_sqlite_tuning()
       print(f"{db_name}: {config['profile']} profile applied")

Security Considerations
^^^^^^^^^^^^^^^^^^^^^^^

Remote tuning requires appropriate API permissions:

- **API Key Requirements**: Tuning operations require ``read_write`` API keys
- **Administrative Access**: Tuning changes affect server-wide database performance
- **Audit Logging**: All remote tuning operations are logged on the server
- **Rate Limiting**: Tuning endpoints respect server rate limiting policies

.. code-block:: python

   # Use administrative API key for tuning operations
   remote_db = RemoteVectorDB(
       name="production_db",
       base_url="https://db-server.company.com",
       api_key="admin_write_key"  # Must be read_write key
   )

   try:
       remote_db.set_sqlite_tuning("read_optimized")
       print("Tuning applied successfully")
   except PermissionError:
       print("Insufficient API key permissions for tuning operations")

Best Practices for Remote Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Monitor Before Changes**: Always check current performance before applying new tuning
2. **Use Auto-Tuning**: Leverage auto-tuning for initial recommendations based on server hardware
3. **Test in Staging**: Test tuning changes in staging environments before production
4. **Schedule Appropriately**: Apply tuning changes during low-traffic periods
5. **Document Changes**: Keep records of tuning changes and their performance impact
6. **Monitor After Changes**: Track performance metrics after applying tuning changes

.. code-block:: python

   # Example monitoring and tuning workflow
   def optimize_remote_database(remote_db, target_profile):
       # 1. Get baseline performance info
       baseline_config = remote_db.get_sqlite_tuning()
       print(f"Current profile: {baseline_config['profile']}")

       # 2. Get auto-tuning recommendation
       recommendation = remote_db.auto_tune()

       # 3. Apply changes during maintenance window
       if recommendation['profile_name'] != baseline_config['profile']:
           remote_db.set_sqlite_tuning(
               recommendation['profile_name'],
               recommendation['pragma_overrides']
           )
           print(f"Applied new profile: {recommendation['profile_name']}")

       # 4. Perform maintenance if recommended
       if recommendation['estimated_memory_mb'] > 1000:
           remote_db.sqlite_checkpoint("TRUNCATE")
           remote_db.sqlite_optimize()

       return recommendation

API Reference
=============

For complete API documentation, see:

- :doc:`modules/localvectordb.sqlite_tuning` - Core tuning module
- :doc:`modules/localvectordb.database` - Database classes with tuning support
- :doc:`cli` - Command-line interface for tuning management