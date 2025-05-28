Routes API
==========

The LocalVectorDB HTTP API provides comprehensive endpoints for database and document management.

Authentication
--------------

Most endpoints require API key authentication when enabled:

.. code-block:: bash

   # Include API key in Authorization header
   curl -H "Authorization: Bearer your_api_key_here" \
        http://localhost:5000/api/v1/endpoint



Database Management
-------------------

Create Database
^^^^^^^^^^^^^^^

Create a new vector database with optional configuration.

**Endpoint**: ``POST /api/v1/databases``

**Request Body**:

.. code-block:: json

   {
     "name": "my_database",
     "metadata_schema": {
       "title": {"type": "text", "indexed": true},
       "author": {"type": "text", "indexed": true},
       "date": {"type": "date", "indexed": true},
       "tags": {"type": "json"}
     },
     "embedding_provider": "ollama",
     "embedding_model": "nomic-embed-text",
     "chunk_size": 500,
     "chunking_method": "sentences",
     "chunk_overlap": 1,
     "enable_fts": true
   }

**curl Example**:

.. code-block:: bash

   curl -X POST http://localhost:5000/api/v1/databases \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "name": "research_papers",
       "metadata_schema": {
         "title": {"type": "text", "indexed": true},
         "authors": {"type": "json"},
         "journal": {"type": "text", "indexed": true}
       },
       "embedding_model": "nomic-embed-text",
       "chunk_size": 600
     }'

**Python Example**:

.. code-block:: python

   import requests

   response = requests.post(
       "http://localhost:5000/api/v1/databases",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "name": "research_papers",
           "metadata_schema": {
               "title": {"type": "text", "indexed": True},
               "authors": {"type": "json"},
               "journal": {"type": "text", "indexed": True}
           },
           "embedding_model": "nomic-embed-text",
           "chunk_size": 600
       }
   )

   print(response.json())

**Response**:

.. code-block:: json

   {
     "message": "Successfully created database 'research_papers'",
     "status": "success",
     "config": {
       "name": "research_papers",
       "embedding_provider": "ollama",
       "embedding_model": "nomic-embed-text",
       "embedding_dimension": 768,
       "chunking_method": "sentences",
       "chunk_size": 600,
       "chunk_overlap": 1,
       "metadata_schema": {
         "title": {"type": "text", "indexed": true, "required": false},
         "authors": {"type": "json", "indexed": false, "required": false},
         "journal": {"type": "text", "indexed": true, "required": false}
       },
       "fts_enabled": true
     }
   }

List Databases
^^^^^^^^^^^^^^

Get a list of all available databases.

**Endpoint**: ``GET /api/v1/databases``

**curl Example**:

.. code-block:: bash

   curl -H "Authorization: Bearer your_api_key" \
        http://localhost:5000/api/v1/databases

**Python Example**:

.. code-block:: python

   response = requests.get(
       "http://localhost:5000/api/v1/databases",
       headers={"Authorization": "Bearer your_api_key"}
   )

   databases = response.json()["databases"]
   print(f"Available databases: {databases}")

**Response**:

.. code-block:: json

   {
     "databases": ["research_papers", "customer_support", "code_docs"],
     "count": 3
   }

Get Database Info
^^^^^^^^^^^^^^^^^

Retrieve detailed information about a specific database.

**Endpoint**: ``GET /api/v1/{db_name}/info``

**curl Example**:

.. code-block:: bash

   curl -H "Authorization: Bearer your_api_key" \
        http://localhost:5000/api/v1/research_papers/info

**Python Example**:

.. code-block:: python

   response = requests.get(
       "http://localhost:5000/api/v1/research_papers/info",
       headers={"Authorization": "Bearer your_api_key"}
   )

   info = response.json()
   print(f"Documents: {info['stats']['documents']}")
   print(f"Embedding model: {info['config']['embedding_model']}")

**Response**:

.. code-block:: json

   {
     "name": "research_papers",
     "stats": {
       "documents": 1250,
       "chunks": 8500,
       "index_vectors": 8500,
       "embedding_dimension": 768
     },
     "config": {
       "embedding_provider": "ollama",
       "embedding_model": "nomic-embed-text",
       "chunking_method": "sentences",
       "chunk_size": 600,
       "metadata_schema": {
         "title": {"type": "text", "indexed": true},
         "authors": {"type": "json", "indexed": false}
       },
       "fts_enabled": true
     }
   }

Delete Database
^^^^^^^^^^^^^^^

Delete a database and all its data.

**Endpoint**: ``DELETE /api/v1/{db_name}``

**curl Example**:

.. code-block:: bash

   curl -X DELETE \
     -H "Authorization: Bearer your_api_key" \
     http://localhost:5000/api/v1/old_database

**Python Example**:

.. code-block:: python

   response = requests.delete(
       "http://localhost:5000/api/v1/old_database",
       headers={"Authorization": "Bearer your_api_key"}
   )

   print(response.json()["message"])

Document Management
-------------------

Upsert Documents
^^^^^^^^^^^^^^^^

Insert or update documents in the database.

**Endpoint**: ``POST /api/v1/{db_name}/documents``

**Request Body**:

.. code-block:: json

   {
     "documents": ["Document content 1", "Document content 2"],
     "metadata": [
       {"title": "First Doc", "author": "Alice"},
       {"title": "Second Doc", "author": "Bob"}
     ],
     "ids": ["doc_1", "doc_2"],
     "batch_size": 100
   }

**curl Example**:

.. code-block:: bash

   curl -X POST http://localhost:5000/api/v1/research_papers/documents \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "documents": [
         "This paper presents a novel approach to machine learning...",
         "In this study, we investigate the effects of climate change..."
       ],
       "metadata": [
         {
           "title": "Novel ML Approach",
           "authors": ["Dr. Smith", "Dr. Jones"],
           "journal": "AI Research Quarterly"
         },
         {
           "title": "Climate Change Effects",
           "authors": ["Prof. Brown"],
           "journal": "Environmental Science"
         }
       ]
     }'

**Python Example**:

.. code-block:: python

   documents = [
       "This paper presents a novel approach to machine learning...",
       "In this study, we investigate the effects of climate change..."
   ]

   metadata = [
       {
           "title": "Novel ML Approach",
           "authors": ["Dr. Smith", "Dr. Jones"],
           "journal": "AI Research Quarterly"
       },
       {
           "title": "Climate Change Effects",
           "authors": ["Prof. Brown"],
           "journal": "Environmental Science"
       }
   ]

   response = requests.post(
       "http://localhost:5000/api/v1/research_papers/documents",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "documents": documents,
           "metadata": metadata
       }
   )

   doc_ids = response.json()["ids"]
   print(f"Created documents: {doc_ids}")

**Response**:

.. code-block:: json

   {
     "message": "Successfully processed 2 documents",
     "ids": ["doc_1", "doc_2"],
     "status": "success"
   }

Insert Documents
^^^^^^^^^^^^^^^^

Insert new documents (fails if ID already exists).

**Endpoint**: ``POST /api/v1/{db_name}/documents/insert``

**Request Body**:

.. code-block:: json

   {
     "documents": ["New document content"],
     "metadata": [{"category": "new"}],
     "ids": ["unique_id"],
     "errors": "raise",
     "similarity_threshold": 0.95
   }

**curl Example**:

.. code-block:: bash

   curl -X POST http://localhost:5000/api/v1/research_papers/documents/insert \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "documents": ["This is a completely new research paper..."],
       "metadata": [{"title": "Breakthrough Research", "journal": "Science"}],
       "errors": "ignore",
       "similarity_threshold": 0.95
     }'

**Python Example**:

.. code-block:: python

   response = requests.post(
       "http://localhost:5000/api/v1/research_papers/documents/insert",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "documents": ["This is a completely new research paper..."],
           "metadata": [{"title": "Breakthrough Research", "journal": "Science"}],
           "errors": "ignore",  # Don't fail on duplicates
           "similarity_threshold": 0.95  # Skip if 95%+ similar
       }
   )

   print(f"Inserted: {len(response.json()['ids'])} documents")

Get Document
^^^^^^^^^^^^

Retrieve a specific document by ID.

**Endpoint**: ``GET /api/v1/{db_name}/documents/{doc_id}``

**curl Example**:

.. code-block:: bash

   curl -H "Authorization: Bearer your_api_key" \
        http://localhost:5000/api/v1/research_papers/documents/doc_1

**Python Example**:

.. code-block:: python

   response = requests.get(
       "http://localhost:5000/api/v1/research_papers/documents/doc_1",
       headers={"Authorization": "Bearer your_api_key"}
   )

   doc = response.json()
   print(f"Title: {doc['metadata']['title']}")
   print(f"Content: {doc['content'][:200]}...")

**Response**:

.. code-block:: json

   {
     "id": "doc_1",
     "content": "This paper presents a novel approach to machine learning...",
     "metadata": {
       "title": "Novel ML Approach",
       "authors": ["Dr. Smith", "Dr. Jones"],
       "journal": "AI Research Quarterly"
     },
     "created_at": "2024-01-15T10:30:00Z",
     "updated_at": "2024-01-15T10:30:00Z",
     "content_hash": "abc123..."
   }

Update Document
^^^^^^^^^^^^^^^

Update a document's content and/or metadata.

**Endpoint**: ``PUT /api/v1/{db_name}/documents/{doc_id}``

**Request Body**:

.. code-block:: json

   {
     "content": "Updated document content...",
     "metadata": {"status": "revised", "version": 2}
   }

**curl Example**:

.. code-block:: bash

   curl -X PUT http://localhost:5000/api/v1/research_papers/documents/doc_1 \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "content": "This revised paper presents an improved approach...",
       "metadata": {"status": "revised", "version": 2}
     }'

**Python Example**:

.. code-block:: python

   response = requests.put(
       "http://localhost:5000/api/v1/research_papers/documents/doc_1",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "content": "This revised paper presents an improved approach...",
           "metadata": {"status": "revised", "version": 2}
       }
   )

   print(response.json()["message"])

Delete Document
^^^^^^^^^^^^^^^

Delete a document from the database.

**Endpoint**: ``DELETE /api/v1/{db_name}/documents/{doc_id}``

**curl Example**:

.. code-block:: bash

   curl -X DELETE \
     -H "Authorization: Bearer your_api_key" \
     http://localhost:5000/api/v1/research_papers/documents/doc_1

**Python Example**:

.. code-block:: python

   response = requests.delete(
       "http://localhost:5000/api/v1/research_papers/documents/doc_1",
       headers={"Authorization": "Bearer your_api_key"}
   )

   if response.status_code == 200:
       print("Document deleted successfully")

Check Document Existence
^^^^^^^^^^^^^^^^^^^^^^^^

Check if documents exist by their IDs.

**Endpoint**: ``POST /api/v1/{db_name}/documents/exists``

**Request Body**:

.. code-block:: json

   {
     "ids": ["doc_1", "doc_2", "doc_3"]
   }

**curl Example**:

.. code-block:: bash

   curl -X POST http://localhost:5000/api/v1/research_papers/documents/exists \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{"ids": ["doc_1", "doc_2", "nonexistent"]}'

**Python Example**:

.. code-block:: python

   response = requests.post(
       "http://localhost:5000/api/v1/research_papers/documents/exists",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={"ids": ["doc_1", "doc_2", "nonexistent"]}
   )

   exists = response.json()["exists"]  # [true, true, false]

List Documents
^^^^^^^^^^^^^^

List documents with pagination and filtering.

**Endpoint**: ``GET /api/v1/{db_name}/documents``

**Query Parameters**:

- ``page``: Page number (default: 1)
- ``limit``: Items per page (default: 100, max: 1000)
- ``{field_name}``: Filter by metadata field value

**curl Example**:

.. code-block:: bash

   # List first 20 documents
   curl -H "Authorization: Bearer your_api_key" \
        "http://localhost:5000/api/v1/research_papers/documents?limit=20&page=1"

   # Filter by journal
   curl -H "Authorization: Bearer your_api_key" \
        "http://localhost:5000/api/v1/research_papers/documents?journal=Science&limit=10"

**Python Example**:

.. code-block:: python

   # Get second page of documents
   response = requests.get(
       "http://localhost:5000/api/v1/research_papers/documents",
       headers={"Authorization": "Bearer your_api_key"},
       params={"page": 2, "limit": 50}
   )

   docs = response.json()["documents"]
   pagination = response.json()["pagination"]

   print(f"Page {pagination['current_page']} of {pagination['total_pages']}")
   print(f"Total documents: {pagination['total_count']}")

   # Filter documents
   response = requests.get(
       "http://localhost:5000/api/v1/research_papers/documents",
       headers={"Authorization": "Bearer your_api_key"},
       params={"journal": "AI Research Quarterly", "limit": 20}
   )

   filtered_docs = response.json()["documents"]

Search Operations
-----------------

Unified Query Interface
^^^^^^^^^^^^^^^^^^^^^^^

The main search endpoint supporting vector, keyword, and hybrid search.

**Endpoint**: ``POST /api/v1/{db_name}/query``

**Request Body**:

.. code-block:: json

   {
     "query": "machine learning algorithms",
     "search_type": "vector",
     "return_type": "documents",
     "k": 10,
     "score_threshold": 0.7,
     "filters": {"journal": "AI Research Quarterly"},
     "vector_weight": 0.7
   }

**Parameters**:

- ``query``: Search text (required)
- ``search_type``: "vector", "keyword", or "hybrid" (default: "vector")
- ``return_type``: "documents" or "chunks" (default: "documents")
- ``k``: Maximum results to return (default: 10)
- ``score_threshold``: Minimum score (0-1, higher=better)
- ``filters``: Metadata filter conditions
- ``vector_weight``: Weight for vector search in hybrid mode (0-1)

**curl Example**:

.. code-block:: bash

   # Vector search
   curl -X POST http://localhost:5000/api/v1/research_papers/query \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "query": "neural networks deep learning",
       "search_type": "vector",
       "k": 5,
       "score_threshold": 0.8,
       "filters": {"journal": "AI Research Quarterly"}
     }'

   # Hybrid search
   curl -X POST http://localhost:5000/api/v1/research_papers/query \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "query": "climate change effects",
       "search_type": "hybrid",
       "k": 3,
       "vector_weight": 0.6
     }'

**Python Example**:

.. code-block:: python

   # Vector search for research papers
   response = requests.post(
       "http://localhost:5000/api/v1/research_papers/query",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "query": "neural networks deep learning",
           "search_type": "vector",
           "k": 5,
           "score_threshold": 0.8,
           "filters": {"journal": "AI Research Quarterly"}
       }
   )

   results = response.json()["results"]
   for result in results:
       print(f"Score: {result['score']:.3f}")
       print(f"Title: {result['metadata']['title']}")
       print(f"Content: {result['content'][:200]}...")
       print("---")

   # Hybrid search combining vector and keyword
   response = requests.post(
       "http://localhost:5000/api/v1/research_papers/query",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "query": "climate change effects",
           "search_type": "hybrid",
           "k": 3,
           "vector_weight": 0.6  # 60% vector, 40% keyword
       }
   )

   hybrid_results = response.json()["results"]

**Response**:

.. code-block:: json

   {
     "results": [
       {
         "id": "doc_123",
         "score": 0.892,
         "type": "document",
         "content": "Neural networks have revolutionized deep learning...",
         "metadata": {
           "title": "Deep Learning Advances",
           "authors": ["Dr. Smith"],
           "journal": "AI Research Quarterly"
         }
       },
       {
         "id": "doc_456",
         "score": 0.854,
         "type": "document",
         "content": "Recent developments in neural network architectures...",
         "metadata": {
           "title": "Network Architecture Evolution",
           "authors": ["Prof. Johnson"],
           "journal": "AI Research Quarterly"
         }
       }
     ],
     "search_type": "vector",
     "return_type": "documents",
     "total_results": 2
   }

Vector Search (Convenience Endpoint)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Endpoint**: ``POST /api/v1/{db_name}/search/vector``

**curl Example**:

.. code-block:: bash

   curl -X POST http://localhost:5000/api/v1/research_papers/search/vector \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "query": "machine learning optimization",
       "k": 8,
       "return_type": "chunks"
     }'

Keyword Search (Convenience Endpoint)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Endpoint**: ``POST /api/v1/{db_name}/search/keyword``

**curl Example**:

.. code-block:: bash

   curl -X POST http://localhost:5000/api/v1/research_papers/search/keyword \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "query": "\"deep learning\" AND optimization",
       "k": 10
     }'

Hybrid Search (Convenience Endpoint)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Endpoint**: ``POST /api/v1/{db_name}/search/hybrid``

**curl Example**:

.. code-block:: bash

   curl -X POST http://localhost:5000/api/v1/research_papers/search/hybrid \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "query": "neural network training",
       "k": 5,
       "vector_weight": 0.8
     }'

Filtering and Metadata Operations
---------------------------------

Filter Documents
^^^^^^^^^^^^^^^^

Advanced filtering using SQL-like queries.

**Endpoint**: ``POST /api/v1/{db_name}/filter``

**Request Body**:

.. code-block:: json

   {
     "where": {"journal": "Science", "year": {">=": 2020}},
     "order_by": "year DESC",
     "limit": 50,
     "offset": 0
   }

**curl Example**:

.. code-block:: bash

   curl -X POST http://localhost:5000/api/v1/research_papers/filter \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "where": {
         "journal": "Science",
         "year": {">=": 2020}
       },
       "order_by": "year DESC",
       "limit": 20
     }'

**Python Example**:

.. code-block:: python

   # Complex filtering
   response = requests.post(
       "http://localhost:5000/api/v1/research_papers/filter",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "where": {
               "journal": {"in": ["Science", "Nature"]},
               "year": {"between": [2020, 2024]},
               "authors": {"contains": "Smith"}
           },
           "order_by": "year DESC, title ASC",
           "limit": 50
       }
   )

   filtered_docs = response.json()["documents"]

   # Raw SQL filtering
   response = requests.post(
       "http://localhost:5000/api/v1/research_papers/filter",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "sql": "journal = 'Science' AND year >= 2020",
           "order_by": "year DESC",
           "limit": 25
       }
   )

Global Operations
-----------------

Global Search
^^^^^^^^^^^^^

Search across multiple databases simultaneously.

**Endpoint**: ``POST /api/v1/search``

**Request Body**:

.. code-block:: json

   {
     "query": "machine learning",
     "search_type": "vector",
     "k": 5,
     "databases": ["research_papers", "tech_docs", "tutorials"]
   }

**curl Example**:

.. code-block:: bash

   curl -X POST http://localhost:5000/api/v1/search \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "query": "artificial intelligence",
       "search_type": "hybrid",
       "k": 3,
       "databases": ["research_papers", "tech_blogs"]
     }'

**Python Example**:

.. code-block:: python

   # Search across all databases
   response = requests.post(
       "http://localhost:5000/api/v1/search",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "query": "artificial intelligence",
           "search_type": "vector",
           "k": 5
           # databases not specified = search all
       }
   )

   # Results organized by database
   results_by_db = response.json()["results"]
   for db_name, results in results_by_db.items():
       print(f"\nResults from {db_name}:")
       for result in results:
           print(f"  {result['id']}: {result['score']:.3f}")

Embedding Operations
--------------------

Get Embeddings from Database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate embeddings using a database's configured embedding provider.

**Endpoint**: ``POST /api/v1/{db_name}/embeddings``

**Request Body**:

.. code-block:: json

   {
     "texts": ["Text to embed", "Another text"]
   }

**curl Example**:

.. code-block:: bash

   curl -X POST http://localhost:5000/api/v1/research_papers/embeddings \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "texts": ["neural networks", "machine learning algorithms"]
     }'

**Python Example**:

.. code-block:: python

   response = requests.post(
       "http://localhost:5000/api/v1/research_papers/embeddings",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "texts": ["neural networks", "machine learning algorithms"]
       }
   )

   embeddings = response.json()["embeddings"]
   print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")

Get Embeddings from Specific Provider
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate embeddings using a specified provider and model.

**Endpoint**: ``POST /api/v1/embeddings``

**Request Body**:

.. code-block:: json

   {
     "texts": ["Text to embed"],
     "provider": "ollama",
     "model": "nomic-embed-text"
   }

**curl Example**:

.. code-block:: bash

   curl -X POST http://localhost:5000/api/v1/embeddings \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "texts": ["compare different embedding models"],
       "provider": "openai",
       "model": "text-embedding-3-small"
     }'

System Operations
-----------------

Health Check
^^^^^^^^^^^^

Check server and system health.

**Endpoint**: ``GET /api/v1/health``

**curl Example**:

.. code-block:: bash

   curl http://localhost:5000/api/v1/health

**Python Example**:

.. code-block:: python

   response = requests.get("http://localhost:5000/api/v1/health")
   health = response.json()

   print(f"Status: {health['status']}")
   print(f"Databases: {health['databases']}")
   print(f"Ollama available: {health['ollama_available']}")

**Response**:

.. code-block:: json

   {
     "status": "healthy",
     "version": "2.0.0",
     "databases": 3,
     "ollama_available": true
   }

Error Handling
--------------

The API uses standard HTTP status codes and returns structured error responses:

**Error Response Format**:

.. code-block:: json

   {
     "error": "Error description",
     "type": "error_type"
   }

**Common Error Types**:

- ``database_not_found`` (404): Database doesn't exist
- ``duplicate_document_id`` (409): Document ID already exists
- ``embedding_error`` (503): Embedding generation failed
- ``authentication_error`` (401): Invalid or missing API key
- ``database_error`` (500): General database error

**Python Error Handling**:

.. code-block:: python

   try:
       response = requests.post(
           "http://localhost:5000/api/v1/nonexistent/documents",
           headers={"Authorization": "Bearer your_api_key"},
           json={"documents": ["test"]}
       )
       response.raise_for_status()

   except requests.exceptions.HTTPError as e:
       if e.response.status_code == 404:
           error = e.response.json()
           if error.get("type") == "database_not_found":
               print("Database not found - create it first")
       elif e.response.status_code == 401:
           print("Authentication failed - check API key")
       else:
           print(f"API error: {e.response.json()['error']}")
