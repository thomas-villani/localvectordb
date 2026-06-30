=====================
LocalVectorDB Recipes
=====================

This page contains practical recipes for common LocalVectorDB tasks. Each recipe includes complete,
runnable code examples that you can adapt for your specific use case.

.. contents:: Recipe Index
   :local:
   :depth: 2

Database Setup Recipes
======================

Research Paper Database
-----------------------

Perfect setup for academic papers with rich metadata and optimized chunking for academic content.

.. code-block:: python

   from localvectordb import LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType

   def create_research_db():
       """Create a database optimized for research papers."""
       return LocalVectorDB(
           name="research_papers",
           base_path="./research_db",
           metadata_schema={
               'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=True),
               'authors': MetadataField(type=MetadataFieldType.JSON, indexed=False),
               'journal': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'year': MetadataField(type=MetadataFieldType.INTEGER, indexed=True),
               'doi': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'keywords': MetadataField(type=MetadataFieldType.JSON),
               'citation_count': MetadataField(type=MetadataFieldType.INTEGER),
               'abstract': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
           },
           embedding_provider="ollama",
           embedding_model="nomic-embed-text",
           chunking_method="paragraphs",
           chunk_size=800,
           chunk_overlap=0,
           enable_fts=True
       )

   # Usage
   db = create_research_db()
   doc_ids = db.upsert(
       documents=["This paper presents novel approaches to deep learning..."],
       metadata=[{
           'title': 'Deep Learning Advances in 2024',
           'abstract': 'This paper assesses the novel...',
           'authors': ['Smith, J.', 'Doe, A.'],
           'journal': 'Nature AI',
           'year': 2024,
           'keywords': ['deep learning', 'neural networks', 'transformers'],
           'citation_count': 42
       }]
   )

Code Database
-------------

Optimized for source code files and technical documentation with code-aware chunking.

.. code-block:: python

   def create_code_db():
       """Create a database optimized for code and documentation."""
       return LocalVectorDB(
           name="code_docs",
           base_path="./code_db",
           metadata_schema={
               'filename': MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=True),
               'file_path': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'language': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'project': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'last_modified': MetadataField(type=MetadataFieldType.DATE, indexed=True),
               'function_names': MetadataField(type=MetadataFieldType.JSON),
               'file_size': MetadataField(type=MetadataFieldType.INTEGER),
               'line_count': MetadataField(type=MetadataFieldType.INTEGER)
           },
           embedding_provider="ollama",
           embedding_model="nomic-embed-text",
           chunking_method="code-blocks",
           chunk_size=1000,
           chunk_overlap=5,
           enable_fts=True
       )

Customer Support Database
-------------------------

For support tickets, FAQs, and knowledge base articles with categorization.

.. code-block:: python

   def create_support_db():
       """Create a database for customer support content."""
       return LocalVectorDB(
           name="customer_support",
           base_path="./support_db",
           metadata_schema={
               'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=True),
               'priority': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'status': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'tags': MetadataField(type=MetadataFieldType.JSON),
               'customer_type': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'resolution_time': MetadataField(type=MetadataFieldType.INTEGER),
               'satisfaction_score': MetadataField(type=MetadataFieldType.REAL),
               'created_by': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
           },
           embedding_provider="ollama",
           embedding_model="nomic-embed-text",
           chunking_method="sentences",
           chunk_size=400,
           chunk_overlap=50,
           enable_fts=True  # Critical for keyword search in support
       )

Directory Synchronization
=========================

Sync Code Directory Recursively
-------------------------------

Keep a directory of code files synchronized with your database, using filenames as document IDs for easy updates.

.. code-block:: python

   import os
   from pathlib import Path
   from typing import List, Dict, Set

   def sync_code_directory(db, directory_path: str, file_extensions: List[str] = None):
       """
       Sync a directory of code files with the database.

       Uses relative file paths as document IDs for easy updates.
       Only processes files that have changed since last sync.
       """
       if file_extensions is None:
           file_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.rb', '.go']

       directory = Path(directory_path)
       if not directory.exists():
           raise ValueError(f"Directory does not exist: {directory_path}")

       # Find all code files recursively
       code_files = []
       for ext in file_extensions:
           code_files.extend(directory.rglob(f"*{ext}"))

       print(f"Found {len(code_files)} code files to sync")

       # Get existing documents to check for updates
       existing_docs = db.filter(limit=10000)  # Adjust limit as needed
       existing_files = {doc.metadata.get('file_path'): doc for doc in existing_docs}

       # Process files in batches
       batch_size = 20
       processed_count = 0
       updated_count = 0
       new_count = 0

       for i in range(0, len(code_files), batch_size):
           batch_files = code_files[i:i + batch_size]

           documents = []
           metadata = []
           ids = []

           for file_path in batch_files:
               try:
                   # Read file content
                   with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                       content = f.read()

                   if len(content.strip()) < 10:  # Skip very small files
                       continue

                   # Use relative path as ID
                   relative_path = str(file_path.relative_to(directory))
                   file_stat = file_path.stat()

                   # Check if file needs updating
                   if relative_path in existing_files:
                       existing_doc = existing_files[relative_path]
                       existing_mtime = existing_doc.metadata.get('last_modified')
                       current_mtime = file_stat.st_mtime

                       # Skip if file hasn't changed
                       if existing_mtime and abs(float(existing_mtime) - current_mtime) < 1:
                           continue

                       updated_count += 1
                   else:
                       new_count += 1

                   # Detect programming language
                   language = detect_language(file_path.suffix)

                   # Extract function/class names (simplified)
                   functions = extract_function_names(content, language)

                   documents.append(content)
                   metadata.append({
                       'filename': file_path.name,
                       'file_path': relative_path,
                       'language': language,
                       'project': directory.name,
                       'last_modified': file_stat.st_mtime,
                       'file_size': file_stat.st_size,
                       'line_count': content.count('\n') + 1,
                       'function_names': functions,
                       'file_extension': file_path.suffix
                   })
                   ids.append(relative_path)  # Use relative path as ID

               except Exception as e:
                   print(f"Failed to process {file_path}: {e}")
                   continue

           # Upsert batch (will update existing or create new)
           if documents:
               try:
                   result_ids = db.upsert(
                       documents=documents,
                       metadata=metadata,
                       ids=ids,
                       batch_size=batch_size
                   )
                   processed_count += len(result_ids)
               except Exception as e:
                   print(f"Failed to upsert batch: {e}")

           print(f"Processed {processed_count} files so far...")

       print(f"Sync complete: {new_count} new files, {updated_count} updated files")
       return {'new': new_count, 'updated': updated_count, 'total': processed_count}

   def detect_language(file_extension: str) -> str:
       """Detect programming language from file extension."""
       language_map = {
           '.py': 'python',
           '.js': 'javascript',
           '.ts': 'typescript',
           '.java': 'java',
           '.cpp': 'cpp',
           '.c': 'c',
           '.h': 'c',
           '.cs': 'csharp',
           '.rb': 'ruby',
           '.go': 'go',
           '.rs': 'rust',
           '.php': 'php',
           '.swift': 'swift',
           '.kt': 'kotlin'
       }
       return language_map.get(file_extension.lower(), 'unknown')

   def extract_function_names(content: str, language: str) -> List[str]:
       """Extract function names from code content (simplified)."""
       import re

       patterns = {
           'python': r'def\s+(\w+)\s*\(',
           'javascript': r'function\s+(\w+)\s*\(|(\w+)\s*:\s*function',
           'java': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
           'cpp': r'\w+\s+(\w+)\s*\([^)]*\)\s*{',
           'c': r'\w+\s+(\w+)\s*\([^)]*\)\s*{',
       }

       pattern = patterns.get(language)
       if not pattern:
           return []

       matches = re.findall(pattern, content)
       # Handle tuple results from complex patterns
       functions = []
       for match in matches:
           if isinstance(match, tuple):
               functions.extend([m for m in match if m])
           else:
               functions.append(match)

       return list(set(functions))  # Remove duplicates

   # Usage
   db = create_code_db()

   # Initial sync
   results = sync_code_directory(db, "./my_project")

   # Re-run periodically to sync changes
   # Only changed files will be updated thanks to timestamp checking

Adding Multiple File Types
==========================

Using Built-in Extraction
--------------------------

LocalVectorDB extracts text from files through the :class:`All2MdExtractor`
(backed by `all2md <https://all2md.readthedocs.io/>`_), which covers 20+
document formats (PDF, DOCX, PPTX, XLSX, HTML, EPUB, ODT, email, notebooks, …)
plus many source/text formats. Extracted content is **Markdown**, preserving
headings, tables, and lists. Install the extended-format parsers with
``pip install "localvectordb[file-extraction]"`` (see :doc:`/file-extraction`).

The simplest path is :meth:`~localvectordb.LocalVectorDB.upsert_from_file`,
which runs each file through the extractor, merges any extracted metadata
(``title``, ``author``, ``source_format``, …) with your own, then chunks and
embeds:

.. code-block:: python

   db = create_code_db()  # Or any database

   db.upsert_from_file(
       ["./documents/report.pdf",
        "./docs/presentation.pptx",
        "./data/analysis.xlsx"],
       metadata=[
           {"category": "report", "department": "research"},
           {"category": "presentation", "author": "John Doe"},
           {"category": "data", "project": "quarterly_review"},
       ],
   )

To inspect extraction output before ingesting, call the registry directly. Note
that :meth:`ExtractorRegistry.extract_text` returns an ``ExtractionResult``
(with ``.success``, ``.text``, ``.method``, ``.metadata``) — not a bare string:

.. code-block:: python

   from localvectordb.extractors import ExtractorRegistry

   with open("report.pdf", "rb") as f:
       result = ExtractorRegistry.extract_text(file_content=f.read(), filename="report.pdf")

   if result.success:
       print(result.text[:500])   # Markdown
       print(result.metadata)     # {'title': ..., 'source_format': 'pdf', ...}
   else:
       print("Extraction failed:", result.error)

.. note::

   Because extracted content is Markdown, ``chunking_method="sections"`` is a
   strong choice for ingested documents — it splits on Markdown headings while
   ignoring ``#`` lines inside fenced code blocks. See :doc:`/chunking`.

Using Directory Processing
---------------------------

Process every supported file in a directory by collecting paths and passing them
to :meth:`~localvectordb.LocalVectorDB.upsert_from_file` in one call. Use
:meth:`ExtractorRegistry.get_supported_formats` to discover which extensions the
installed extractors can handle, rather than hard-coding a list:

.. code-block:: python

   from pathlib import Path
   from localvectordb.extractors import ExtractorRegistry

   def process_directory(db, directory_path: str, recursive: bool = True):
       directory = Path(directory_path)
       if not directory.exists():
           raise ValueError(f"Directory does not exist: {directory_path}")

       supported = set(ExtractorRegistry.get_supported_formats())  # {'pdf', 'docx', ...} (no dot)
       paths = directory.rglob("*") if recursive else directory.iterdir()

       files, metas = [], []
       for file_path in paths:
           if not file_path.is_file() or file_path.suffix.lower().lstrip(".") not in supported:
               continue
           if file_path.stat().st_size > 10 * 1024 * 1024:   # skip files > 10 MB
               print(f"Skipping large file: {file_path.name}")
               continue
           files.append(str(file_path))
           metas.append({
               "source_directory": directory.name,
               "relative_path": str(file_path.relative_to(directory)),
           })

       doc_ids = db.upsert_from_file(files, metadata=metas)
       print(f"Ingested {len(doc_ids)} document(s) from {len(files)} file(s)")
       return doc_ids

   # Usage
   doc_ids = process_directory(db, "./mixed_documents", recursive=True)

Multi-Column Search Recipes
============================

Basic Multi-Column Search
-------------------------

Enable embeddings on metadata fields to search across both content and metadata:

.. code-block:: python

   from localvectordb import LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType

   def create_multi_column_db():
       """Create a database with multi-column search capabilities."""
       return LocalVectorDB(
           name="multi_search_db",
           base_path="./multi_db",
           metadata_schema={
               'title': MetadataField(
                   type=MetadataFieldType.TEXT,
                   indexed=True,
                   embedding_enabled=True,  # Enable vector search on title
                   fts_enabled=True  # Also enable FTS
               ),
               'abstract': MetadataField(
                   type=MetadataFieldType.TEXT,
                   embedding_enabled=True  # Enable embeddings without indexing
               ),
               'summary': MetadataField(
                   type=MetadataFieldType.TEXT,
                   embedding_enabled=True
               ),
               'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'tags': MetadataField(type=MetadataFieldType.JSON, embedding_enabled=True)
           },
           embedding_provider="ollama",
           embedding_model="nomic-embed-text"
       )

   # Usage example
   db = create_multi_column_db()
   
   # Add documents with rich metadata
   db.upsert(
       documents=["Full document content about machine learning..."],
       metadata=[{
           'title': 'Introduction to Deep Learning',
           'abstract': 'This paper explores neural networks and deep learning architectures',
           'summary': 'A comprehensive guide to modern AI techniques',
           'category': 'AI',
           'author': 'Dr. Smith',
           'tags': ['machine learning', 'neural networks', 'AI']
       }]
   )
   
   # Search across all embedding-enabled fields
   results = db.query_multi_column("neural networks", k=5)
   for result in results:
       column = result.metadata.get('_search_column', 'unknown')
       print(f"Found in {column}: {result.content[:100]}... (Score: {result.score:.3f})")

Advanced Multi-Column Queries
-----------------------------

Control which columns to search and combine with filters:

.. code-block:: python

   # Search only specific columns
   title_abstract_results = db.query_multi_column(
       "machine learning",
       columns=['title', 'abstract'],  # Only search these fields
       search_type='vector',
       k=10,
       score_threshold=0.7
   )
   
   # Multi-column search with metadata filtering
   filtered_results = db.query_multi_column(
       "deep learning",
       columns=['content', 'title', 'summary'],
       filters={'category': 'AI', 'author': {'$ne': 'Anonymous'}},
       k=5
   )
   
   # Hybrid multi-column search
   hybrid_results = db.query_multi_column(
       "transformer architecture",
       search_type='hybrid',
       vector_weight=0.8,
       k=10
   )

Scientific Paper Search
-----------------------

Complete example for searching academic papers across multiple fields:

.. code-block:: python

   def setup_paper_search():
       """Setup comprehensive paper search database."""
       db = LocalVectorDB(
           name="papers",
           base_path="./paper_db",
           metadata_schema={
               'title': MetadataField(
                   type=MetadataFieldType.TEXT,
                   indexed=True,
                   required=True,
                   embedding_enabled=True,
                   fts_enabled=True
               ),
               'abstract': MetadataField(
                   type=MetadataFieldType.TEXT,
                   embedding_enabled=True
               ),
               'introduction': MetadataField(
                   type=MetadataFieldType.TEXT,
                   embedding_enabled=True
               ),
               'conclusion': MetadataField(
                   type=MetadataFieldType.TEXT,
                   embedding_enabled=True
               ),
               'authors': MetadataField(type=MetadataFieldType.JSON),
               'year': MetadataField(type=MetadataFieldType.INTEGER, indexed=True),
               'journal': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'citations': MetadataField(type=MetadataFieldType.INTEGER, indexed=True),
               'keywords': MetadataField(
                   type=MetadataFieldType.JSON,
                   embedding_enabled=True  # JSON fields can have embeddings too
               )
           },
           embedding_model="nomic-embed-text",
           chunk_size=1000
       )
       return db
   
   def search_papers(db, query: str, min_citations: int = 10):
       """
       Search papers across multiple sections with citation filtering.
       """
       results = db.query_multi_column(
           query,
           columns=['content', 'title', 'abstract', 'introduction', 'conclusion'],
           filters={'citations': {'$gte': min_citations}},
           search_type='hybrid',
           k=20,
           document_scoring_method='frequency_boost'
       )
       
       # Group results by column for analysis
       by_column = {}
       for result in results:
           column = result.metadata.get('_search_column', 'unknown')
           if column not in by_column:
               by_column[column] = []
           by_column[column].append(result)
       
       # Show distribution
       print(f"Results distribution for query '{query}':")
       for column, items in by_column.items():
           avg_score = sum(r.score for r in items) / len(items)
           print(f"  {column}: {len(items)} results (avg score: {avg_score:.3f})")
       
       return results
   
   # Usage
   db = setup_paper_search()
   results = search_papers(db, "transformer attention mechanisms", min_citations=50)

Column Attribution Analysis
---------------------------

Analyze which columns are most relevant for different queries:

.. code-block:: python

   def analyze_column_relevance(db, queries: list, columns: list = None):
       """
       Analyze which columns are most relevant for different query types.
       """
       results_analysis = {}
       
       for query in queries:
           results = db.query_multi_column(
               query,
               columns=columns,
               k=20
           )
           
           # Analyze column distribution
           column_scores = {}
           for result in results:
               col = result.metadata.get('_search_column', 'content')
               if col not in column_scores:
                   column_scores[col] = []
               column_scores[col].append(result.score)
           
           # Calculate statistics
           query_stats = {}
           for col, scores in column_scores.items():
               query_stats[col] = {
                   'count': len(scores),
                   'avg_score': sum(scores) / len(scores),
                   'max_score': max(scores),
                   'min_score': min(scores)
               }
           
           results_analysis[query] = query_stats
       
       return results_analysis
   
   # Example usage
   queries = [
       "machine learning algorithms",
       "experimental results",
       "future work",
       "related research"
   ]
   
   analysis = analyze_column_relevance(db, queries)
   
   for query, stats in analysis.items():
       print(f"\nQuery: '{query}'")
       for col, metrics in stats.items():
           print(f"  {col}: {metrics['count']} hits, "
                 f"avg score {metrics['avg_score']:.3f}")

Server Setup and JavaScript Usage
==================================

Basic Server Setup
------------------

To set up a LocalVectorDB server for remote access and file uploads, you can use the :doc:`command-line interface <cli>`:

.. code-block:: console

   # We specify flags to enable auth with API keys, CORS, and file upload
   $ lvdb config init \
       --enable-auth \
       --enable-cors --cors-origins http://localhost:5000 \
       --enable-file-upload

   # Now we create an API key
   $ lvdb auth create-key

   ✓ API Key Created Successfully

   Key Details:
     Key ID: key_20250616_9zb9mf
     Description: None
     Created: 2025-06-16 15:40:51 UTC
     Expires: Never

   API Key (save this now - it won't be shown again):
     lvdb_h5UBMnAyAkVMYtAgqLr3Cv5VWbF497dI

   ⚠️  Store this key securely - it cannot be retrieved again!

Copy this API key somewhere, you will need it below!

Now start the server:

.. code-block:: bash

   lvdb serve


Simple as that! While the server is running, you can connect to it via the REST API.

JavaScript Client Usage
-----------------------

Complete JavaScript examples for interacting with the LocalVectorDB server.

.. code-block:: javascript

   // vectordb-client.js
   class LocalVectorDBClient {
       constructor(baseUrl, apiKey) {
           this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
           this.apiKey = apiKey;
       }

       async makeRequest(endpoint, options = {}) {
           const url = `${this.baseUrl}${endpoint}`;
           const headers = {
               'Authorization': `Bearer ${this.apiKey}`,
               ...options.headers
           };

           const response = await fetch(url, {
               ...options,
               headers
           });

           if (!response.ok) {
               const error = await response.json().catch(() => ({ error: 'Unknown error' }));
               throw new Error(`API Error: ${response.status} - ${error.message || error.error}`);
           }

           return response.json();
       }

       // Create a new database
       async createDatabase(name, config = {}) {
           return this.makeRequest(`/api/v1/databases`, {
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify({
                   name: name,
                   metadata_schema: config.metadata_schema || {},
                   embedding: {
                       provider: config.embedding_provider || 'ollama',
                       model: config.embedding_model || 'nomic-embed-text'
                   },
                   database: {
                       chunking_method: config.chunking_method || 'sentences',
                       chunk_size: config.chunk_size || 500,
                       enable_fts: config.enable_fts !== false
                   }
               })
           });
       }

       // Add documents to database
       async addDocuments(dbName, documents, metadata = null) {
           const payload = { documents };
           if (metadata) payload.metadata = metadata;

           return this.makeRequest(`/api/v1/${dbName}/documents`, {
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify(payload)
           });
       }

       // Search documents
       async search(dbName, query, options = {}) {
           const payload = {
               query: query,
               search_type: options.searchType || 'hybrid',
               return_type: options.returnType || 'documents',
               k: options.limit || 10,
               score_threshold: options.scoreThreshold || 0.0
           };

           // Add optional parameters
           if (options.filters) {
               payload.filters = options.filters;
           }
           if (options.vectorWeight !== undefined) {
               payload.vector_weight = options.vectorWeight;
           }
           if (options.contextWindow !== undefined) {
               payload.context_window = options.contextWindow;
           }
           if (options.semanticDedupThreshold !== undefined) {
               payload.semantic_dedup_threshold = options.semanticDedupThreshold;
           }
           if (options.documentScoringMethod) {
               payload.document_scoring_method = options.documentScoringMethod;
           }

           return this.makeRequest(`/api/v1/${dbName}/query`, {
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify(payload)
           });
       }

       // Upload files with extraction
       async uploadFiles(dbName, files, options = {}) {
           const formData = new FormData();

           // Add files
           for (let i = 0; i < files.length; i++) {
               formData.append('files', files[i]);
           }

           // Add options
           if (options.metadata) {
               formData.append('metadata', JSON.stringify(options.metadata));
           }
           if (options.useFilenameAsId) {
               formData.append('use_filename_as_id', 'true');
           }
           if (options.batchSize) {
               formData.append('batch_size', options.batchSize.toString());
           }

           return this.makeRequest(`/api/v1/${dbName}/upload`, {
               method: 'POST',
               body: formData
               // Don't set Content-Type header - let browser set it with boundary
           });
       }

       // Get supported file formats
       async getSupportedFormats() {
           return this.makeRequest('/api/v1/upload/supported-formats');
       }

       // Get database info (stats, schema, configuration)
       async getInfo(dbName) {
           return this.makeRequest(`/api/v1/${dbName}/info`);
       }

       // Get document by ID
       async getDocument(dbName, docId) {
           return this.makeRequest(`/api/v1/${dbName}/documents/${docId}`);
       }
   }

HTML Integration Example
------------------------

Complete HTML page showing how to integrate with the LocalVectorDB server.

.. code-block:: html

   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>LocalVectorDB Web Interface</title>
       <style>
           body { font-family: Arial, sans-serif; margin: 20px; }
           .container { max-width: 800px; margin: 0 auto; }
           .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
           .result { background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 3px; }
           .error { background: #ffe6e6; color: #cc0000; }
           .success { background: #e6ffe6; color: #006600; }
           button { padding: 8px 16px; margin: 5px; }
           input, textarea, select { width: 100%; padding: 5px; margin: 5px 0; box-sizing: border-box; }
       </style>
   </head>
   <body>
       <div class="container">
           <h1>LocalVectorDB Web Interface</h1>

           <div class="section">
               <h2>Configuration</h2>
               <input type="text" id="serverUrl" placeholder="Server URL (e.g., http://localhost:5000)" value="http://localhost:5000">
               <input type="text" id="apiKey" placeholder="API Key">
               <input type="text" id="dbName" placeholder="Database Name" value="my_docs">
               <button onclick="initializeClient()">Connect</button>
           </div>

           <div class="section">
               <h2>Add Documents</h2>
               <textarea id="documentText" placeholder="Enter document text..." rows="4"></textarea>
               <input type="text" id="documentCategory" placeholder="Category">
               <input type="text" id="documentTags" placeholder="Tags (comma-separated)">
               <button onclick="addDocument()">Add Document</button>
           </div>

           <div class="section">
               <h2>Upload Files</h2>
               <input type="file" id="fileInput" multiple accept=".txt,.pdf,.docx,.md,.py,.js">
               <label>
                   <input type="checkbox" id="useFilenameAsId"> Use filename as document ID
               </label>
               <button onclick="uploadFiles()">Upload Files</button>
           </div>

           <div class="section">
               <h2>Search</h2>
               <input type="text" id="searchQuery" placeholder="Enter search query...">
               <select id="searchType">
                   <option value="hybrid">Hybrid</option>
                   <option value="vector">Vector</option>
                   <option value="keyword">Keyword</option>
               </select>
               <input type="number" id="searchLimit" placeholder="Limit" value="5" min="1" max="50">
               <button onclick="searchDocuments()">Search</button>
           </div>

           <div class="section">
               <h2>Results</h2>
               <div id="results"></div>
           </div>
       </div>

       <script>
           let client = null;

       function initializeClient() {
           const serverUrl = document.getElementById('serverUrl').value;
           const apiKey = document.getElementById('apiKey').value;

           if (!serverUrl || !apiKey) {
               showResult('Please enter server URL and API key', 'error');
               return;
           }

           client = new LocalVectorDBClient(serverUrl, apiKey);
           showResult('Client initialized successfully', 'success');
       }

       async function addDocument() {
           if (!client) {
               showResult('Please initialize client first', 'error');
               return;
           }

           const text = document.getElementById('documentText').value;
           const category = document.getElementById('documentCategory').value;
           const tags = document.getElementById('documentTags').value.split(',').map(t => t.trim()).filter(t => t);
           const dbName = document.getElementById('dbName').value;

           if (!text || !dbName) {
               showResult('Please enter document text and database name', 'error');
               return;
           }

           try {
               const metadata = {};
               if (category) metadata.category = category;
               if (tags.length > 0) metadata.tags = tags;

               const result = await client.addDocuments(dbName, [text], [metadata]);
               showResult(`Document added successfully: ${result.ids[0]}`, 'success');

               // Clear form
               document.getElementById('documentText').value = '';
               document.getElementById('documentCategory').value = '';
               document.getElementById('documentTags').value = '';
           } catch (error) {
               showResult(`Error adding document: ${error.message}`, 'error');
           }
       }

       async function uploadFiles() {
           if (!client) {
               showResult('Please initialize client first', 'error');
               return;
           }

           const fileInput = document.getElementById('fileInput');
           const useFilenameAsId = document.getElementById('useFilenameAsId').checked;
           const dbName = document.getElementById('dbName').value;

           if (!fileInput.files.length || !dbName) {
               showResult('Please select files and enter database name', 'error');
               return;
           }

           try {
               const result = await client.uploadFiles(dbName, fileInput.files, {
                   useFilenameAsId,
                   metadata: { source: 'web_upload', upload_time: new Date().toISOString() }
               });
               showResult(`Files uploaded successfully: ${result.files_processed} files processed`, 'success');
               fileInput.value = ''; // Clear file input
           } catch (error) {
               showResult(`Error uploading files: ${error.message}`, 'error');
           }
       }

       async function searchDocuments() {
           if (!client) {
               showResult('Please initialize client first', 'error');
               return;
           }

           const query = document.getElementById('searchQuery').value;
           const searchType = document.getElementById('searchType').value;
           const limit = parseInt(document.getElementById('searchLimit').value);
           const dbName = document.getElementById('dbName').value;

           if (!query || !dbName) {
               showResult('Please enter search query and database name', 'error');
               return;
           }

           try {
               const result = await client.search(dbName, query, {
                   searchType,
                   limit
               });

               let html = `<h3>Search Results (${result.results.length} found)</h3>`;
               result.results.forEach((doc, index) => {
                   html += `
                       <div class="result">
                           <strong>Result ${index + 1}</strong> (Score: ${doc.score.toFixed(3)})<br>
                           <strong>ID:</strong> ${doc.id}<br>
                           <strong>Content:</strong> ${doc.content.substring(0, 200)}${doc.content.length > 200 ? '...' : ''}<br>
                           <strong>Metadata:</strong> ${JSON.stringify(doc.metadata)}
                       </div>
                   `;
               });

               document.getElementById('results').innerHTML = html;
           } catch (error) {
               showResult(`Error searching: ${error.message}`, 'error');
           }
       }

       function showResult(message, type) {
           const resultsDiv = document.getElementById('results');
           resultsDiv.innerHTML = `<div class="result ${type}">${message}</div>`;
       }

           // Include the LocalVectorDBClient class here
           // (Copy the class definition from the previous JavaScript example)
       </script>
   </body>
   </html>

.. note::

   Remember to:

   - Start the LocalVectorDB server with ``lvdb serve`` or use the setup script above
   - Create an API key using the CLI: ``lvdb auth create-key --description "Web Interface"``
   - Enable CORS if accessing from a different domain
   - Check supported file formats with ``GET /api/v1/upload/supported-formats``

Document Comparison Recipes
===========================

Find Near-Duplicate Documents
-----------------------------

Scan the database for document pairs with very high similarity.

.. code-block:: python

   def find_duplicates(db, threshold=0.95):
       """Find near-duplicate documents in the database."""
       matrix = db.pairwise_similarity_matrix()
       duplicates = []
       for i in range(len(matrix.doc_ids)):
           for j in range(i + 1, len(matrix.doc_ids)):
               if matrix.matrix[i, j] >= threshold:
                   duplicates.append((
                       matrix.doc_ids[i],
                       matrix.doc_ids[j],
                       float(matrix.matrix[i, j]),
                   ))
       return sorted(duplicates, key=lambda x: x[2], reverse=True)

   # Usage
   for doc_a, doc_b, score in find_duplicates(db):
       print(f"  {doc_a} <-> {doc_b}: {score:.3f}")

Cluster Documents by Topic
---------------------------

Automatically group documents into topic clusters and print the groups.

.. code-block:: python

   from localvectordb.visualization import cluster_embeddings, find_optimal_clusters

   def cluster_documents(db):
       """Cluster all documents and return groups."""
       embeddings, doc_ids = db._get_document_embeddings_batch(None)
       k = find_optimal_clusters(embeddings)
       clusters = cluster_embeddings(embeddings, n_clusters=k)

       groups = {}
       for i, label in enumerate(clusters.labels):
           groups.setdefault(int(label), []).append(doc_ids[i])
       return groups

   # Usage
   for cluster_id, members in cluster_documents(db).items():
       print(f"Cluster {cluster_id}: {members}")

Compare Document Versions
--------------------------

Detect what changed between two versions of a document using chunk-level comparison.

.. code-block:: python

   def diff_documents(db, old_id, new_id, threshold=0.6):
       """Show content differences between two document versions."""
       result = db.compare_documents_detailed(old_id, new_id, chunk_threshold=threshold)

       print(f"Overall similarity: {result.overall_similarity:.3f}")
       print(f"Matched: {result.matched_ratio_1:.0%} of old, {result.matched_ratio_2:.0%} of new")

       if result.unmatched_chunks_2:
           new_doc = db.get(new_id)
           print("\nNew content in updated version:")
           for chunk in new_doc.chunks or []:
               if chunk.index in result.unmatched_chunks_2:
                   print(f"  + {chunk.content[:120]}...")

       if result.unmatched_chunks_1:
           old_doc = db.get(old_id)
           print("\nRemoved from old version:")
           for chunk in old_doc.chunks or []:
               if chunk.index in result.unmatched_chunks_1:
                   print(f"  - {chunk.content[:120]}...")

Visualization Recipes
=====================

Generate a Similarity Heatmap
------------------------------

Create a heatmap of pairwise document similarities, useful for spotting clusters and outliers.

.. code-block:: python

   from localvectordb.visualization import plot_similarity_matrix

   matrix = db.pairwise_similarity_matrix()
   fig = plot_similarity_matrix(matrix, title="Document Similarity Heatmap")
   fig.savefig("similarity_heatmap.png", dpi=150, bbox_inches="tight")

Build a Document Similarity Network
-------------------------------------

Visualise documents as a network graph where edges connect similar documents.

.. code-block:: python

   from localvectordb.visualization import plot_similarity_graph

   matrix = db.pairwise_similarity_matrix()
   fig = plot_similarity_graph(
       matrix,
       threshold=0.5,      # only show edges with similarity >= 0.5
       title="Document Network",
   )
   fig.savefig("doc_network.png", dpi=150, bbox_inches="tight")

Visualise Query Relevance
--------------------------

Overlay multiple queries on the document embedding map. Documents more relevant to the
queries appear as larger dots.

.. code-block:: python

   fig = db.visualize_queries(
       queries=[
           "machine learning algorithms",
           "database performance tuning",
       ],
       method="pca",
   )
   fig.savefig("query_relevance.png", dpi=150, bbox_inches="tight")