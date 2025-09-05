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

Using Built-in Extractors
--------------------------

The LocalVectorDB server includes built-in extractors for many file types. Here's how to add various file formats to your database.

.. code-block:: python

   import mimetypes
   from pathlib import Path
   from typing import Optional, Dict, Any
   from localvectordb_server.extractors import ExtractorRegistry

   def add_file_with_extraction(db, file_path: str, metadata: Optional[Dict[str, Any]] = None):
       """
       Add a file to the database with automatic content extraction.

       Supports: PDF, DOCX, PPTX, XLSX, RTF, and all text-based files.
       For advanced extraction, use the server's upload endpoints.
       """
       path = Path(file_path)
       if not path.exists():
           raise FileNotFoundError(f"File not found: {file_path}")

       # Detect file type
       mime_type, _ = mimetypes.guess_type(str(path))

       try:
           with open(path, "rb") as file_obj:
               content_bytes = file_obj.read()
               content = ExtractorRegistry.extract_text(content_bytes, path, mime_type)

           if not content or len(content.strip()) < 10:
               print(f"Warning: No meaningful content extracted from {file_path}")
               return None

           # Prepare metadata
           file_metadata = {
               'filename': path.name,
               'file_path': str(path),
               'file_extension': path.suffix,
               'file_size': path.stat().st_size,
               'mime_type': mime_type or 'unknown',
               'character_count': len(content)
           }

           if metadata:
               file_metadata.update(metadata)

           # Insert into database
           doc_ids = db.upsert(
               documents=[content],
               metadata=[file_metadata],
               ids=[str(path)]  # Use file path as ID
           )

           print(f"Successfully added: {path.name} ({len(content)} characters)")
           return doc_ids[0] if doc_ids else None

       except Exception as e:
           print(f"Failed to process {file_path}: {e}")
           return None


   # Usage examples
   db = create_code_db()  # Or any database

   # Add different file types
   add_file_with_extraction(db, "./documents/report.pdf",
                           metadata={'category': 'report', 'department': 'research'})

   add_file_with_extraction(db, "./docs/presentation.pptx",
                           metadata={'category': 'presentation', 'author': 'John Doe'})

   add_file_with_extraction(db, "./data/analysis.xlsx",
                           metadata={'category': 'data', 'project': 'quarterly_review'})

.. note::

   **Server Extractors**: The LocalVectorDB server includes a comprehensive extraction system
   that supports many more file types and extraction methods. When using the server, you can
   upload files directly via the API and let the server handle extraction automatically.

   Use ``GET /api/v1/upload/supported-formats`` to see all supported formats in your server installation.

Using Directory Processing
---------------------------

Process multiple file types from a directory automatically.

.. code-block:: python

   def process_mixed_directory(db, directory_path: str, recursive: bool = True):
       """
       Process a directory containing mixed file types.

       Automatically detects and extracts content from supported file types.
       """
       directory = Path(directory_path)
       if not directory.exists():
           raise ValueError(f"Directory does not exist: {directory_path}")

       # Find all files
       if recursive:
           all_files = [f for f in directory.rglob("*") if f.is_file()]
       else:
           all_files = [f for f in directory.iterdir() if f.is_file()]

       results = {'success': [], 'failed': [], 'skipped': []}

       # Supported extensions for content extraction
       supported_extensions = {
           '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv',
           '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.rtf'
       }

       for file_path in all_files:
           try:
               # Skip unsupported file types
               if file_path.suffix.lower() not in supported_extensions:
                   results['skipped'].append(str(file_path))
                   continue

               # Skip very large files (>10MB)
               if file_path.stat().st_size > 10 * 1024 * 1024:
                   print(f"Skipping large file: {file_path.name}")
                   results['skipped'].append(str(file_path))
                   continue

               # Process file
               relative_path = file_path.relative_to(directory)
               doc_id = add_file_with_extraction(
                   db,
                   str(file_path),
                   metadata={
                       'source_directory': directory.name,
                       'relative_path': str(relative_path),
                       'processed_date': Path(file_path).stat().st_mtime
                   }
               )

               if doc_id:
                   results['success'].append(str(file_path))
               else:
                   results['failed'].append(str(file_path))

           except Exception as e:
               print(f"Error processing {file_path}: {e}")
               results['failed'].append(str(file_path))

       print(f"Processing complete:")
       print(f"  Successful: {len(results['success'])}")
       print(f"  Failed: {len(results['failed'])}")
       print(f"  Skipped: {len(results['skipped'])}")

       return results

   # Usage
   results = process_mixed_directory(db, "./mixed_documents", recursive=True)

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
           return this.makeRequest(`/api/v1/${name}`, {
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify({
                   embedding_provider: config.embedding_provider || 'ollama',
                   embedding_model: config.embedding_model || 'nomic-embed-text',
                   metadata_schema: config.metadata_schema || {},
                   chunking_method: config.chunking_method || 'sentences',
                   chunk_size: config.chunk_size || 500,
                   enable_fts: config.enable_fts !== false
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

       // Get database statistics
       async getStats(dbName) {
           return this.makeRequest(`/api/v1/${dbName}/stats`);
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

   - Start the LocalVectorDB server with ``python -m localvectordb_server`` or use the setup script above
   - Create an API key using the CLI: ``lvdb auth create-key --description "Web Interface"``
   - Enable CORS if accessing from a different domain
   - Check supported file formats with ``GET /api/v1/upload/supported-formats``