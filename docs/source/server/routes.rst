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
     "embedding": {
        "provider": "ollama",
        "model": "nomic-embed-text"
     },
     "database": {
        "chunk_size": 500,
        "chunking_method": "sentences",
        "chunk_overlap": 1,
        "enable_fts": true
     }
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
           "embedding": {
               "model": "nomic-embed-text"
           }
           "database": {
               "chunk_size": 600
           }
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

List documents with pagination, filtering, or bulk-get by ID.

**Endpoint**: ``GET /api/v1/{db_name}/documents``

**Query Parameters**:

- ``ids``
  Comma-separated list of document IDs to retrieve.
  If **ids** is provided:

  - Only the listed IDs are returned (in the order given).
  - Pagination (``page``/``limit``) and metadata filters are **ignored**.
  - If any ID does not exist, a ``404`` is returned with a JSON error listing the missing IDs.

- ``page``
  Page number (default: 1). Used **only** when **ids** is *not* provided.

- ``limit``
  Items per page (default: 100, max: 1000). Used **only** when **ids** is *not* provided.

- ``{field_name}``
  Any metadata field may be passed to filter (e.g. ``author=Alice``). Used **only** when **ids** is *not* provided.

**curl Examples**:

.. code-block:: bash

   # 1) Bulk-get three docs by ID
   curl -H "Authorization: Bearer your_api_key" \
        "http://localhost:5000/api/v1/research_papers/documents?ids=doc_1,doc_42,doc_xyz"

   # 2) Paginate through all docs (no ids)
   curl -H "Authorization: Bearer your_api_key" \
        "http://localhost:5000/api/v1/research_papers/documents?page=2&limit=50"

   # 3) Filter by metadata field
   curl -H "Authorization: Bearer your_api_key" \
        "http://localhost:5000/api/v1/research_papers/documents?journal=Science&limit=10"

**Python Examples**:

.. code-block:: python

   import requests

   base = "http://localhost:5000/api/v1/research_papers/documents"
   headers = {"Authorization": "Bearer your_api_key"}

   # Bulk-get by IDs
   resp = requests.get(base, headers=headers, params={"ids": "doc_1,doc_42,doc_xyz"})
   docs = resp.json()["documents"]
   print(f"Retrieved {len(docs)} docs")

   # Paginated listing
   resp = requests.get(base, headers=headers, params={"page": 3, "limit": 25})
   data = resp.json()
   print(f"Page {data['pagination']['current_page']} of {data['pagination']['total_pages']}")

   # Filter by metadata
   resp = requests.get(base, headers=headers, params={"author": "Alice", "limit": 5})
   print(resp.json()["documents"])


File Upload Operations
----------------------

The LocalVectorDB Server supports file uploads with automatic text extraction from various document formats.
This feature allows you to upload documents directly to your vector database without manual text extraction.

.. important:: The file upload routes are only enabled if the ``server.file_upload_enabled`` is set to ``true``.


Supported File Formats
^^^^^^^^^^^^^^^^^^^^^^

**Always Supported (Basic Text)**:

- ``.txt`` - Plain text files
- ``.md`` - Markdown files
- ``.py``, ``.js``, ``.html``, ``.css`` - Code and markup files
- ``.json``, ``.xml``, ``.csv`` - Structured text files

**Optionally Supported (With Additional Dependencies)**:

- ``.pdf`` - PDF documents (via PyPDF2 or pdfplumber)
- ``.docx`` - Microsoft Word documents (via python-docx)
- ``.pptx`` - Microsoft PowerPoint presentations (via python-pptx)
- ``.xlsx``, ``.xls`` - Microsoft Excel spreadsheets (via openpyxl)
- ``.html``, ``.htm`` - HTML web pages (via beautifulsoup4)
- ``.xml``, ``.rss``, ``.atom``, ``.svg`` - XML documents and feeds (via beautifulsoup4)
- ``.rtf`` - Rich Text Format documents (via striprtf)

Installation Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^

**Basic Installation**:

.. code-block:: bash

   pip install localvectordb-server

**With File Extraction Support**:

.. code-block:: bash

   # Standard file extraction capabilities
   pip install localvectordb-server[file-extraction]

   # For only .docx, .pptx, and .xlsx support
   pip install localvectordb-server[file-extraction-office]

   # Manual installation of specific extractors
   pip install PyPDF2 python-docx python-pptx openpyxl striprtf beautifulsoup4

Upload Files to Database
^^^^^^^^^^^^^^^^^^^^^^^^

Upload one or more files to a database with automatic text extraction.

**Endpoint**: ``POST /api/v1/{db_name}/upload``

**Headers**:

- ``Authorization: Bearer {api_key}`` (if authentication enabled)
- ``Content-Type: multipart/form-data``

**Form Data**:

- ``files``: File(s) to upload (required, supports multiple files)
- ``metadata``: JSON string with base metadata to apply to all files (optional)
- ``ids``: JSON array or comma-separated string of document IDs (optional)
- ``use_filename_as_id``: Boolean to use filename as document ID (optional, ignored if ``ids`` provided)
- ``extract_text``: Boolean to enable text extraction (default: true)
- ``batch_size``: Batch size for processing (default: 100, max: 1000)

.. note::
   Only metadata fields that exist in the database's metadata schema will be stored. Extraction metadata and file
   metadata that don't match schema fields will be ignored but reported in the response.

**curl Example**:

.. code-block:: bash

   # Upload multiple files with metadata
   curl -X POST "http://localhost:5000/api/v1/research_papers/upload" \
     -H "Authorization: Bearer lvdb_your_api_key" \
     -F "files=@document.pdf" \
     -F "files=@presentation.pptx" \
     -F "metadata={\"category\":\"research\",\"project\":\"AI\"}" \
     -F "ids=[\"research_doc_1\", \"presentation_slides\"]" \
     -F "extract_text=true" \
     -F "batch_size=50"

   # Upload single file using filename as ID
   curl -X POST "http://localhost:5000/api/v1/my_database/upload" \
     -H "Authorization: Bearer lvdb_your_api_key" \
     -F "files=@important_doc.pdf" \
     -F "use_filename_as_id=true" \
     -F "metadata={\"priority\":\"high\"}"

**Python Example**:

.. code-block:: python

   import requests

   # Upload multiple files
   files = [
       ('files', ('document.pdf', open('document.pdf', 'rb'), 'application/pdf')),
       ('files', ('presentation.pptx', open('presentation.pptx', 'rb'),
                  'application/vnd.openxmlformats-officedocument.presentationml.presentation'))
   ]

   data = {
       'metadata': '{"category":"research","project":"AI"}',
       'ids': '["research_doc_1", "presentation_slides"]',
       'extract_text': 'true',
       'batch_size': '50'
   }

   response = requests.post(
       "http://localhost:5000/api/v1/research_papers/upload",
       headers={"Authorization": "Bearer lvdb_your_api_key"},
       files=files,
       data=data
   )

   result = response.json()
   print(f"Processed {result['files_processed']} files")
   print(f"Document IDs: {result['document_ids']}")

   # Close files
   for _, file_tuple in files:
       file_tuple[1].close()

**Response**:

.. code-block:: json

   {
     "message": "Successfully processed 2 file(s)",
     "files_processed": 2,
     "document_ids": ["research_doc_1", "presentation_slides"],
     "extraction_results": [
       {
         "filename": "document.pdf",
         "extraction_success": true,
         "extraction_method": "pdfplumber",
         "text_length": 1543,
         "error": null,
         "metadata_fields_used": ["extraction_method", "file_size_bytes"],
         "metadata_fields_ignored": ["page_count", "extraction_library"]
       },
       {
         "filename": "presentation.pptx",
         "extraction_success": true,
         "extraction_method": "python-pptx",
         "text_length": 892,
         "error": null,
         "metadata_fields_used": ["extraction_method"],
         "metadata_fields_ignored": ["slide_count", "extraction_library"]
       }
     ],
     "extraction_summary": {
       "total_files": 2,
       "successful_extractions": 2,
       "failed_extractions": 0,
       "supported_formats": {
         "pdf": true,
         "docx": true,
         "pptx": true,
         "xlsx": true,
         "rtf": false
       }
     },
     "status": "success"
   }

Get Supported File Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get information about supported file formats and extraction capabilities.

**Endpoint**: ``GET /api/v1/upload/supported-formats``

**curl Example**:

.. code-block:: bash

   curl -X GET "http://localhost:5000/api/v1/upload/supported-formats" \
     -H "Authorization: Bearer lvdb_your_api_key"

**Python Example**:

.. code-block:: python

   response = requests.get(
       "http://localhost:5000/api/v1/upload/supported-formats",
       headers={"Authorization": "Bearer lvdb_your_api_key"}
   )

   formats = response.json()["supported_formats"]
   print("Supported formats:")
   for format_name, info in formats.items():
       if info["supported"]:
           print(f"  {format_name}: {info['extensions']} - {info['description']}")

**Response**:

.. code-block:: json

   {
     "extraction_available": true,
     "supported_formats": {
       "pdf": {
         "extensions": [".pdf"],
         "mimetypes": ["application/pdf"],
         "description": "PDF files",
         "extractors": ["PDFPlumberExtractor", "PyPDF2Extractor"],
         "supported": true
       },
       "docx": {
         "extensions": [".docx"],
         "mimetypes": ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
         "description": "DOCX files",
         "extractors": ["DocxExtractor"],
         "supported": true
       }
     },
     "basic_text_support": true,
     "text_file_extensions": [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv"],
     "installation_hints": {  // Only returned in development mode
       "pdf": "pip install pdfplumber or pip install PyPDF2",
       "docx": "pip install python-docx",
       "pptx": "pip install python-pptx",
       "xlsx": "pip install openpyxl",
       "rtf": "pip install striprtf"
     }
   }

Preview Text Extraction
^^^^^^^^^^^^^^^^^^^^^^^

Preview text extraction from a file without adding it to the database.

**Endpoint**: ``POST /api/v1/upload/extract-preview``

**Form Data**:

- ``file``: Single file to preview (required)

**curl Example**:

.. code-block:: bash

   curl -X POST "http://localhost:5000/api/v1/my_database/upload/extract-preview" \
     -H "Authorization: Bearer lvdb_your_api_key" \
     -F "file=@sample.pdf"

**Python Example**:

.. code-block:: python

   with open('sample.pdf', 'rb') as f:
       files = {'file': ('sample.pdf', f, 'application/pdf')}

       response = requests.post(
           "http://localhost:5000/api/v1/my_database/upload/extract-preview",
           headers={"Authorization": "Bearer lvdb_your_api_key"},
           files=files
       )

   preview = response.json()
   print(f"File: {preview['filename']}")
   print(f"Extraction method: {preview['extraction_method']}")
   print(f"Text length: {preview['text_length']}")
   print(f"Preview: {preview['text_preview']}")

**Response**:

.. code-block:: json

   {
     "filename": "sample.pdf",
     "original_filename": "sample.pdf",
     "file_size_bytes": 245760,
     "mimetype": "application/pdf",
     "extraction_success": true,
     "extraction_method": "pdfplumber",
     "extraction_metadata": {
       "page_count": 5,
       "extraction_library": "pdfplumber",
       "processing_time_ms": 234
     },
     "extracted_text": "Full extracted text content here...",
     "text_length": 1543,
     "text_preview": "First 500 characters of extracted text..."
   }

Metadata Schema Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file upload API respects the database's metadata schema. Only metadata fields that are defined in the database schema will be stored with the documents. This includes both:

1. **User-provided metadata** (via the ``metadata`` form field)
2. **Extraction metadata** (generated by file extractors)

**Common Extraction Metadata Fields**:

Different extractors generate various metadata fields. To capture this information, consider adding these fields to your database schema:

.. code-block:: python

   from localvectordb.core import MetadataField, MetadataFieldType

   # File upload metadata schema
   upload_schema = {
       # Basic file information
       'source': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       'original_filename': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       'file_size_bytes': MetadataField(type=MetadataFieldType.INTEGER),
       'mimetype': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       'upload_timestamp': MetadataField(type=MetadataFieldType.DATE, indexed=True),

       # Extraction information
       'extraction_method': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       'extraction_success': MetadataField(type=MetadataFieldType.BOOLEAN, indexed=True),
       'text_length': MetadataField(type=MetadataFieldType.INTEGER),

       # Format-specific metadata (add as needed)
       'page_count': MetadataField(type=MetadataFieldType.INTEGER),      # PDF
       'slide_count': MetadataField(type=MetadataFieldType.INTEGER),     # PowerPoint
       'sheet_count': MetadataField(type=MetadataFieldType.INTEGER),     # Excel
       'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True), # HTML, Office docs
       'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True), # Office docs
   }

**Viewing Metadata Usage**:

The upload response includes ``metadata_fields_used`` and ``metadata_fields_ignored`` arrays showing which extraction metadata was stored vs. ignored due to schema constraints.

**Example with Database Creation**:

.. code-block:: python

   # Create database with file upload schema
   response = requests.post(
       "http://localhost:5000/api/v1/databases",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "name": "document_library",
           "metadata_schema": {
               "source": {"type": "text", "indexed": True},
               "original_filename": {"type": "text", "indexed": True},
               "file_size_bytes": {"type": "integer"},
               "mimetype": {"type": "text", "indexed": True},
               "upload_timestamp": {"type": "date", "indexed": True},
               "extraction_method": {"type": "text", "indexed": True},
               "page_count": {"type": "integer"},
               "category": {"type": "text", "indexed": True},
               "tags": {"type": "json"}
           }
       }
   )

   # Now upload files with metadata that matches the schema
   files = [('files', ('document.pdf', open('document.pdf', 'rb'), 'application/pdf'))]
   data = {
       'metadata': '{"category":"research","tags":["AI","ML"]}',
       'extract_text': 'true'
   }

   upload_response = requests.post(
       "http://localhost:5000/api/v1/document_library/upload",
       headers={"Authorization": "Bearer your_api_key"},
       files=files,
       data=data
   )

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
     "where": {"journal": "Science", "year": {"$gte": 2020}},
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
         "year": {"$gte": 2020}
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

Update Metadata Schema
^^^^^^^^^^^^^^^^^^^^^^^

Update the metadata schema for an existing database. This allows you to add new metadata fields, modify existing ones, or remove fields from the schema.

**Endpoint**: ``PUT /api/v1/{db_name}/schema``

**Request Body**:

.. code-block:: json

   {
     "metadata_schema": {
       "category": {"type": "text", "indexed": true, "required": true, "default_value": "general"},
       "priority": {"type": "integer", "default_value": 0},
       "tags": {"type": "json", "default_value": []},
       "rating": {"type": "real", "indexed": true}
     },
     "drop_columns": false,
     "column_mapping": {
         "old_column": "new_column"   // Optionally map the previous metadata columns to new columns
     }
   }

**Parameters**:

- ``metadata_schema``: New schema definition (required)
- ``drop_columns``: Whether to actually drop removed columns (default: false)

**curl Example**:

.. code-block:: bash

   curl -X PUT http://localhost:5000/api/v1/research_papers/schema \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "metadata_schema": {
         "category": {"type": "text", "indexed": true, "required": true, "default_value": "research"},
         "impact_factor": {"type": "real", "indexed": true},
         "keywords": {"type": "json"},
         "peer_reviewed": {"type": "boolean", "default_value": true}
       },
       "drop_columns": false
     }'

**Python Example**:

.. code-block:: python

   new_schema = {
       "category": {"type": "text", "indexed": True, "required": True, "default_value": "research"},
       "impact_factor": {"type": "real", "indexed": True},
       "keywords": {"type": "json"},
       "peer_reviewed": {"type": "boolean", "default_value": True}
   }

   response = requests.put(
       "http://localhost:5000/api/v1/research_papers/schema",
       headers={
           "Content-Type": "application/json",
           "Authorization": "Bearer your_api_key"
       },
       json={
           "metadata_schema": new_schema,
           "drop_columns": False  # Keep old columns for safety
       }
   )

   changes = response.json()["changes"]
   print(f"Added fields: {changes['added_fields']}")
   print(f"Modified fields: {changes['modified_fields']}")
   print(f"Populated defaults: {changes['populated_defaults']}")

**Response**:

.. code-block:: json

   {
     "message": "Successfully updated metadata schema for database 'research_papers'",
     "status": "success",
     "changes": {
       "added_fields": ["impact_factor", "keywords", "peer_reviewed"],
       "removed_fields": [],
       "modified_fields": [
         {
           "field_name": "category",
           "changes": ["added_default_value", "made_required"]
         }
       ],
       "populated_defaults": [
         {
           "field_name": "category",
           "rows_updated": 1250,
           "default_value": "research"
         },
         {
           "field_name": "peer_reviewed",
           "rows_updated": 1250,
           "default_value": true
         }
       ],
       "dropped_columns": [],
       "warnings": [],
       "errors": []
     },
     "new_schema": {
       "category": {"type": "text", "indexed": true, "required": true, "default_value": "research"},
       "impact_factor": {"type": "real", "indexed": true, "required": false, "default_value": null},
       "keywords": {"type": "json", "indexed": false, "required": false, "default_value": null},
       "peer_reviewed": {"type": "boolean", "indexed": false, "required": false, "default_value": true}
     }
   }

Get Metadata Schema Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get detailed information about the current metadata schema for a database.

**Endpoint**: ``GET /api/v1/{db_name}/schema``

**curl Example**:

.. code-block:: bash

   curl -H "Authorization: Bearer your_api_key" \
        http://localhost:5000/api/v1/research_papers/schema

**Python Example**:

.. code-block:: python

   response = requests.get(
       "http://localhost:5000/api/v1/research_papers/schema",
       headers={"Authorization": "Bearer your_api_key"}
   )

   schema_info = response.json()["schema_info"]

   print(f"Total fields: {schema_info['field_count']}")
   print(f"Indexed fields: {schema_info['indexed_fields']}")
   print(f"Required fields: {schema_info['required_fields']}")

   # Show field details
   for field_name, field_info in schema_info['fields'].items():
       print(f"{field_name}: {field_info['type']} "
             f"(indexed={field_info['indexed']}, required={field_info['required']})")

**Response**:

.. code-block:: json

   {
     "database": "research_papers",
     "schema_info": {
       "fields": {
         "title": {
           "type": "text",
           "indexed": true,
           "required": false,
           "default_value": null
         },
         "authors": {
           "type": "json",
           "indexed": false,
           "required": false,
           "default_value": null
         },
         "journal": {
           "type": "text",
           "indexed": true,
           "required": false,
           "default_value": null
         },
         "impact_factor": {
           "type": "real",
           "indexed": true,
           "required": false,
           "default_value": null
         }
       },
       "field_count": 4,
       "indexed_fields": ["title", "journal", "impact_factor"],
       "required_fields": [],
       "field_types": {
         "text": 2,
         "json": 1,
         "real": 1
       }
     },
     "status": "success"
   }

.. note::
   - Field names cannot conflict with reserved columns: ``id``, ``content``, ``content_hash``, ``created_at``, ``updated_at``
   - Removed fields are removed from the schema but columns are kept for data safety unless ``drop_columns=true``
   - Type changes are recorded but don't modify existing data (SQLite limitation)
   - Index changes are applied immediately
   - Changes are applied in a transaction and rolled back on error

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
