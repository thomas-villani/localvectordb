.. _file-extraction:

=============================
File Extraction System
=============================

LocalVectorDB includes a sophisticated, plugin-based file extraction system that automatically extracts text content from various document formats. This system allows you to upload files directly to your vector database without manual text extraction.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The file extraction system is built around a **plugin architecture** that uses Python entry points for extensibility. Each extractor is designed to handle specific file formats and gracefully degrades if optional dependencies are not available.

Key Features
^^^^^^^^^^^^

- **Plugin-Based Architecture**: Extensible system using Python entry points
- **Graceful Degradation**: Missing dependencies don't break the system
- **Priority-Based Selection**: Multiple extractors per format with priority ordering
- **Rich Metadata Extraction**: Extracts document metadata alongside text content
- **Optional Dependencies**: Only install what you need
- **Automatic Format Detection**: Uses file extensions and MIME types

Architecture Components
^^^^^^^^^^^^^^^^^^^^^^^

**ExtractorRegistry**
  Central registry that discovers and manages all available extractors

**BaseExtractor**
  Abstract base class that all extractors inherit from

**ExtractionResult**
  Standardized result object containing extracted text, metadata, and status

**Entry Points**
  Python packaging mechanism for plugin discovery (defined in ``pyproject.toml``)

Supported File Formats
-----------------------

The following formats are supported through various optional dependency groups:

Office Documents
^^^^^^^^^^^^^^^^

**Extensions**: ``.docx``, ``.pptx``, ``.xlsx``

**Dependencies**: Install with ``pip install localvectordb[file-extraction-office]``

.. list-table:: Office Format Support
   :header-rows: 1
   :widths: 15 25 40 20

   * - Format
     - Extensions
     - Extracted Content
     - Required Package
   * - Word Documents
     - ``.docx``
     - Text, tables, headers/footers
     - ``python-docx``
   * - PowerPoint
     - ``.pptx``
     - Slide text, notes, metadata
     - ``python-pptx``
   * - Excel Spreadsheets
     - ``.xlsx``
     - Cell values, sheet names
     - ``openpyxl``

**Example**:

.. code-block:: python

   # Upload a Word document
   with open("report.docx", "rb") as f:
       db.upload_file(
           file_content=f.read(),
           filename="report.docx",
           metadata={"category": "reports", "department": "marketing"}
       )

PDF Documents
^^^^^^^^^^^^^

**Extensions**: ``.pdf``

**Dependencies**: Install with ``pip install localvectordb[file-extraction-pdf]``

.. list-table:: PDF Extractor Comparison
   :header-rows: 1
   :widths: 20 40 20 20

   * - Extractor
     - Strengths
     - Priority
     - Package
   * - PDFPlumber
     - Better table extraction, layout preservation
     - High (20)
     - ``pdfplumber``
   * - PyPDF
     - Faster, smaller memory footprint
     - Medium (15)
     - ``pypdf``

**Example**:

.. code-block:: python

   # Upload PDF with automatic extractor selection
   with open("research_paper.pdf", "rb") as f:
       result = db.upload_file(
           file_content=f.read(),
           filename="research_paper.pdf"
       )

   # Check which extractor was used
   print(f"Extraction method: {result['extraction_method']}")

Web Formats
^^^^^^^^^^^

**Extensions**: ``.html``, ``.htm``, ``.xml``

**Dependencies**: Install with ``pip install localvectordb[file-extraction-web]``

.. list-table:: Web Format Support
   :header-rows: 1
   :widths: 15 25 40 20

   * - Format
     - Extensions
     - Extracted Content
     - Required Package
   * - HTML
     - ``.html``, ``.htm``
     - Text content, title, meta tags
     - ``beautifulsoup4``, ``lxml``
   * - XML
     - ``.xml``
     - Text content, structured data
     - ``beautifulsoup4``, ``lxml``

**Example**:

.. code-block:: python

   # Upload HTML file
   with open("webpage.html", "rb") as f:
       db.upload_file(
           file_content=f.read(),
           filename="webpage.html",
           metadata={"source": "web_scraping", "url": "https://example.com"}
       )

E-Books and Rich Text
^^^^^^^^^^^^^^^^^^^^^

**Extensions**: ``.epub``, ``.rtf``

**Dependencies**: Install with ``pip install localvectordb[file-extraction-rtf-epub]``

.. list-table:: E-Book Format Support
   :header-rows: 1
   :widths: 15 25 40 20

   * - Format
     - Extensions
     - Extracted Content
     - Required Package
   * - EPUB
     - ``.epub``
     - Chapter text, metadata, TOC
     - ``ebooklib``
   * - Rich Text Format
     - ``.rtf``
     - Formatted text content
     - ``striprtf``

Text Files
^^^^^^^^^^

**Extensions**: ``.txt``, ``.md``, ``.py``, ``.js``, ``.css``, and many others

**Dependencies**: None (built-in)

The text extractor handles any file that appears to contain readable text and is always available as a fallback.

Installation Options
--------------------

Choose the installation option that matches your needs:

Complete Installation
^^^^^^^^^^^^^^^^^^^^^

Install all extraction capabilities:

.. code-block:: bash

   # Install everything
   pip install localvectordb[all]

   # Or specifically file extraction
   pip install localvectordb[file-extraction]

Selective Installation
^^^^^^^^^^^^^^^^^^^^^^

Install only the extractors you need:

.. code-block:: bash

   # Office documents only
   pip install localvectordb[file-extraction-office]

   # PDFs only
   pip install localvectordb[file-extraction-pdf]

   # Web formats only
   pip install localvectordb[file-extraction-web]

   # E-books and RTF only
   pip install localvectordb[file-extraction-rtf-epub]

   # Combine multiple groups
   pip install localvectordb[file-extraction-office,file-extraction-pdf]

Checking Available Extractors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See what extractors are currently available:

.. code-block:: python

   from localvectordb.extractors import get_supported_formats

   # Get all supported formats
   formats = get_supported_formats()

   for format_name, info in formats.items():
       print(f"{format_name}: {info['extractors'][0]['name']}")

   # Example output:
   # pdf: PDFPlumberExtractor
   # docx: DocxExtractor
   # txt: TextFileExtractor

Using the Extraction System
----------------------------

Server Upload API
^^^^^^^^^^^^^^^^^

The most common way to use file extraction is through the server upload API:

.. code-block:: bash

   # Upload via curl
   curl -X POST \
        -H "Authorization: Bearer your_api_key" \
        -F "files=@document.pdf" \
        -F "metadata={\"category\": \"research\"}" \
        http://localhost:5000/api/v1/mydatabase/upload

**Python Client**:

.. code-block:: python

   from localvectordb import RemoteVectorDB

   # Connect to server
   db = RemoteVectorDB("mydatabase", "http://localhost:5000", api_key="your_key")

   # Upload file with automatic extraction
   with open("document.pdf", "rb") as f:
       result = db.upload_file(
           file_content=f.read(),
           filename="document.pdf",
           metadata={"category": "research", "year": 2024}
       )

   print(f"Extracted {len(result['documents'])} documents")
   print(f"Total text length: {sum(len(doc['content']) for doc in result['documents'])}")

Direct Extraction
^^^^^^^^^^^^^^^^^

You can also use the extraction system directly without uploading:

.. code-block:: python

   from localvectordb.extractors import ExtractorRegistry

   # Extract text from file content
   with open("document.pdf", "rb") as f:
       file_content = f.read()

   result = ExtractorRegistry.extract_text(
       file_content=file_content,
       filename="document.pdf"
   )

   if result.success:
       print(f"Extracted text: {result.text[:500]}...")
       print(f"Extraction method: {result.method}")
       print(f"Metadata: {result.metadata}")
   else:
       print(f"Extraction failed: {result.error}")

Preview Extraction
^^^^^^^^^^^^^^^^^^

Preview extraction without adding to the database:

.. code-block:: bash

   # Preview extraction via API
   curl -X POST \
        -H "Authorization: Bearer your_api_key" \
        -F "file=@document.pdf" \
        http://localhost:5000/api/v1/extract/preview

**Response**:

.. code-block:: json

   {
     "success": true,
     "text": "Document content here...",
     "extraction_method": "PDFPlumberExtractor",
     "metadata": {
       "pages": 10,
       "author": "John Doe",
       "creation_date": "2024-01-15"
     },
     "text_length": 5420,
     "chunks_preview": [
       {
         "content": "First chunk of text...",
         "position": 0,
         "metadata": {}
       }
     ]
   }

Metadata Extraction
-------------------

Each extractor can extract format-specific metadata alongside text content:

Common Metadata Fields
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Extracted Metadata Fields
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Formats
     - Description
   * - ``author``
     - PDF, DOCX, PPTX, XLSX
     - Document author
   * - ``title``
     - PDF, HTML, EPUB
     - Document title
   * - ``creation_date``
     - PDF, Office formats
     - When document was created
   * - ``modification_date``
     - PDF, Office formats
     - Last modification date
   * - ``pages``
     - PDF, DOCX
     - Number of pages
   * - ``word_count``
     - DOCX
     - Approximate word count
   * - ``language``
     - PDF, HTML
     - Document language
   * - ``keywords``
     - PDF, Office formats
     - Document keywords/tags

Using Extracted Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Upload with metadata schema that captures extraction metadata
   db = LocalVectorDB(
       name="documents",
       metadata_schema={
           "author": {"type": "text", "indexed": True},
           "title": {"type": "text", "indexed": True},
           "pages": {"type": "integer", "indexed": True},
           "creation_date": {"type": "date", "indexed": True},
           "extraction_method": {"type": "text", "indexed": False}
       }
   )

   # Upload file - metadata will be automatically extracted and stored
   with open("research_paper.pdf", "rb") as f:
       result = db.upload_file(
           file_content=f.read(),
           filename="research_paper.pdf",
           metadata={"category": "research"}  # Additional manual metadata
       )

   # Query by extracted metadata
   results = db.filter(where={"author": "Jane Smith", "pages": {"$gte": 10}})

Extractor Priority System
--------------------------

When multiple extractors can handle the same file format, LocalVectorDB uses a priority system to select the best one:

Priority Levels
^^^^^^^^^^^^^^^

.. list-table:: Extractor Priorities
   :header-rows: 1
   :widths: 20 15 65

   * - Extractor
     - Priority
     - Reasoning
   * - PDFPlumberExtractor
     - 20
     - Better table extraction and layout preservation
   * - PyPDFExtractor
     - 15
     - Faster but less accurate for complex layouts
   * - DocxExtractor
     - 10
     - Standard priority for office formats
   * - TextFileExtractor
     - 5
     - Fallback for any text-readable format

**Priority Selection Example**:

.. code-block:: python

   from localvectordb.extractors import ExtractorRegistry

   # Get extractors for a PDF file (returns list sorted by priority)
   extractors = ExtractorRegistry.get_extractors_for_file("document.pdf")

   for extractor in extractors:
       print(f"{extractor.name}: priority {extractor.priority}")

   # Output:
   # PDFPlumberExtractor: priority 20
   # PyPDFExtractor: priority 15

**Manual Extractor Selection**:

.. code-block:: python

   # Force a specific extractor
   pypdf_extractor = ExtractorRegistry.get_extractor("PyPDFExtractor")

   if pypdf_extractor and pypdf_extractor.available:
       result = pypdf_extractor.extract_text(file_content, "document.pdf")

Creating Custom Extractors
---------------------------

You can extend the extraction system with custom extractors for specialized formats.

Basic Custom Extractor
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from localvectordb.extractors import BaseExtractor, ExtractionResult
   from localvectordb.core import MetadataField

   class CustomCSVExtractor(BaseExtractor):
       """Extract text from CSV files by converting to readable format."""

       @property
       def supported_extensions(self):
           return ['.csv']

       @property
       def supported_mimetypes(self):
           return ['text/csv', 'application/csv']

       @property
       def required_packages(self):
           return ['pandas']  # Optional dependency

       @property
       def priority(self):
           return 10

       @property
       def metadata_schema(self):
           return {
               'rows': MetadataField(type='integer', indexed=True),
               'columns': MetadataField(type='integer', indexed=True),
               'encoding': MetadataField(type='text', indexed=False)
           }

       def _check_availability(self):
           try:
               import pandas
               return True
           except ImportError:
               return False

       def _extract_text_impl(self, file_content, filename, mimetype, **kwargs):
           import pandas as pd
           from io import StringIO

           try:
               # Detect encoding
               import chardet
               encoding = chardet.detect(file_content)['encoding'] or 'utf-8'

               # Read CSV
               csv_text = file_content.decode(encoding)
               df = pd.read_csv(StringIO(csv_text))

               # Convert to readable text
               text_content = f"CSV File: {filename}\n\n"
               text_content += df.to_string(index=False)

               # Extract metadata
               metadata = {
                   'rows': len(df),
                   'columns': len(df.columns),
                   'encoding': encoding,
                   'column_names': list(df.columns)
               }

               return ExtractionResult(
                   text=text_content,
                   success=True,
                   method=self.name,
                   metadata=metadata
               )

           except Exception as e:
               return ExtractionResult(
                   text="",
                   success=False,
                   method=self.name,
                   error=str(e)
               )

Registering Custom Extractors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Method 1: Direct Registration**

.. code-block:: python

   from localvectordb.extractors import ExtractorRegistry

   # Register the custom extractor
   ExtractorRegistry.register(CustomCSVExtractor)

   # Now it's available for extraction
   result = ExtractorRegistry.extract_text(csv_content, "data.csv")

**Method 2: Entry Points (Recommended for packages)**

In your package's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."localvectordb.file_extractors"]
   csv = "mypackage.extractors:CustomCSVExtractor"

The extractor will be automatically discovered when LocalVectorDB starts.

Advanced Features
-----------------

Fallback Chain
^^^^^^^^^^^^^^

The extraction system includes a fallback chain for robustness:

1. **Format-specific extractors** (e.g., PDFPlumberExtractor for PDFs)
2. **Alternative extractors** (e.g., PyPDFExtractor as backup)
3. **Text fallback extractor** (attempts to read as plain text)
4. **Failure result** (if no extractor can handle the file)

.. code-block:: python

   # The fallback chain in action
   def extract_with_fallback(file_content, filename):
       # Try specific extractors first
       extractors = ExtractorRegistry.get_extractors_for_file(filename)

       for extractor in extractors:
           result = extractor.extract_text(file_content, filename)
           if result.success:
               return result

       # Try text fallback
       text_extractor = ExtractorRegistry.get_extractor('TextFileExtractor')
       if text_extractor:
           return text_extractor.extract_text(file_content, filename)

       # Complete failure
       return ExtractionResult(
           text="", success=False,
           error="No suitable extractor found"
       )

Batch Processing
^^^^^^^^^^^^^^^^

Process multiple files efficiently:

.. code-block:: python

   import os
   from pathlib import Path

   def batch_extract_directory(directory_path):
       """Extract text from all files in a directory."""
       results = []

       for file_path in Path(directory_path).iterdir():
           if file_path.is_file():
               with open(file_path, 'rb') as f:
                   file_content = f.read()

               result = ExtractorRegistry.extract_text(
                   file_content, file_path.name
               )

               results.append({
                   'filename': file_path.name,
                   'success': result.success,
                   'text_length': len(result.text) if result.success else 0,
                   'method': result.method,
                   'error': result.error
               })

       return results

   # Usage
   extraction_results = batch_extract_directory("./documents/")
   successful = [r for r in extraction_results if r['success']]
   print(f"Successfully extracted {len(successful)} of {len(extraction_results)} files")

Performance Considerations
--------------------------

Memory Usage
^^^^^^^^^^^^

- **Large Files**: Extract text in chunks for files over 100MB
- **Batch Processing**: Process files sequentially to avoid memory spikes
- **Streaming**: Use streaming extraction for very large documents

.. code-block:: python

   # Memory-efficient extraction for large files
   def extract_large_file(file_path, chunk_size_mb=50):
       """Extract large files in chunks to manage memory."""
       file_size = os.path.getsize(file_path)
       chunk_size = chunk_size_mb * 1024 * 1024

       if file_size <= chunk_size:
           # Small file, extract normally
           with open(file_path, 'rb') as f:
               return ExtractorRegistry.extract_text(f.read(), file_path.name)

       # Large file, consider alternative approaches
       print(f"Warning: Large file ({file_size / (1024*1024):.1f}MB). "
             "Consider preprocessing or chunking.")

       with open(file_path, 'rb') as f:
           return ExtractorRegistry.extract_text(f.read(), file_path.name)

Extractor Selection
^^^^^^^^^^^^^^^^^^^

Choose extractors based on your priorities:

.. code-block:: python

   # For speed, prefer simpler extractors
   def get_fast_extractor(filename):
       extractors = ExtractorRegistry.get_extractors_for_file(filename)
       # Choose the fastest available extractor
       speed_priority = {
           'TextFileExtractor': 1,
           'PyPDFExtractor': 2,
           'PDFPlumberExtractor': 3,  # Slower but more accurate
       }

       return min(extractors, key=lambda x: speed_priority.get(x.name, 999))

   # For accuracy, prefer advanced extractors
   def get_accurate_extractor(filename):
       extractors = ExtractorRegistry.get_extractors_for_file(filename)
       # Use default priority system (accuracy-focused)
       return extractors[0] if extractors else None

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**"No suitable extractor found"**

- Install the required optional dependencies
- Check that the file format is supported
- Verify file extension matches the content

.. code-block:: python

   # Diagnose extraction issues
   def diagnose_extraction_failure(filename):
       print(f"Diagnosing extraction for: {filename}")

       # Check available extractors
       extractors = ExtractorRegistry.get_extractors_for_file(filename)
       if not extractors:
           print("❌ No extractors found for this file")
           print("Available extractors:")
           for name in ExtractorRegistry.list_extractors():
               extractor = ExtractorRegistry.get_extractor(name)
               print(f"  {name}: {extractor.supported_extensions}")
           return

       # Check extractor availability
       for extractor in extractors:
           if extractor.available:
               print(f"✅ {extractor.name} is available")
           else:
               print(f"❌ {extractor.name} missing packages: {extractor.required_packages}")

**"Required packages not available"**

Install the missing optional dependencies:

.. code-block:: bash

   # Check what's missing
   python -c "
   from localvectordb.extractors import ExtractorRegistry
   for name in ExtractorRegistry.list_extractors(available_only=False):
       extractor = ExtractorRegistry.get_extractor(name)
       if not extractor.available:
           print(f'{name}: missing {extractor.required_packages}')
   "

   # Install missing packages
   pip install package_name

**Memory errors with large files**

- Process files in smaller chunks
- Use streaming extraction when available
- Increase system memory or use a more powerful machine

Debug Mode
^^^^^^^^^^

Enable debug logging to see extraction details:

.. code-block:: python

   import logging
   logging.getLogger('localvectordb.extractors').setLevel(logging.DEBUG)

   # Now extraction will show detailed logs
   result = ExtractorRegistry.extract_text(file_content, filename)

Best Practices
--------------

File Format Handling
^^^^^^^^^^^^^^^^^^^^^

1. **Install Relevant Dependencies**: Only install extractors for formats you actually use
2. **Test Extraction Quality**: Verify that extracted text meets your quality requirements
3. **Handle Failures Gracefully**: Always check ``result.success`` before using extracted text
4. **Monitor Performance**: Track extraction times for different file types and sizes

Metadata Management
^^^^^^^^^^^^^^^^^^^

1. **Define Schema First**: Set up metadata schema to capture extracted fields
2. **Validate Extracted Metadata**: Some extracted metadata may be unreliable
3. **Combine with Manual Metadata**: Augment extracted metadata with manually provided fields
4. **Index Important Fields**: Index metadata fields you'll frequently search or filter by

Production Deployment
^^^^^^^^^^^^^^^^^^^^^

1. **Resource Planning**: Allocate sufficient memory and CPU for extraction workloads
2. **Error Monitoring**: Monitor extraction failure rates and common errors
3. **Performance Tuning**: Optimize extractor selection based on your file mix
4. **Backup Strategy**: Consider extracting and storing text separately as backup

.. code-block:: python

   # Production-ready extraction with monitoring
   import time
   import logging

   class ProductionExtractor:
       def __init__(self):
           self.logger = logging.getLogger(__name__)
           self.metrics = {'successes': 0, 'failures': 0, 'total_time': 0}

       def extract_with_monitoring(self, file_content, filename):
           start_time = time.time()

           try:
               result = ExtractorRegistry.extract_text(file_content, filename)

               if result.success:
                   self.metrics['successes'] += 1
                   self.logger.info(f"Extracted {filename}: {len(result.text)} chars")
               else:
                   self.metrics['failures'] += 1
                   self.logger.error(f"Failed to extract {filename}: {result.error}")

               return result

           except Exception as e:
               self.metrics['failures'] += 1
               self.logger.error(f"Exception extracting {filename}: {e}")
               raise

           finally:
               self.metrics['total_time'] += time.time() - start_time

       def get_metrics(self):
           total = self.metrics['successes'] + self.metrics['failures']
           if total > 0:
               success_rate = self.metrics['successes'] / total * 100
               avg_time = self.metrics['total_time'] / total
               return {
                   'success_rate': f"{success_rate:.1f}%",
                   'average_time': f"{avg_time:.2f}s",
                   'total_processed': total
               }
           return {'message': 'No files processed yet'}

See Also
--------

- :ref:`installation` - Installing optional dependencies
- :ref:`server-routes` - File upload API endpoints
- :ref:`metadata-filtering` - Working with extracted metadata
- :ref:`cli` - Command-line file upload tools