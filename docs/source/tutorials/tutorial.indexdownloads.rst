===========================
Index Your Downloads Folder
===========================

Turn your messy Downloads folder into a searchable knowledge base! This tutorial shows you how to index all your downloaded documents and find them instantly with natural language search.

The Problem We're Solving
=========================

Sound familiar?

- Downloads folder with hundreds of files
- Can't remember what you named that PDF
- Endless scrolling to find that document
- "I know I downloaded something about..."

**Solution**: Index everything and search by content, not filename!

Quick Start (10 Lines)
======================

Create ``index_downloads.py``:

.. code-block:: python

   from localvectordb import LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType
   import os
   from pathlib import Path

   # Create database. NOTE: metadata is only stored for fields declared in the
   # metadata_schema -- anything else is dropped with a warning. We declare "filename"
   # and "file_path" so they actually persist and appear in search results.
   metadata_schema = {
       "filename": MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=True),
       "file_path": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
   }
   db = LocalVectorDB(
       name="downloads_search",
       metadata_schema=metadata_schema,
       embedding_provider="ollama",
       embedding_model="nomic-embed-text",
   )

   # Find your Downloads folder
   downloads_path = Path.home() / "Downloads"
   print(f"Scanning: {downloads_path}")

   # Index text files
   for file_path in downloads_path.glob("*.txt"):
       try:
           with open(file_path, 'r', encoding='utf-8') as f:
               content = f.read()
           if content.strip():  # Skip empty files
               db.upsert([content], metadata=[{"filename": file_path.name, "file_path": str(file_path)}])
               print(f"Indexed: {file_path.name}")
       except Exception as e:
           print(f"Skipped {file_path.name}: {e}")

   print(f"\nDone! Indexed {db.get_stats()['documents']} files")
   print("Try: python search_downloads.py")

Create ``search_downloads.py``:

.. code-block:: python

   from localvectordb import LocalVectorDB

   db = LocalVectorDB(name="downloads_search", create_if_not_exists=False)

   while True:
       query = input("\nSearch your downloads: ").strip()
       if query.lower() in ['quit', 'exit']:
           break
       
       results = db.query(query, k=5)
       print(f"\nFound {len(results)} matches:")
       for i, result in enumerate(results, 1):
           filename = result.metadata.get('filename', 'Unknown')
           print(f"{i}. {filename} (score: {result.score:.3f})")
           print(f"   Preview: {result.content[:100]}...")

Run it:

.. code-block:: bash

   python index_downloads.py
   python search_downloads.py

Complete Solution (Production Ready)
====================================

Here's a full-featured version that handles multiple file types:

.. code-block:: python

   from localvectordb import LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType
   from pathlib import Path
   import mimetypes
   import json
   from datetime import datetime

   class DownloadsIndexer:
       """Index and search your Downloads folder."""
       
       def __init__(self, downloads_path=None):
           # Define metadata schema for file information
           metadata_schema = {
               'filename': MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=True),
               'file_path': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'file_type': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'file_size': MetadataField(type=MetadataFieldType.INTEGER),
               'date_modified': MetadataField(type=MetadataFieldType.DATE, indexed=True),
               'file_extension': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
           }
           
           self.db = LocalVectorDB(
               name="downloads_index",
               base_path="./downloads_db",
               metadata_schema=metadata_schema,
               embedding_provider="ollama",
               embedding_model="nomic-embed-text"
           )
           
           self.downloads_path = Path(downloads_path) if downloads_path else Path.home() / "Downloads"
           
           # Supported file types
           self.supported_extensions = {
               '.txt': self._read_text_file,
               '.md': self._read_text_file,
               '.py': self._read_text_file,
               '.js': self._read_text_file,
               '.html': self._read_text_file,
               '.css': self._read_text_file,
               '.json': self._read_json_file,
               '.csv': self._read_csv_file,
           }
       
       def _read_text_file(self, file_path):
           """Read plain text files."""
           with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
               return f.read()
       
       def _read_json_file(self, file_path):
           """Read JSON files."""
           with open(file_path, 'r', encoding='utf-8') as f:
               data = json.load(f)
               return json.dumps(data, indent=2)
       
       def _read_csv_file(self, file_path):
           """Read CSV files."""
           import csv
           content = []
           with open(file_path, 'r', encoding='utf-8') as f:
               reader = csv.reader(f)
               for row in reader:
                   content.append(', '.join(row))
           return '\n'.join(content)
       
       def index_downloads(self, max_files=100):
           """Index all supported files in Downloads folder."""
           
           print(f"Scanning Downloads folder: {self.downloads_path}")
           print(f"Looking for: {list(self.supported_extensions.keys())}")
           
           indexed_count = 0
           skipped_count = 0
           
           # Get all files (not just direct children)
           all_files = []
           for ext in self.supported_extensions.keys():
               all_files.extend(self.downloads_path.rglob(f"*{ext}"))
           
           # Limit files to avoid overwhelming
           all_files = all_files[:max_files]
           
           print(f"Found {len(all_files)} files to process")
           
           for file_path in all_files:
               try:
                   # Skip if already indexed (check by path)
                   existing = self.db.filter(where={"file_path": str(file_path)}, limit=1)
                   if existing:
                       print(f"Already indexed: {file_path.name}")
                       continue
                   
                   # Read file content
                   extension = file_path.suffix.lower()
                   if extension in self.supported_extensions:
                       content = self.supported_extensions[extension](file_path)
                   else:
                       continue
                   
                   if not content.strip():
                       print(f"Empty file: {file_path.name}")
                       continue
                   
                   # Get file stats
                   stat = file_path.stat()
                   
                   # Prepare metadata
                   metadata = {
                       'filename': file_path.name,
                       'file_path': str(file_path),
                       'file_type': mimetypes.guess_type(str(file_path))[0] or 'unknown',
                       'file_size': stat.st_size,
                       'date_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                       'file_extension': extension
                   }
                   
                   # Add to database
                   doc_ids = self.db.upsert([content], metadata=[metadata])
                   
                   indexed_count += 1
                   print(f"Indexed: {file_path.name} ({stat.st_size} bytes)")
                   
               except Exception as e:
                   skipped_count += 1
                   print(f"Error with {file_path.name}: {e}")
           
           print(f"\nIndexing complete!")
           print(f"Successfully indexed: {indexed_count} files")
           print(f"Skipped: {skipped_count} files")
           print(f"Total in database: {self.db.get_stats()['documents']} documents")
       
       def search(self, query, max_results=10):
           """Search indexed files."""
           results = self.db.query(
               query=query,
               search_type="hybrid",  # Use both semantic and keyword search
               k=max_results,
               score_threshold=0.1
           )
           
           return [(result, result.metadata) for result in results]
       
       def search_by_file_type(self, query, file_extension=None, max_results=10):
           """Search within specific file types."""
           filters = {}
           if file_extension:
               filters['file_extension'] = file_extension
           
           results = self.db.query(
               query=query,
               search_type="hybrid",
               k=max_results,
               filters=filters
           )
           
           return [(result, result.metadata) for result in results]
       
       def get_file_stats(self):
           """Get statistics about indexed files."""
           # Get all documents to analyze
           all_docs = self.db.filter(limit=1000)
           
           if not all_docs:
               return {"message": "No files indexed yet"}
           
           # Analyze by file type
           by_extension = {}
           total_size = 0
           
           for doc in all_docs:
               ext = doc.metadata.get('file_extension', 'unknown')
               size = doc.metadata.get('file_size', 0)
               
               if ext not in by_extension:
                   by_extension[ext] = {'count': 0, 'total_size': 0}
               
               by_extension[ext]['count'] += 1
               by_extension[ext]['total_size'] += size
               total_size += size
           
           return {
               'total_files': len(all_docs),
               'total_size_mb': total_size / (1024 * 1024),
               'by_file_type': by_extension,
               'database_stats': self.db.get_stats()
           }

   def main():
       """Interactive main function."""
       indexer = DownloadsIndexer()
       
       print("Downloads Folder Indexer")
       print("=" * 30)
       
       while True:
           print("\nWhat would you like to do?")
           print("1. Index Downloads folder")
           print("2. Search files")
           print("3. Search specific file type")
           print("4. View statistics")
           print("5. Quit")
           
           choice = input("\nChoice (1-5): ").strip()
           
           if choice == '1':
               max_files = input("Max files to index (default 100): ").strip()
               max_files = int(max_files) if max_files.isdigit() else 100
               indexer.index_downloads(max_files)
           
           elif choice == '2':
               query = input("\nEnter search query: ").strip()
               if query:
                   results = indexer.search(query)
                   
                   if results:
                       print(f"\nFound {len(results)} matches:")
                       for i, (result, metadata) in enumerate(results, 1):
                           filename = metadata.get('filename', 'Unknown')
                           file_path = metadata.get('file_path', '')
                           size_mb = metadata.get('file_size', 0) / 1024 / 1024
                           
                           print(f"\n{i}. {filename}")
                           print(f"   Path: {file_path}")
                           print(f"   Size: {size_mb:.2f} MB")
                           print(f"   Score: {result.score:.3f}")
                           print(f"   Preview: {result.content[:150]}...")
                   else:
                       print("No matches found. Try different search terms.")
           
           elif choice == '3':
               ext = input("File extension (e.g., .txt, .json): ").strip()
               query = input("Search query: ").strip()
               if query:
                   results = indexer.search_by_file_type(query, ext)
                   print(f"\nFound {len(results)} matches in {ext} files:")
                   for i, (result, metadata) in enumerate(results, 1):
                       print(f"{i}. {metadata.get('filename', 'Unknown')} (score: {result.score:.3f})")
           
           elif choice == '4':
               stats = indexer.get_file_stats()
               print(f"\nFile Statistics:")
               print(f"Total files: {stats.get('total_files', 0)}")
               print(f"Total size: {stats.get('total_size_mb', 0):.2f} MB")
               
               if 'by_file_type' in stats:
                   print("\nBy file type:")
                   for ext, data in stats['by_file_type'].items():
                       print(f"  {ext}: {data['count']} files ({data['total_size']/1024/1024:.2f} MB)")
           
           elif choice == '5':
               print("Goodbye!")
               break
           
           else:
               print("Derp! Invalid choice. Please enter 1-5.")

   if __name__ == "__main__":
       main()

Quick Search Script
===================

For daily use, create this simple search script ``find.py``:

.. code-block:: python

   #!/usr/bin/env python3
   import sys
   from localvectordb import LocalVectorDB

   if len(sys.argv) < 2:
       print("Usage: python find.py 'your search query'")
       sys.exit(1)

   query = ' '.join(sys.argv[1:])

   try:
       db = LocalVectorDB(name="downloads_index", base_path="./downloads_db", create_if_not_exists=False)
       results = db.query(query, k=5)
       
       if results:
           print(f"Found {len(results)} matches for '{query}':")
           for i, result in enumerate(results, 1):
               filename = result.metadata.get('filename', 'Unknown')
               path = result.metadata.get('file_path', '')
               print(f"\n{i}. {filename}")
               print(f"   {path}")
               print(f"   {result.content[:100]}...")
       else:
           print(f"No matches found for '{query}'")
           
   except Exception as e:
       print(f"Error: {e}")
       print("Make sure you've indexed your downloads first!")

Then search from the command line:

.. code-block:: bash

   python find.py "machine learning tutorial"
   python find.py "python code"
   python find.py "meeting notes"

Advanced Features
=================

**Built-in Extraction for PDF, DOCX, and More**

You don't need to hand-roll readers for binary formats. LocalVectorDB ships with
built-in extraction (via `all2md <https://all2md.readthedocs.io/>`_) covering
PDF, DOCX, PPTX, XLSX, HTML, EPUB, ODT, and 20+ other document formats. Install
the extended-format parsers once::

   pip install "localvectordb[file-extraction]"

Then let the database extract and ingest files for you with
:meth:`~localvectordb.database.LocalVectorDB.upsert_from_file`, which replaces the manual
``_read_*`` helpers above for these formats:

.. code-block:: python

   # PDFs, Word docs, spreadsheets, slides — extracted automatically
   self.db.upsert_from_file(
       [str(p) for p in self.downloads_path.glob("*.pdf")],
   )

Extracted content is **Markdown**, so headings, tables, and lists are preserved
— which makes ``chunking_method="sections"`` an effective choice for these
documents. See :doc:`/file-extraction` for the full format list and security
options.

**Automatic Re-indexing**:

.. code-block:: python

   def auto_reindex(self):
       """Check for new/modified files and re-index."""
       # Compare file modification times with database
       # Re-index only changed files

Real-World Usage Examples
==========================

**Find that Research Paper**:
```
Search: "neural networks deep learning"
Found: "Deep_Learning_Review_2023.pdf"
```

**Locate Code Examples**:
```
Search: "python web scraping"
Found: "web_scraper_tutorial.py"
```

**Find Meeting Notes**:
```
Search: "quarterly review meeting"
Found: "Q4_Review_Notes.txt"
```

Tips for Best Results
=====================

1. **Index Regularly**: Run the indexer weekly to catch new downloads
2. **Use Descriptive Searches**: "machine learning tutorial" vs just "ML"
3. **Try Different File Types**: Search within .txt, .pdf, .md files separately
4. **Check File Previews**: Read the content preview to confirm relevance
5. **Organize Before Indexing**: Consider organizing Downloads first for better metadata

Troubleshooting
===============

**No Files Found**: Check that your Downloads path is correct and contains supported file types

**Encoding Errors**: Some files might have unusual encoding - the script handles this gracefully

**Large Files**: Very large files might be slow to index - consider the max_files limit

**Memory Usage**: For thousands of files, index in batches

What's Next?
============

Your Downloads folder is now searchable! Consider:

- **Expand to Other Folders**: Index Documents, Desktop, or project folders
- **Index Scanned PDFs**: Enable OCR with ``pip install "localvectordb[file-extraction-ocr]"``
- **Web Interface**: Build a simple web UI for family/team use
- **Scheduled Indexing**: Set up automatic re-indexing with cron jobs
