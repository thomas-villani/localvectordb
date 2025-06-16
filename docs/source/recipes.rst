=======
Recipes
=======

This page contains practical recipes for common tasks when working with LocalVectorDB. Each recipe includes complete,
runnable code examples that you can adapt for your specific use case.

.. contents:: Recipe Index
   :local:
   :depth: 2

Database Setup and Configuration
================================

Creating a Database with Custom Settings
----------------------------------------

.. code-block:: python

   from localvectordb import LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType

   # Create a database optimized for research papers
   def create_research_database():
       metadata_schema = {
           'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=True),
           'authors': MetadataField(type=MetadataFieldType.JSON, indexed=False),
           'journal': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'publication_date': MetadataField(type=MetadataFieldType.DATE, indexed=True),
           'doi': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'keywords': MetadataField(type=MetadataFieldType.JSON),
           'citation_count': MetadataField(type=MetadataFieldType.INTEGER),
           'impact_factor': MetadataField(type=MetadataFieldType.REAL)
       }

       db = LocalVectorDB(
           name="research_papers",
           base_path="./research_db",
           metadata_schema=metadata_schema,
           embedding_provider="ollama",
           embedding_model="nomic-embed-text",
           chunking_method="sentences",
           chunk_size=800,  # Larger chunks for academic content
           chunk_overlap=100,
           enable_fts=True
       )

       print(f"Created research database with {db.embedding_dimension}D embeddings")
       return db

Using Different Embedding Providers
-----------------------------------

.. code-block:: python

   # Ollama embeddings (local, free)
   def create_ollama_db():
       return LocalVectorDB(
           name="ollama_db",
           base_path="./ollama_storage",
           embedding_provider="ollama",
           embedding_model="nomic-embed-text",
           embedding_config={
               "base_url": "http://localhost:11434"  # Default Ollama URL
           }
       )

   # OpenAI embeddings (cloud, paid)
   def create_openai_db():
       return LocalVectorDB(
           name="openai_db",
           base_path="./openai_storage",
           embedding_provider="openai",
           embedding_model="text-embedding-3-small",
           embedding_config={
               "api_key": "your-openai-api-key"  # Or set OPENAI_API_KEY env var
           }
       )

   # Custom Hugging Face embeddings (using custom provider)
   def create_huggingface_db():
       # Assuming you've registered a custom HF provider
       return LocalVectorDB(
           name="hf_db",
           base_path="./hf_storage",
           embedding_provider="huggingface",
           embedding_model="sentence-transformers/all-MiniLM-L6-v2",
           embedding_config={
               "device": "auto",
               "batch_size": 64
           }
       )

Database Factory Pattern
------------------------

.. code-block:: python

   from typing import Dict, Any, Optional
   import os

   class DatabaseFactory:
       """Factory for creating LocalVectorDB instances with common configurations."""

       PRESET_CONFIGS = {
           "general": {
               "chunk_size": 500,
               "chunking_method": "sentences",
               "chunk_overlap": 50
           },
           "code": {
               "chunk_size": 800,
               "chunking_method": "code-blocks",
               "chunk_overlap": 100
           },
           "academic": {
               "chunk_size": 1000,
               "chunking_method": "paragraphs",
               "chunk_overlap": 150
           },
           "chat": {
               "chunk_size": 300,
               "chunking_method": "sentences",
               "chunk_overlap": 25
           }
       }

       @classmethod
       def create_database(
           cls,
           name: str,
           preset: str = "general",
           base_path: str = "./vector_dbs",
           metadata_schema: Optional[Dict[str, MetadataField]] = None,
           **kwargs
       ) -> LocalVectorDB:
           """Create a database with preset configuration."""

           if preset not in cls.PRESET_CONFIGS:
               raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.PRESET_CONFIGS.keys())}")

           config = cls.PRESET_CONFIGS[preset].copy()
           config.update(kwargs)  # Allow overrides

           return LocalVectorDB(
               name=name,
               base_path=base_path,
               metadata_schema=metadata_schema,
               **config
           )

   # Usage examples
   general_db = DatabaseFactory.create_database("my_docs", "general")
   code_db = DatabaseFactory.create_database("my_code", "code")
   academic_db = DatabaseFactory.create_database("papers", "academic")

Document Ingestion Patterns
============================

Batch Processing Large Document Collections
--------------------------------------------

.. code-block:: python

   from pathlib import Path
   from typing import List, Generator, Dict, Any
   import logging

   def process_document_directory(
       db: LocalVectorDB,
       directory_path: str,
       batch_size: int = 50,
       file_patterns: List[str] = None
   ) -> List[str]:
       """Process all documents in a directory with progress tracking."""

       if file_patterns is None:
           file_patterns = ["*.txt", "*.md", "*.py", "*.js", "*.html"]

       # Find all matching files
       directory = Path(directory_path)
       all_files = []
       for pattern in file_patterns:
           all_files.extend(directory.rglob(pattern))

       print(f"Found {len(all_files)} files to process")

       all_doc_ids = []

       # Process in batches
       for i in range(0, len(all_files), batch_size):
           batch_files = all_files[i:i + batch_size]
           print(f"Processing batch {i//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size}")

           documents = []
           metadata = []

           for file_path in batch_files:
               try:
                   # Read file content
                   with open(file_path, 'r', encoding='utf-8') as f:
                       content = f.read()

                   if not content.strip():
                       continue

                   documents.append(content)
                   metadata.append({
                       'filename': file_path.name,
                       'file_path': str(file_path),
                       'file_extension': file_path.suffix,
                       'file_size': len(content),
                       'directory': str(file_path.parent)
                   })

               except Exception as e:
                   logging.warning(f"Failed to process {file_path}: {e}")

           # Insert batch
           if documents:
               try:
                   batch_ids = db.upsert(
                       documents=documents,
                       metadata=metadata,
                       batch_size=batch_size,
                       similarity_threshold=0.9  # Avoid near-duplicates
                   )
                   all_doc_ids.extend(batch_ids)
                   print(f"Successfully added {len(batch_ids)} documents")
               except Exception as e:
                   logging.error(f"Failed to insert batch: {e}")

       print(f"Total documents added: {len(all_doc_ids)}")
       return all_doc_ids

Processing Different File Types
-------------------------------

.. code-block:: python

   import mimetypes
   from typing import Optional

   def smart_document_processor(file_path: str) -> Optional[Dict[str, Any]]:
       """Process different file types intelligently."""

       path = Path(file_path)
       mime_type, _ = mimetypes.guess_type(str(path))

       try:
           if mime_type == 'text/plain' or path.suffix in ['.txt', '.md', '.py', '.js', '.html', '.css']:
               # Plain text files
               with open(path, 'r', encoding='utf-8') as f:
                   content = f.read()

           elif path.suffix == '.pdf':
               # PDF files (requires PyPDF2 or similar)
               try:
                   import PyPDF2
                   with open(path, 'rb') as f:
                       reader = PyPDF2.PdfReader(f)
                       content = '\n'.join([page.extract_text() for page in reader.pages])
               except ImportError:
                   print("PyPDF2 not installed. Install with: pip install PyPDF2")
                   return None

           elif path.suffix in ['.doc', '.docx']:
               # Word documents (requires python-docx)
               try:
                   from docx import Document
                   doc = Document(path)
                   content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
               except ImportError:
                   print("python-docx not installed. Install with: pip install python-docx")
                   return None

           elif path.suffix == '.json':
               # JSON files
               import json
               with open(path, 'r', encoding='utf-8') as f:
                   data = json.load(f)
                   content = json.dumps(data, indent=2)

           else:
               print(f"Unsupported file type: {path.suffix}")
               return None

           return {
               'content': content,
               'metadata': {
                   'filename': path.name,
                   'file_type': mime_type or 'unknown',
                   'file_extension': path.suffix,
                   'file_size': path.stat().st_size,
                   'character_count': len(content)
               }
           }

       except Exception as e:
           print(f"Error processing {path}: {e}")
           return None

   # Usage with database
   def add_file_to_db(db: LocalVectorDB, file_path: str) -> Optional[str]:
       """Add a single file to database with automatic type detection."""

       doc_data = smart_document_processor(file_path)
       if doc_data:
           doc_ids = db.upsert(
               documents=[doc_data['content']],
               metadata=[doc_data['metadata']]
           )
           return doc_ids[0] if doc_ids else None
       return None

Streaming and Incremental Updates
----------------------------------

.. code-block:: python

   import time
   from datetime import datetime
   from typing import Iterator

   def stream_documents_to_db(
       db: LocalVectorDB,
       document_stream: Iterator[Dict[str, Any]],
       batch_size: int = 20,
       auto_save_interval: int = 100
   ):
       """Stream documents to database with periodic saves."""

       batch = []
       total_processed = 0
       last_save = time.time()

       for doc_data in document_stream:
           batch.append(doc_data)

           # Process batch when full
           if len(batch) >= batch_size:
               process_batch(db, batch)
               total_processed += len(batch)
               batch = []

               print(f"Processed {total_processed} documents...")

               # Auto-save periodically
               if time.time() - last_save > auto_save_interval:
                   db.save()
                   last_save = time.time()
                   print(f"Auto-saved at {datetime.now()}")

       # Process remaining documents
       if batch:
           process_batch(db, batch)
           total_processed += len(batch)

       # Final save
       db.save()
       print(f"Completed! Total processed: {total_processed}")

   def process_batch(db: LocalVectorDB, batch: List[Dict[str, Any]]):
       """Process a batch of documents."""
       documents = [item['content'] for item in batch]
       metadata = [item.get('metadata', {}) for item in batch]

       try:
           db.upsert(documents=documents, metadata=metadata)
       except Exception as e:
           print(f"Batch processing failed: {e}")

Search and Retrieval Patterns
==============================

Multi-Step Search Pipeline
---------------------------

.. code-block:: python

   from typing import List, Dict, Any, Optional

   class SearchPipeline:
       """Advanced search pipeline with multiple strategies."""

       def __init__(self, db: LocalVectorDB):
           self.db = db

       def comprehensive_search(
           self,
           query: str,
           strategies: List[str] = None,
           max_results: int = 10,
           score_threshold: float = 0.3
       ) -> Dict[str, List[Any]]:
           """Perform search using multiple strategies and combine results."""

           if strategies is None:
               strategies = ["vector", "keyword", "hybrid"]

           all_results = {}

           for strategy in strategies:
               try:
                   results = self.db.query(
                       query=query,
                       search_type=strategy,
                       k=max_results * 2,  # Get more results for better selection
                       score_threshold=score_threshold
                   )
                   all_results[strategy] = results
                   print(f"{strategy.title()} search: {len(results)} results")

               except Exception as e:
                   print(f"{strategy} search failed: {e}")
                   all_results[strategy] = []

           # Combine and rank results
           combined = self._combine_results(all_results, max_results)

           return {
               'combined': combined,
               'by_strategy': all_results
           }

       def _combine_results(self, results: Dict[str, List], max_results: int) -> List[Any]:
           """Combine results from different strategies."""

           # Create a scoring system that combines different search types
           doc_scores = {}

           for strategy, strategy_results in results.items():
               weight = {'vector': 0.4, 'keyword': 0.3, 'hybrid': 0.3}.get(strategy, 0.3)

               for result in strategy_results:
                   doc_id = result.id
                   if doc_id not in doc_scores:
                       doc_scores[doc_id] = {'result': result, 'total_score': 0, 'strategies': []}

                   doc_scores[doc_id]['total_score'] += result.score * weight
                   doc_scores[doc_id]['strategies'].append(strategy)

           # Sort by combined score and return top results
           ranked = sorted(
               doc_scores.values(),
               key=lambda x: x['total_score'],
               reverse=True
           )

           return [item['result'] for item in ranked[:max_results]]

Contextual Search with Filters
-------------------------------

.. code-block:: python

   def search_with_context(
       db: LocalVectorDB,
       query: str,
       context_filters: Dict[str, Any] = None,
       date_range: Optional[tuple] = None,
       category: Optional[str] = None
   ) -> List[Any]:
       """Search with contextual filters and constraints."""

       # Build dynamic filters
       filters = {}

       if context_filters:
           filters.update(context_filters)

       if date_range:
           start_date, end_date = date_range
           filters['publication_date'] = {
               '>=': start_date.isoformat(),
               '<=': end_date.isoformat()
           }

       if category:
           filters['category'] = category

       # Perform filtered search
       results = db.query(
           query=query,
           search_type="hybrid",
           k=20,
           filters=filters if filters else None
       )

       print(f"Found {len(results)} results with filters: {filters}")
       return results

   # Example usage
   from datetime import datetime, timedelta

   def find_recent_papers_about_ai(db: LocalVectorDB):
       """Find recent AI papers using contextual search."""

       last_year = datetime.now() - timedelta(days=365)
       today = datetime.now()

       results = search_with_context(
           db=db,
           query="artificial intelligence machine learning",
           date_range=(last_year, today),
           category="research",
           context_filters={
               'journal': {'IN': ['Nature', 'Science', 'IEEE', 'ACM']},
               'citation_count': {'>=': 10}
           }
       )

       return results

Semantic Search with Re-ranking
--------------------------------

.. code-block:: python

   def semantic_search_with_reranking(
       db: LocalVectorDB,
       query: str,
       initial_k: int = 50,
       final_k: int = 10,
       rerank_threshold: float = 0.7
   ) -> List[Any]:
       """Two-stage search: broad retrieval + semantic re-ranking."""

       # Stage 1: Broad retrieval
       initial_results = db.query(
           query=query,
           search_type="hybrid",
           k=initial_k,
           score_threshold=0.1  # Lower threshold for broad retrieval
       )

       print(f"Stage 1: Retrieved {len(initial_results)} candidates")

       if not initial_results:
           return []

       # Stage 2: Re-rank based on semantic similarity
       query_embedding = db.embedding_provider.embed_sync([query])[0]

       reranked_results = []
       for result in initial_results:
           # Get embedding for result content
           result_embedding = db.embedding_provider.embed_sync([result.content])[0]

           # Calculate cosine similarity
           import numpy as np
           similarity = np.dot(query_embedding, result_embedding) / (
               np.linalg.norm(query_embedding) * np.linalg.norm(result_embedding)
           )

           # Update score with semantic similarity
           if similarity >= rerank_threshold:
               result.score = similarity  # Override with semantic score
               reranked_results.append(result)

       # Sort by new scores and return top results
       reranked_results.sort(key=lambda x: x.score, reverse=True)

       print(f"Stage 2: Re-ranked to {len(reranked_results)} results")
       return reranked_results[:final_k]

Document Management and Updates
===============================

Smart Document Updates
-----------------------

.. code-block:: python

   def smart_update_document(
       db: LocalVectorDB,
       doc_id: str,
       new_content: str,
       new_metadata: Dict[str, Any] = None,
       force_update: bool = False
   ) -> bool:
       """Update document only if content has actually changed."""

       # Get existing document
       existing_doc = db.get(doc_id)
       if not existing_doc:
           print(f"Document {doc_id} not found")
           return False

       # Check if update is needed
       content_changed = existing_doc.needs_update(new_content)
       metadata_changed = new_metadata and new_metadata != existing_doc.metadata

       if not (content_changed or metadata_changed) and not force_update:
           print(f"Document {doc_id} is already up to date")
           return False

       # Perform update
       success = db.update(
           doc_id=doc_id,
           content=new_content if content_changed else None,
           metadata=new_metadata if metadata_changed else None
       )

       if success:
           change_type = []
           if content_changed:
               change_type.append("content")
           if metadata_changed:
               change_type.append("metadata")

           print(f"Updated {doc_id} ({', '.join(change_type)})")

       return success

Bulk Document Operations
------------------------

.. code-block:: python

   def bulk_update_metadata(
       db: LocalVectorDB,
       metadata_updates: Dict[str, Dict[str, Any]],
       batch_size: int = 50
   ) -> Dict[str, bool]:
       """Update metadata for multiple documents efficiently."""

       results = {}
       doc_ids = list(metadata_updates.keys())

       # Process in batches to avoid memory issues
       for i in range(0, len(doc_ids), batch_size):
           batch_ids = doc_ids[i:i + batch_size]

           # Get existing documents
           existing_docs = db.get(batch_ids)
           if not isinstance(existing_docs, list):
               existing_docs = [existing_docs] if existing_docs else []

           # Update each document
           for doc in existing_docs:
               if doc and doc.id in metadata_updates:
                   new_metadata = {**doc.metadata, **metadata_updates[doc.id]}
                   success = db.update(doc.id, metadata=new_metadata)
                   results[doc.id] = success

           print(f"Processed batch {i//batch_size + 1}/{(len(doc_ids) + batch_size - 1)//batch_size}")

       successful = sum(results.values())
       print(f"Successfully updated {successful}/{len(doc_ids)} documents")

       return results

   def cleanup_duplicate_documents(
       db: LocalVectorDB,
       similarity_threshold: float = 0.95,
       dry_run: bool = True
   ) -> List[str]:
       """Find and optionally remove duplicate documents."""

       # Get all documents
       all_docs = db.filter(limit=1000)  # Adjust limit as needed

       duplicates_to_remove = []
       processed_hashes = set()

       for doc in all_docs:
           if doc.content_hash in processed_hashes:
               duplicates_to_remove.append(doc.id)
               print(f"Found duplicate: {doc.id} (hash: {doc.content_hash[:8]}...)")
           else:
               processed_hashes.add(doc.content_hash)

       if duplicates_to_remove and not dry_run:
           for doc_id in duplicates_to_remove:
               try:
                   db.delete(doc_id)
                   print(f"Removed duplicate: {doc_id}")
               except Exception as e:
                   print(f"Failed to remove {doc_id}: {e}")

       print(f"Found {len(duplicates_to_remove)} duplicates {'(dry run)' if dry_run else '(removed)'}")
       return duplicates_to_remove

Advanced Querying and Analytics
===============================

Document Analytics and Statistics
----------------------------------

.. code-block:: python

   from collections import Counter, defaultdict
   from datetime import datetime, timedelta

   class DocumentAnalytics:
       """Analytics and insights for document collections."""

       def __init__(self, db: LocalVectorDB):
           self.db = db

       def get_overview_stats(self) -> Dict[str, Any]:
           """Get comprehensive database statistics."""

           base_stats = self.db.get_stats()

           # Get sample of documents for analysis
           sample_docs = self.db.filter(limit=500)

           if not sample_docs:
               return base_stats

           # Calculate additional metrics
           content_lengths = [len(doc.content) for doc in sample_docs]

           analytics = {
               **base_stats,
               'content_analysis': {
                   'avg_content_length': sum(content_lengths) / len(content_lengths),
                   'min_content_length': min(content_lengths),
                   'max_content_length': max(content_lengths),
                   'total_characters': sum(content_lengths)
               },
               'metadata_analysis': self._analyze_metadata(sample_docs),
               'recent_activity': self._analyze_recent_activity(sample_docs)
           }

           return analytics

       def _analyze_metadata(self, docs: List) -> Dict[str, Any]:
           """Analyze metadata patterns."""

           metadata_fields = defaultdict(list)
           field_coverage = defaultdict(int)

           for doc in docs:
               for field, value in doc.metadata.items():
                   metadata_fields[field].append(value)
                   field_coverage[field] += 1

           analysis = {}
           for field, values in metadata_fields.items():
               if values:
                   analysis[field] = {
                       'coverage': field_coverage[field] / len(docs),
                       'unique_values': len(set(str(v) for v in values)),
                       'most_common': Counter(str(v) for v in values).most_common(5)
                   }

           return analysis

       def _analyze_recent_activity(self, docs: List) -> Dict[str, Any]:
           """Analyze recent document activity."""

           now = datetime.now()
           recent_counts = {
               'last_24h': 0,
               'last_week': 0,
               'last_month': 0
           }

           for doc in docs:
               if doc.created_at:
                   age = now - doc.created_at
                   if age <= timedelta(days=1):
                       recent_counts['last_24h'] += 1
                   if age <= timedelta(days=7):
                       recent_counts['last_week'] += 1
                   if age <= timedelta(days=30):
                       recent_counts['last_month'] += 1

           return recent_counts

       def find_content_gaps(self, reference_queries: List[str]) -> Dict[str, Any]:
           """Identify potential gaps in content coverage."""

           gaps = {}

           for query in reference_queries:
               results = self.db.query(query, k=5, score_threshold=0.3)

               if len(results) < 3:  # Low coverage
                   gaps[query] = {
                       'result_count': len(results),
                       'max_score': max([r.score for r in results]) if results else 0,
                       'status': 'low_coverage' if results else 'no_coverage'
                   }

           return gaps

Similarity and Clustering Analysis
-----------------------------------

.. code-block:: python

   import numpy as np
   from typing import List, Tuple

   def find_similar_documents(
       db: LocalVectorDB,
       doc_id: str,
       similarity_threshold: float = 0.7,
       max_results: int = 10
   ) -> List[Tuple[str, float]]:
       """Find documents similar to a given document."""

       # Get the reference document
       reference_doc = db.get(doc_id)
       if not reference_doc:
           raise ValueError(f"Document {doc_id} not found")

       # Search for similar content
       results = db.query(
           query=reference_doc.content[:500],  # Use first part as query
           search_type="vector",
           k=max_results + 1,  # +1 to account for self-match
           score_threshold=similarity_threshold
       )

       # Filter out the reference document itself and return similarities
       similar_docs = []
       for result in results:
           if result.id != doc_id:
               similar_docs.append((result.id, result.score))

       return similar_docs[:max_results]

   def cluster_documents_by_similarity(
       db: LocalVectorDB,
       min_cluster_size: int = 3,
       similarity_threshold: float = 0.8
   ) -> Dict[int, List[str]]:
       """Group documents into similarity clusters."""

       # Get all documents (limit for performance)
       all_docs = db.filter(limit=200)

       if len(all_docs) < min_cluster_size:
           return {}

       # Generate embeddings for all documents
       contents = [doc.content[:500] for doc in all_docs]  # Truncate for speed
       embeddings = db.embedding_provider.embed_sync(contents)

       # Calculate similarity matrix
       similarity_matrix = np.dot(embeddings, embeddings.T)

       # Simple clustering based on similarity threshold
       clusters = {}
       cluster_id = 0
       assigned = set()

       for i, doc in enumerate(all_docs):
           if doc.id in assigned:
               continue

           # Find similar documents
           similar_indices = np.where(similarity_matrix[i] >= similarity_threshold)[0]

           if len(similar_indices) >= min_cluster_size:
               cluster_docs = []
               for idx in similar_indices:
                   if all_docs[idx].id not in assigned:
                       cluster_docs.append(all_docs[idx].id)
                       assigned.add(all_docs[idx].id)

               if len(cluster_docs) >= min_cluster_size:
                   clusters[cluster_id] = cluster_docs
                   cluster_id += 1

       return clusters

Performance and Monitoring
===========================

Database Performance Monitoring
--------------------------------

.. code-block:: python

   import time
   import psutil
   import threading
   from typing import Dict, List, Callable

   class PerformanceMonitor:
       """Monitor database performance and resource usage."""

       def __init__(self, db: LocalVectorDB):
           self.db = db
           self.metrics = []
           self.monitoring = False

       def monitor_operation(self, operation_name: str):
           """Decorator to monitor operation performance."""
           def decorator(func: Callable):
               def wrapper(*args, **kwargs):
                   start_time = time.time()
                   start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                   try:
                       result = func(*args, **kwargs)
                       success = True
                       error = None
                   except Exception as e:
                       result = None
                       success = False
                       error = str(e)
                       raise
                   finally:
                       end_time = time.time()
                       end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                       metric = {
                           'operation': operation_name,
                           'duration': end_time - start_time,
                           'memory_used': end_memory - start_memory,
                           'success': success,
                           'error': error,
                           'timestamp': time.time()
                       }

                       self.metrics.append(metric)
                       print(f"{operation_name}: {metric['duration']:.3f}s, "
                             f"Memory: {metric['memory_used']:+.1f}MB")

                   return result
               return wrapper
           return decorator

       def get_performance_summary(self) -> Dict[str, Any]:
           """Get performance summary statistics."""

           if not self.metrics:
               return {"message": "No metrics collected"}

           by_operation = {}
           for metric in self.metrics:
               op = metric['operation']
               if op not in by_operation:
                   by_operation[op] = []
               by_operation[op].append(metric)

           summary = {}
           for op, metrics in by_operation.items():
               durations = [m['duration'] for m in metrics if m['success']]
               if durations:
                   summary[op] = {
                       'count': len(metrics),
                       'success_rate': len(durations) / len(metrics),
                       'avg_duration': sum(durations) / len(durations),
                       'min_duration': min(durations),
                       'max_duration': max(durations)
                   }

           return summary

   # Usage example
   def benchmark_database_operations(db: LocalVectorDB):
       """Benchmark common database operations."""

       monitor = PerformanceMonitor(db)

       # Test search performance
       @monitor.monitor_operation("vector_search")
       def test_vector_search():
           return db.query("test query", search_type="vector", k=10)

       @monitor.monitor_operation("keyword_search")
       def test_keyword_search():
           return db.query("test query", search_type="keyword", k=10)

       @monitor.monitor_operation("document_insert")
       def test_document_insert():
           return db.upsert(["Test document for benchmarking"])

       # Run tests
       print("Running performance benchmarks...")

       for i in range(10):
           test_vector_search()
           test_keyword_search()
           if i < 5:  # Don't insert too many test docs
               test_document_insert()

       # Get results
       summary = monitor.get_performance_summary()
       print("\nPerformance Summary:")
       for operation, stats in summary.items():
           print(f"{operation}:")
           print(f"  Average: {stats['avg_duration']:.3f}s")
           print(f"  Success rate: {stats['success_rate']:.1%}")

Database Health Checks
-----------------------

.. code-block:: python

   def health_check(db: LocalVectorDB) -> Dict[str, Any]:
       """Comprehensive database health check."""

       health_report = {
           'timestamp': datetime.now().isoformat(),
           'database_name': db.name,
           'status': 'unknown',
           'checks': {},
           'recommendations': []
       }

       checks_passed = 0
       total_checks = 5

       # Check 1: Basic connectivity
       try:
           stats = db.get_stats()
           health_report['checks']['connectivity'] = {'status': 'pass', 'details': 'Database accessible'}
           checks_passed += 1
       except Exception as e:
           health_report['checks']['connectivity'] = {'status': 'fail', 'details': str(e)}

       # Check 2: Data integrity
       try:
           sample_docs = db.filter(limit=10)
           if sample_docs:
               health_report['checks']['data_integrity'] = {'status': 'pass', 'details': f'{len(sample_docs)} documents accessible'}
               checks_passed += 1
           else:
               health_report['checks']['data_integrity'] = {'status': 'warning', 'details': 'No documents found'}
       except Exception as e:
           health_report['checks']['data_integrity'] = {'status': 'fail', 'details': str(e)}

       # Check 3: Search functionality
       try:
           if sample_docs:
               results = db.query("test", k=1)
               health_report['checks']['search_functionality'] = {'status': 'pass', 'details': 'Search working'}
               checks_passed += 1
           else:
               health_report['checks']['search_functionality'] = {'status': 'skip', 'details': 'No documents to search'}
       except Exception as e:
           health_report['checks']['search_functionality'] = {'status': 'fail', 'details': str(e)}

       # Check 4: Index health
       try:
           if hasattr(db, 'index') and hasattr(db.index, 'ntotal'):
               index_size = db.index.ntotal
               doc_count = stats.get('documents', 0)
               if index_size > 0:
                   health_report['checks']['index_health'] = {'status': 'pass', 'details': f'{index_size} vectors indexed'}
                   checks_passed += 1
               else:
                   health_report['checks']['index_health'] = {'status': 'warning', 'details': 'Empty index'}
           else:
               health_report['checks']['index_health'] = {'status': 'fail', 'details': 'Index not accessible'}
       except Exception as e:
           health_report['checks']['index_health'] = {'status': 'fail', 'details': str(e)}

       # Check 5: Resource usage
       try:
           import os
           import psutil

           # Check disk space
           if hasattr(db, 'db_path'):
               disk_usage = psutil.disk_usage(os.path.dirname(db.db_path))
               free_space_gb = disk_usage.free / (1024**3)

               if free_space_gb > 1:
                   health_report['checks']['resources'] = {'status': 'pass', 'details': f'{free_space_gb:.1f}GB free space'}
                   checks_passed += 1
               else:
                   health_report['checks']['resources'] = {'status': 'warning', 'details': f'Low disk space: {free_space_gb:.1f}GB'}
           else:
               health_report['checks']['resources'] = {'status': 'skip', 'details': 'Cannot check disk space'}
       except Exception as e:
           health_report['checks']['resources'] = {'status': 'fail', 'details': str(e)}

       # Generate recommendations
       if checks_passed < total_checks:
           health_report['recommendations'].append("Review failed health checks")

       if stats.get('documents', 0) == 0:
           health_report['recommendations'].append("Add documents to the database")

       if stats.get('documents', 0) > 10000:
           health_report['recommendations'].append("Consider performance optimization for large database")

       # Overall status
       if checks_passed == total_checks:
           health_report['status'] = 'healthy'
       elif checks_passed >= total_checks * 0.8:
           health_report['status'] = 'warning'
       else:
           health_report['status'] = 'unhealthy'

       health_report['score'] = f"{checks_passed}/{total_checks}"

       return health_report

Error Handling and Recovery
===========================

Robust Error Handling
---------------------

.. code-block:: python

   import functools
   import logging
   from typing import Optional, Any, Callable

   def with_retry(max_retries: int = 3, backoff_factor: float = 1.0, exceptions: tuple = (Exception,)):
       """Decorator for retrying operations with exponential backoff."""

       def decorator(func: Callable):
           @functools.wraps(func)
           def wrapper(*args, **kwargs):
               last_exception = None

               for attempt in range(max_retries + 1):
                   try:
                       return func(*args, **kwargs)
                   except exceptions as e:
                       last_exception = e

                       if attempt < max_retries:
                           wait_time = backoff_factor * (2 ** attempt)
                           logging.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {e}")
                           time.sleep(wait_time)
                       else:
                           logging.error(f"All {max_retries + 1} attempts failed: {e}")

               raise last_exception

           return wrapper
       return decorator

   class SafeDatabase:
       """Wrapper for LocalVectorDB with enhanced error handling."""

       def __init__(self, db: LocalVectorDB):
           self.db = db
           self.logger = logging.getLogger(__name__)

       @with_retry(max_retries=3)
       def safe_search(self, query: str, **kwargs) -> Optional[List[Any]]:
           """Search with retry logic and error handling."""
           try:
               return self.db.query(query, **kwargs)
           except Exception as e:
               self.logger.error(f"Search failed for query '{query}': {e}")
               return None

       @with_retry(max_retries=2)
       def safe_upsert(self, documents: List[str], **kwargs) -> Optional[List[str]]:
           """Upsert with error handling and partial success tracking."""
           try:
               return self.db.upsert(documents, **kwargs)
           except Exception as e:
               self.logger.error(f"Upsert failed for {len(documents)} documents: {e}")

               # Try individual documents if batch fails
               if len(documents) > 1:
                   self.logger.info("Attempting individual document insertion...")
                   successful_ids = []

                   for i, doc in enumerate(documents):
                       try:
                           doc_ids = self.db.upsert([doc])
                           successful_ids.extend(doc_ids)
                       except Exception as doc_error:
                           self.logger.warning(f"Failed to insert document {i}: {doc_error}")

                   return successful_ids if successful_ids else None

               return None

Database Backup and Recovery
----------------------------

.. code-block:: python

   import shutil
   import zipfile
   from pathlib import Path

   def backup_database(
       db: LocalVectorDB,
       backup_path: str,
       include_config: bool = True
   ) -> str:
       """Create a backup of the database."""

       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       backup_name = f"{db.name}_backup_{timestamp}"
       backup_dir = Path(backup_path) / backup_name
       backup_dir.mkdir(parents=True, exist_ok=True)

       # Save database state
       db.save()

       # Copy database files
       if hasattr(db, 'db_path'):
           shutil.copy2(db.db_path, backup_dir / f"{db.name}.sqlite")

       if hasattr(db, 'index_path') and db.index_path.exists():
           shutil.copy2(db.index_path, backup_dir / f"{db.name}.faiss")

       # Create backup info
       backup_info = {
           'database_name': db.name,
           'backup_timestamp': timestamp,
           'stats': db.get_stats(),
           'embedding_provider': db.embedding_provider.provider_name,
           'embedding_model': db.embedding_provider.model
       }

       with open(backup_dir / 'backup_info.json', 'w') as f:
           json.dump(backup_info, f, indent=2)

       # Create zip archive
       zip_path = Path(backup_path) / f"{backup_name}.zip"
       with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
           for file_path in backup_dir.rglob('*'):
               if file_path.is_file():
                   zipf.write(file_path, file_path.relative_to(backup_dir))

       # Clean up temporary directory
       shutil.rmtree(backup_dir)

       print(f"Database backup created: {zip_path}")
       return str(zip_path)

   def restore_database(
       backup_path: str,
       restore_location: str,
       new_name: Optional[str] = None
   ) -> LocalVectorDB:
       """Restore database from backup."""

       backup_path = Path(backup_path)
       restore_location = Path(restore_location)
       restore_location.mkdir(parents=True, exist_ok=True)

       # Extract backup
       with zipfile.ZipFile(backup_path, 'r') as zipf:
           zipf.extractall(restore_location)

       # Read backup info
       backup_info_path = restore_location / 'backup_info.json'
       with open(backup_info_path, 'r') as f:
           backup_info = json.load(f)

       original_name = backup_info['database_name']
       db_name = new_name or original_name

       # Move files to correct location
       sqlite_file = restore_location / f"{original_name}.sqlite"
       faiss_file = restore_location / f"{original_name}.faiss"

       final_location = restore_location / db_name
       final_location.mkdir(exist_ok=True)

       if sqlite_file.exists():
           shutil.move(sqlite_file, final_location / f"{db_name}.sqlite")

       if faiss_file.exists():
           shutil.move(faiss_file, final_location / f"{db_name}.faiss")

       # Recreate database instance
       restored_db = LocalVectorDB(
           name=db_name,
           base_path=str(final_location.parent),
           create_if_not_exists=False
       )

       print(f"Database restored as '{db_name}' with {restored_db.get_stats()['documents']} documents")
       return restored_db

Conclusion
==========

These recipes provide a solid foundation for working with LocalVectorDB in production environments. Key patterns include:

**Database Management**
   - Factory patterns for consistent database creation
   - Health monitoring and performance tracking
   - Backup and recovery strategies

**Document Processing**
   - Batch processing for large collections
   - Smart content detection and processing
   - Incremental updates and deduplication

**Advanced Search**
   - Multi-strategy search pipelines
   - Semantic re-ranking and clustering
   - Contextual filtering and analytics

**Production Readiness**
   - Comprehensive error handling and retry logic
   - Performance monitoring and optimization
   - Robust backup and disaster recovery

These patterns can be combined and customized for your specific use cases. Remember to always test thoroughly in
development environments before deploying to production!