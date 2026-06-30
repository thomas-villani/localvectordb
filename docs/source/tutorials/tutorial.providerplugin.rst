==========================================================
Custom Embedding Plugin Tutorial: Hugging Face Integration
==========================================================

This tutorial will guide you through creating a custom embedding plugin for LocalVectorDB using Hugging Face Transformers. You'll learn how to integrate any Hugging Face embedding model into the LocalVectorDB ecosystem, giving you access to thousands of pre-trained models.

Why Custom Embedding Plugins?
==============================

While LocalVectorDB comes with built-in support for Ollama and OpenAI embeddings, you might want to use other models for various reasons:

* **Specialized Models**: Domain-specific models trained for legal, medical, or scientific texts
* **Performance**: Models optimized for speed or accuracy for your specific use case
* **Privacy**: Running completely offline with Hugging Face models
* **Cost**: Using free models instead of API-based services
* **Experimentation**: Testing different embedding strategies and models

Understanding the Plugin Architecture
======================================

LocalVectorDB uses a plugin-based architecture for embedding providers. All embedding plugins must implement the
``EmbeddingProvider`` interface, which defines the contract for:

* Model validation and dimension retrieval
* Synchronous and asynchronous embedding generation
* Batch processing capabilities
* Error handling and configuration management

.. note::

   **You may not need a custom plugin at all.** LocalVectorDB already ships providers that
   run Hugging Face models locally:

   * ``sentence_transformers`` -- runs any `sentence-transformers <https://www.sbert.net/>`_
     model locally (install the ``sentence-transformers`` extra).
   * ``huggingface_local`` -- runs a local Hugging Face Transformers model.
   * ``huggingface`` -- calls the hosted Hugging Face Inference API.

   So for most cases you can just pass ``embedding_provider="sentence_transformers"``. This
   tutorial builds a custom provider anyway, because it's the clearest way to learn the
   plugin contract and the pattern you'd follow for a provider that isn't built in (a private
   model server, a new vendor API, a bespoke pooling strategy, and so on).

Prerequisites
=============

Before starting, ensure you have the required dependencies:

.. code-block:: bash

   pip install localvectordb torch transformers sentence-transformers

For GPU support (recommended for better performance):

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Examining the Base Provider Interface
=====================================

Let's look at the real ``localvectordb.embeddings.EmbeddingProvider`` base class so we
implement the right contract. The important thing to understand is that **the base class
already implements the public, batching, retry, and sync/async machinery for you**
(``embed_batch``, ``embed_async``, and ``embed_sync``). Those methods are concrete -- do
*not* reimplement them. Instead, you implement one async worker method plus a few pieces of
metadata. Here is the shape of the contract (abbreviated):

.. code-block:: python

   from abc import ABC, abstractmethod
   from typing import Any, List, Optional
   import numpy as np

   class EmbeddingProvider(ABC):
       """Abstract base class for embedding providers (abbreviated)."""

       def __init__(
           self,
           model: str,
           *,
           timeout: int = 90,
           max_retries: int = 3,
           retry_delay: float = 1.0,
           max_concurrent_requests: int = 5,
           **kwargs: Any,
       ) -> None:
           self.model = model
           self.config = kwargs
           # ... stores timeout/retry/concurrency settings ...

       # --- The ONE method you must implement: it embeds a single batch ---
       @abstractmethod
       async def _embed_single_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
           """Return one embedding (a list of floats) for each text in ``texts``."""

       # --- Metadata the database needs -----------------------------------
       @abstractmethod
       def get_dimension(self) -> int:
           """Return the embedding dimension for this model."""

       @abstractmethod
       def validate_model(self) -> bool:
           """Check that the model is available and functional."""

       @property
       @abstractmethod
       def provider_name(self) -> str:
           """Short, unique provider name."""

       @property
       @abstractmethod
       def max_batch_size(self) -> int:
           """Largest batch the base class should send to ``_embed_single_batch``."""

       # --- Provided FOR you by the base class (do NOT reimplement) --------
       # async def embed_batch(self, texts: List[str], batch_size=None) -> np.ndarray
       # async def embed_async(self, texts: List[str], batch_size=None) -> np.ndarray
       # def embed_sync(self, texts: List[str], batch_size=None) -> np.ndarray

.. important::

   The public methods take a **list of strings**. The database always calls them that way
   (e.g. ``embed_sync([query])`` and ``embed_sync(batch_texts, batch_size)``). When you call
   ``embed_sync`` yourself, always pass a list -- never a bare string.

Creating the Hugging Face Provider
===================================

Now let's create our custom Hugging Face embedding provider:

.. code-block:: python

   import asyncio
   import logging
   from typing import Any, List, Optional

   from sentence_transformers import SentenceTransformer

   from localvectordb.embeddings import EmbeddingProvider

   logger = logging.getLogger(__name__)

   class HuggingFaceEmbeddingProvider(EmbeddingProvider):
       """
       Custom embedding provider backed by sentence-transformers.

       We inherit from ``EmbeddingProvider`` and implement only the abstract members.
       The base class supplies ``embed_batch`` / ``embed_async`` / ``embed_sync`` (with
       batching, concurrency, and retries) and calls our ``_embed_single_batch`` to do
       the actual work.
       """

       def __init__(
           self,
           model: str,
           *,
           device: Optional[str] = None,
           batch_size: int = 32,
           normalize_embeddings: bool = True,
           trust_remote_code: bool = False,
           **kwargs: Any,
       ) -> None:
           """
           Initialize the Hugging Face embedding provider.

           Args:
               model: Hugging Face / sentence-transformers model name or path
               device: Device to run on ('cpu', 'cuda', 'mps', or 'auto'/None to auto-detect)
               batch_size: Batch size for the underlying ``encode`` call
               normalize_embeddings: Whether to normalize embeddings to unit vectors
               trust_remote_code: Whether to trust remote code (for custom models)
               **kwargs: Forwarded to EmbeddingProvider (timeout, max_retries, etc.)
           """
           # Forward retry/timeout/concurrency settings to the base class.
           super().__init__(model, **kwargs)

           self.device = self._resolve_device(device)
           self._batch_size = batch_size
           self.normalize_embeddings = normalize_embeddings

           # Load the model once, up front, and cache its dimension.
           logger.info(f"Loading sentence-transformer model {model!r} on {self.device}")
           self._model = SentenceTransformer(
               model, device=self.device, trust_remote_code=trust_remote_code
           )
           self._dimension = self._model.get_sentence_embedding_dimension()

       @staticmethod
       def _resolve_device(device: Optional[str]) -> str:
           """Pick a device, auto-detecting CUDA / Apple Silicon when not specified."""
           if device and device != "auto":
               return device
           try:
               import torch

               if torch.cuda.is_available():
                   return "cuda"
               if torch.backends.mps.is_available():  # Apple Silicon
                   return "mps"
           except Exception:
               pass
           return "cpu"

       # --- The one method the base class calls to do real work ------------
       async def _embed_single_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
           """Embed one batch and return a list of vectors (one list of floats per text)."""
           # ``encode`` is blocking (CPU/GPU bound), so run it off the event loop.
           embeddings = await asyncio.to_thread(
               self._model.encode,
               texts,
               batch_size=self._batch_size,
               convert_to_numpy=True,
               normalize_embeddings=self.normalize_embeddings,
           )
           return embeddings.tolist()

       # --- Metadata the database needs ------------------------------------
       def get_dimension(self) -> int:
           """Return the embedding dimension."""
           return self._dimension

       def validate_model(self) -> bool:
           """Check the model is loaded and produces embeddings of the right shape."""
           try:
               # embed_sync is provided by the base class; always pass a LIST.
               result = self.embed_sync(["validation text"])
               return result.shape == (1, self._dimension)
           except Exception as e:
               logger.error(f"Model validation failed: {e}")
               return False

       @property
       def provider_name(self) -> str:
           # Must NOT collide with a built-in provider name.
           return "hf_custom"

       @property
       def max_batch_size(self) -> int:
           return self._batch_size

Registering the Plugin
======================

The recommended, packaging-native way to register a provider is a **Python entry point**
in your project's ``pyproject.toml`` under the ``localvectordb.embedding_providers`` group.
LocalVectorDB auto-discovers entry points in this group, so once your package is installed
the provider name works everywhere -- no imports or manual registration needed.

.. code-block:: toml

   # pyproject.toml of YOUR plugin package
   [project.entry-points."localvectordb.embedding_providers"]
   hf_custom = "your_package.huggingface_provider:HuggingFaceEmbeddingProvider"

The key (``hf_custom``) becomes the ``embedding_provider`` value; the value is
``module:Class``. After installing your package (e.g. ``pip install -e .``) you can use it:

.. code-block:: python

   from localvectordb import LocalVectorDB

   db = LocalVectorDB(
       name="hf_demo",
       embedding_provider="hf_custom",
       embedding_model="sentence-transformers/all-MiniLM-L6-v2",
   )

For quick, in-process experiments (e.g. in a notebook) where you don't want to package
anything yet, register the class at runtime instead. Pass the **class** (the base class's
factory instantiates it as ``ProviderClass(model, **embedding_config)``):

.. code-block:: python

   from localvectordb.embeddings import EmbeddingRegistry

   EmbeddingRegistry.register("hf_custom", HuggingFaceEmbeddingProvider)

.. warning::

   Choose a name that does **not** collide with a built-in provider. LocalVectorDB already
   ships ``sentence_transformers``, ``huggingface_local``, and ``huggingface`` -- registering
   one of those names would silently override the built-in. We use ``hf_custom`` here.

Using the Custom Provider
=========================

Now let's see how to use our custom Hugging Face provider with LocalVectorDB:

.. code-block:: python

   from localvectordb import LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType
   from localvectordb.embeddings import EmbeddingRegistry

   # Import our custom provider and register it for in-process use.
   # (If you installed it via a pyproject.toml entry point, skip the register call.)
   from your_package.huggingface_provider import HuggingFaceEmbeddingProvider
   EmbeddingRegistry.register("hf_custom", HuggingFaceEmbeddingProvider)

   def create_db_with_huggingface_embeddings():
       """Create a LocalVectorDB instance using Hugging Face embeddings."""

       # Define metadata schema
       metadata_schema = {
           'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'domain': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'language': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
       }

       # Create database with Hugging Face embeddings
       db = LocalVectorDB(
           name="hf_knowledge_base",
           base_path="./hf_vector_storage",
           metadata_schema=metadata_schema,
           embedding_provider="hf_custom",
           embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Fast and good quality
           embedding_config={
               "device": "auto",  # Use GPU if available
               "batch_size": 64,
               "normalize_embeddings": True,
               "max_seq_length": 512
           }
       )

       print(f"Created database with {db.embedding_dimension}D Hugging Face embeddings")
       return db

Advanced Configuration Examples
===============================

Here are examples of using different types of Hugging Face models:

Multilingual Models
-------------------

.. code-block:: python

   def create_multilingual_db():
       """Create a database with multilingual embeddings."""

       db = LocalVectorDB(
           name="multilingual_kb",
           base_path="./multilingual_storage",
           embedding_provider="hf_custom",
           embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
           embedding_config={
               "device": "auto",
               "batch_size": 32,
               "normalize_embeddings": True,
               "max_seq_length": 512
           }
       )

       # Test with multiple languages
       test_docs = [
           "Hello, this is an English document.",
           "Bonjour, ceci est un document français.",
           "Hola, este es un documento en español.",
           "Hallo, dies ist ein deutsches Dokument."
       ]

       doc_ids = db.upsert(
           documents=test_docs,
           metadata=[
               {"language": "en", "title": "English Test"},
               {"language": "fr", "title": "French Test"},
               {"language": "es", "title": "Spanish Test"},
               {"language": "de", "title": "German Test"}
           ]
       )

       return db

Domain-Specific Models
----------------------

.. code-block:: python

   def create_scientific_db():
       """Create a database optimized for scientific texts."""

       db = LocalVectorDB(
           name="scientific_kb",
           base_path="./scientific_storage",
           embedding_provider="hf_custom",
           embedding_model="allenai/scibert_scivocab_uncased",  # Scientific domain model
           embedding_config={
               "device": "auto",
               "batch_size": 16,  # Smaller batch for larger model
               "normalize_embeddings": True,
               "max_seq_length": 512
           }
       )

       return db

   def create_legal_db():
       """Create a database optimized for legal texts."""

       db = LocalVectorDB(
           name="legal_kb",
           base_path="./legal_storage",
           embedding_provider="hf_custom",
           embedding_model="nlpaueb/legal-bert-base-uncased",
           embedding_config={
               "device": "auto",
               "batch_size": 16,
               "normalize_embeddings": True,
               "max_seq_length": 512
           }
       )

       return db

Code-Specific Models
--------------------

.. code-block:: python

   def create_code_db():
       """Create a database optimized for code and programming content."""

       db = LocalVectorDB(
           name="code_kb",
           base_path="./code_storage",
           embedding_provider="hf_custom",
           embedding_model="microsoft/codebert-base",
           embedding_config={
               "device": "auto",
               "batch_size": 32,
               "normalize_embeddings": True,
               "max_seq_length": 512
           },
           chunking_method="code-blocks",  # Use code-aware chunking
           chunk_size=800  # Larger chunks for code
       )

       return db

Performance Optimization
========================

Here are some tips for optimizing performance with Hugging Face models:

Model Selection Guide
---------------------

.. code-block:: python

   # Performance vs Quality tradeoffs:

   EMBEDDING_MODELS = {
       # Fast and lightweight (good for development/testing)
       "fast": {
           "model": "sentence-transformers/all-MiniLM-L6-v2",
           "dimension": 384,
           "speed": "very_fast",
           "quality": "good"
       },

       # Balanced performance and quality
       "balanced": {
           "model": "sentence-transformers/all-mpnet-base-v2",
           "dimension": 768,
           "speed": "medium",
           "quality": "very_good"
       },

       # High quality (slower but better results)
       "quality": {
           "model": "sentence-transformers/all-roberta-large-v1",
           "dimension": 1024,
           "speed": "slow",
           "quality": "excellent"
       },

       # Multilingual
       "multilingual": {
           "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
           "dimension": 768,
           "speed": "medium",
           "quality": "very_good"
       }
   }

   def create_optimized_db(model_type: str = "balanced"):
       """Create database with optimized model selection."""

       model_config = EMBEDDING_MODELS[model_type]

       # Adjust batch size based on model size
       if model_config["dimension"] <= 384:
           batch_size = 128
       elif model_config["dimension"] <= 768:
           batch_size = 64
       else:
           batch_size = 32

       db = LocalVectorDB(
           name=f"optimized_kb_{model_type}",
           base_path=f"./storage_{model_type}",
           embedding_provider="hf_custom",
           embedding_model=model_config["model"],
           embedding_config={
               "device": "auto",
               "batch_size": batch_size,
               "normalize_embeddings": True,
               "max_seq_length": 512
           }
       )

       return db

Caching and Model Management
----------------------------

.. code-block:: python

   import os
   from pathlib import Path

   def setup_model_caching():
       """Setup efficient model caching for Hugging Face models."""

       # Set cache directory for Hugging Face models
       cache_dir = Path("./model_cache")
       cache_dir.mkdir(exist_ok=True)

       os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
       os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir)

       print(f"Model cache directory set to: {cache_dir}")

   def preload_models(model_names: List[str]):
       """Preload models to avoid delays during first use."""

       setup_model_caching()

       for model_name in model_names:
           try:
               print(f"Preloading model: {model_name}")
               provider = HuggingFaceEmbeddingProvider(model=model_name)
               if provider.validate_model():
                   print(f"✅ {model_name} loaded successfully")
               else:
                   print(f"❌ {model_name} validation failed")
           except Exception as e:
               print(f"❌ Failed to load {model_name}: {e}")

Testing the Custom Provider
============================

Let's create comprehensive tests for our custom provider:

.. code-block:: python

   import unittest
   import numpy as np
   from typing import List

   class TestHuggingFaceProvider(unittest.TestCase):
       """Test suite for the Hugging Face embedding provider."""

       def setUp(self):
           """Set up test fixtures."""
           self.provider = HuggingFaceEmbeddingProvider(
               model="sentence-transformers/all-MiniLM-L6-v2",
               device="cpu"  # Use CPU for consistent testing
           )

       def test_model_validation(self):
           """Test model validation."""
           self.assertTrue(self.provider.validate_model())

       def test_dimension_consistency(self):
           """Test that dimension is consistent."""
           expected_dim = 384  # Known dimension for all-MiniLM-L6-v2
           self.assertEqual(self.provider.get_dimension(), expected_dim)

       def test_single_text_embedding(self):
           """Test embedding a single text."""
           text = "This is a test sentence."
           # embed_sync always takes a LIST of texts, even for a single one.
           embedding = self.provider.embed_sync([text])

           self.assertEqual(embedding.shape, (1, self.provider.get_dimension()))
           self.assertTrue(np.isfinite(embedding).all())

       def test_batch_embedding(self):
           """Test embedding multiple texts."""
           texts = [
               "First test sentence.",
               "Second test sentence.",
               "Third test sentence."
           ]
           embeddings = self.provider.embed_sync(texts)

           self.assertEqual(embeddings.shape, (len(texts), self.provider.get_dimension()))
           self.assertTrue(np.isfinite(embeddings).all())

       def test_empty_input(self):
           """Test handling of empty input."""
           embedding = self.provider.embed_sync([])
           self.assertEqual(embedding.shape, (0, self.provider.get_dimension()))

       def test_embedding_similarity(self):
           """Test that similar texts have similar embeddings."""
           similar_texts = [
               "The cat sat on the mat.",
               "A cat was sitting on the mat."
           ]
           different_text = "Quantum physics is fascinating."

           similar_embeddings = self.provider.embed_sync(similar_texts)
           different_embedding = self.provider.embed_sync([different_text])

           # Calculate cosine similarity
           sim_similarity = np.dot(similar_embeddings[0], similar_embeddings[1])
           diff_similarity = np.dot(similar_embeddings[0], different_embedding[0])

           # Similar texts should be more similar than different texts
           self.assertGreater(sim_similarity, diff_similarity)

   def run_performance_benchmark():
       """Run performance benchmarks for the provider."""

       print("Running performance benchmarks...")

       provider = HuggingFaceEmbeddingProvider(
           model="sentence-transformers/all-MiniLM-L6-v2",
           device="auto"
       )

       # Test different batch sizes
       test_texts = ["This is test sentence number {}.".format(i) for i in range(1000)]

       import time

       for batch_size in [1, 10, 50, 100]:
           start_time = time.time()

           for i in range(0, len(test_texts), batch_size):
               batch = test_texts[i:i + batch_size]
               provider.embed_sync(batch)

           duration = time.time() - start_time
           texts_per_second = len(test_texts) / duration

           print(f"Batch size {batch_size:3d}: {texts_per_second:.2f} texts/second")

Complete Example Application
============================

Here's a complete example that puts everything together:

.. code-block:: python

   #!/usr/bin/env python3
   """
   LocalVectorDB Custom Embedding Provider Example

   Demonstrates how to create and use a custom Hugging Face embedding provider.
   """

   import logging
   from pathlib import Path
   from typing import List, Dict, Any

   # Configure logging
   logging.basicConfig(level=logging.INFO)

   # Import our custom provider and register it for in-process use.
   # (If you installed it via a pyproject.toml entry point, skip the register call.)
   from huggingface_provider import HuggingFaceEmbeddingProvider
   from localvectordb import LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType
   from localvectordb.embeddings import EmbeddingRegistry

   EmbeddingRegistry.register("hf_custom", HuggingFaceEmbeddingProvider)

   def main():
       """Main demonstration function."""

       print("🤖 LocalVectorDB Custom Embedding Provider Demo")
       print("=" * 55)

       # Setup model caching
       setup_model_caching()

       # Create database with custom provider
       print("📚 Creating database with Hugging Face embeddings...")

       db = LocalVectorDB(
           name="custom_embedding_demo",
           base_path="./custom_demo_storage",
           metadata_schema={
               'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
               'language': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
           },
           embedding_provider="hf_custom",
           embedding_model="sentence-transformers/all-MiniLM-L6-v2",
           embedding_config={
               "device": "auto",
               "batch_size": 64,
               "normalize_embeddings": True
           }
       )

       print(f"✅ Database created with {db.embedding_dimension}D embeddings")

       # Add sample documents
       sample_docs = [
           {
               'content': 'Python is a versatile programming language used for web development, data science, and automation.',
               'title': 'Python Programming',
               'category': 'programming',
               'language': 'en'
           },
           {
               'content': 'Machine learning algorithms can automatically improve through experience and data.',
               'title': 'ML Fundamentals',
               'category': 'ai',
               'language': 'en'
           },
           {
               'content': 'Vector databases enable semantic search by storing high-dimensional embeddings.',
               'title': 'Vector Databases',
               'category': 'database',
               'language': 'en'
           }
       ]

       # Insert documents
       texts = [doc['content'] for doc in sample_docs]
       metadata = [{k: v for k, v in doc.items() if k != 'content'} for doc in sample_docs]

       doc_ids = db.upsert(documents=texts, metadata=metadata)
       print(f"📝 Inserted {len(doc_ids)} documents")

       # Test search functionality
       print("\n🔍 Testing search functionality...")

       query = "programming languages and development"
       results = db.query(query, search_type="vector", k=3)

       print(f"Query: '{query}'")
       print("Results:")
       for i, result in enumerate(results, 1):
           print(f"  {i}. {result.metadata['title']} (score: {result.score:.3f})")
           print(f"     Category: {result.metadata['category']}")
           print(f"     Content: {result.content[:100]}...")
           print()

       # Test different models
       print("🧪 Testing different model configurations...")
       test_different_models()

       # Cleanup
       db.close()
       print("✅ Demo completed successfully!")

   def test_different_models():
       """Test different Hugging Face models."""

       models_to_test = [
           ("sentence-transformers/all-MiniLM-L6-v2", "Fast & Lightweight"),
           ("sentence-transformers/all-mpnet-base-v2", "Balanced Quality"),
           ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "Multilingual")
       ]

       test_text = "This is a test sentence for embedding generation."

       for model_name, description in models_to_test:
           try:
               print(f"Testing {description}: {model_name}")
               provider = HuggingFaceEmbeddingProvider(
                   model=model_name,
                   device="cpu",  # Use CPU for consistent timing
                   batch_size=32
               )

               # Test embedding generation (embed_sync always takes a list)
               import time
               start_time = time.time()
               embedding = provider.embed_sync([test_text])
               duration = time.time() - start_time

               print(f"  ✅ Dimension: {embedding.shape[1]}, Time: {duration:.3f}s")

           except Exception as e:
               print(f"  ❌ Failed: {e}")

   if __name__ == "__main__":
       main()

Deployment Considerations
=========================

When deploying your custom embedding provider in production:

Model Size and Memory
---------------------

.. code-block:: python

   def estimate_memory_usage(model_name: str):
       """Estimate memory usage for a Hugging Face model."""

       try:
           from transformers import AutoConfig
           config = AutoConfig.from_pretrained(model_name)

           # Rough estimation based on parameters
           num_parameters = getattr(config, 'num_parameters', None)
           if num_parameters is None:
               # Estimate from hidden size and layers
               hidden_size = getattr(config, 'hidden_size', 768)
               num_layers = getattr(config, 'num_hidden_layers', 12)
               vocab_size = getattr(config, 'vocab_size', 30000)

               # Rough parameter count estimation
               num_parameters = (hidden_size * hidden_size * num_layers * 4) + (vocab_size * hidden_size)

           # Each parameter is typically 4 bytes (float32)
           memory_gb = (num_parameters * 4) / (1024**3)

           print(f"Model: {model_name}")
           print(f"Estimated parameters: {num_parameters:,}")
           print(f"Estimated memory: {memory_gb:.2f} GB")

           return memory_gb

       except Exception as e:
           print(f"Could not estimate memory for {model_name}: {e}")
           return None

Error Handling and Monitoring
------------------------------

.. code-block:: python

   import functools
   import time
   from typing import Callable

   def with_retry_and_monitoring(max_retries: int = 3, backoff_factor: float = 1.0):
       """Decorator for robust error handling and monitoring."""

       def decorator(func: Callable):
           @functools.wraps(func)
           def wrapper(*args, **kwargs):
               last_exception = None

               for attempt in range(max_retries + 1):
                   try:
                       start_time = time.time()
                       result = func(*args, **kwargs)
                       duration = time.time() - start_time

                       # Log successful execution
                       logger.info(f"{func.__name__} completed in {duration:.3f}s (attempt {attempt + 1})")
                       return result

                   except Exception as e:
                       last_exception = e

                       if attempt < max_retries:
                           wait_time = backoff_factor * (2 ** attempt)
                           logger.warning(f"{func.__name__} failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}")
                           time.sleep(wait_time)
                       else:
                           logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")

               raise last_exception

           return wrapper
       return decorator

   # Enhanced provider with monitoring.
   # NOTE: the base class already retries failed batches (see ``max_retries``); this
   # subclass mainly adds timing/monitoring. ``embed_sync`` keeps the base signature
   # so the database can still call it as ``embed_sync(texts, batch_size)``.
   class ProductionHuggingFaceProvider(HuggingFaceEmbeddingProvider):
       """Production-ready version with enhanced error handling."""

       @with_retry_and_monitoring(max_retries=3, backoff_factor=1.0)
       def embed_sync(self, texts, batch_size=None):
           return super().embed_sync(texts, batch_size)

Conclusion
==========

You've successfully created a custom Hugging Face embedding provider for LocalVectorDB! This tutorial covered:

**Key Concepts**
- Understanding the embedding provider interface
- Implementing custom providers with proper error handling
- Supporting both sentence-transformers and transformers libraries

**Advanced Features**
- Multi-model support and configuration
- Performance optimization and benchmarking
- Async support and batch processing
- Memory management and caching

**Production Readiness**
- Comprehensive testing and validation
- Error handling and retry logic
- Performance monitoring and optimization
- Deployment considerations

**Next Steps**
- Experiment with specialized models for your domain
- Implement additional providers (e.g., OpenAI, Cohere, local models)
- Add support for multimodal embeddings (text + images)
- Contribute your provider back to the LocalVectorDB community

The plugin architecture makes LocalVectorDB highly extensible, allowing you to integrate any embedding model or service that fits your specific needs. Happy embedding! 🚀