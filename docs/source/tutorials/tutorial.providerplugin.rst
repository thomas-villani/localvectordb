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
``BaseEmbeddingProvider`` interface, which defines the contract for:

* Model validation and dimension retrieval
* Synchronous and asynchronous embedding generation
* Batch processing capabilities
* Error handling and configuration management

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

First, let's look at the base class we need to implement. Based on the codebase, here's what we need to know:

.. code-block:: python

   from abc import ABC, abstractmethod
   import numpy as np
   from typing import List, Union, Optional, Any, Dict

   class BaseEmbeddingProvider(ABC):
       """
       Abstract base class for embedding providers.
       All custom embedding providers must inherit from this class.
       """

       def __init__(self, model: str, **kwargs):
           self.model = model
           self.provider_name = "custom_provider"
           self.config = kwargs

       @abstractmethod
       def validate_model(self) -> bool:
           """Check if the specified model is available and valid."""
           pass

       @abstractmethod
       def get_dimension(self) -> int:
           """Return the embedding dimension for this model."""
           pass

       @abstractmethod
       def embed_sync(self, texts: Union[str, List[str]]) -> np.ndarray:
           """Generate embeddings synchronously."""
           pass

       async def embed_async(self, texts: Union[str, List[str]]) -> np.ndarray:
           """Generate embeddings asynchronously (optional implementation)."""
           # Default implementation calls sync version
           return self.embed_sync(texts)

Creating the Hugging Face Provider
===================================

Now let's create our custom Hugging Face embedding provider:

.. code-block:: python

   import torch
   import numpy as np
   from sentence_transformers import SentenceTransformer
   from transformers import AutoTokenizer, AutoModel
   from typing import List, Union, Optional, Dict, Any
   import logging

   logger = logging.getLogger(__name__)

   class HuggingFaceEmbeddingProvider:
       """
       Custom embedding provider for Hugging Face models using sentence-transformers
       and transformers libraries.
       """

       def __init__(
           self,
           model: str,
           device: Optional[str] = None,
           max_seq_length: Optional[int] = None,
           batch_size: int = 32,
           normalize_embeddings: bool = True,
           use_sentence_transformers: bool = True,
           trust_remote_code: bool = False,
           **kwargs
       ):
           """
           Initialize the Hugging Face embedding provider.

           Args:
               model: Hugging Face model name or path
               device: Device to run on ('cpu', 'cuda', 'auto')
               max_seq_length: Maximum sequence length (model default if None)
               batch_size: Batch size for processing multiple texts
               normalize_embeddings: Whether to normalize embeddings to unit vectors
               use_sentence_transformers: Use sentence-transformers library if available
               trust_remote_code: Whether to trust remote code (for custom models)
               **kwargs: Additional configuration options
           """
           self.model = model
           self.provider_name = "huggingface"
           self.device = self._setup_device(device)
           self.max_seq_length = max_seq_length
           self.batch_size = batch_size
           self.normalize_embeddings = normalize_embeddings
           self.use_sentence_transformers = use_sentence_transformers
           self.trust_remote_code = trust_remote_code
           self.config = kwargs

           # Initialize the model
           self._model = None
           self._tokenizer = None
           self._dimension = None

           self._load_model()

       def _setup_device(self, device: Optional[str]) -> str:
           """Setup the appropriate device for the model."""
           if device == "auto" or device is None:
               if torch.cuda.is_available():
                   device = "cuda"
               elif torch.backends.mps.is_available():  # Apple Silicon
                   device = "mps"
               else:
                   device = "cpu"

           logger.info(f"Using device: {device}")
           return device

       def _load_model(self):
           """Load the Hugging Face model."""
           try:
               if self.use_sentence_transformers:
                   # Try sentence-transformers first (easier and often better)
                   try:
                       logger.info(f"Loading sentence-transformer model: {self.model}")
                       self._model = SentenceTransformer(
                           self.model,
                           device=self.device,
                           trust_remote_code=self.trust_remote_code
                       )

                       if self.max_seq_length:
                           self._model.max_seq_length = self.max_seq_length

                       # Get dimension from the model
                       self._dimension = self._model.get_sentence_embedding_dimension()
                       self._use_sentence_transformers = True

                       logger.info(f"Successfully loaded sentence-transformer model with dimension {self._dimension}")
                       return

                   except Exception as e:
                       logger.warning(f"Failed to load as sentence-transformer: {e}")
                       logger.info("Falling back to transformers library")

               # Fallback to transformers library
               logger.info(f"Loading transformers model: {self.model}")
               self._tokenizer = AutoTokenizer.from_pretrained(
                   self.model,
                   trust_remote_code=self.trust_remote_code
               )
               self._model = AutoModel.from_pretrained(
                   self.model,
                   trust_remote_code=self.trust_remote_code
               ).to(self.device)

               # Set max sequence length
               if self.max_seq_length:
                   self._tokenizer.model_max_length = self.max_seq_length

               # Get dimension from model config
               self._dimension = self._model.config.hidden_size
               self._use_sentence_transformers = False

               logger.info(f"Successfully loaded transformers model with dimension {self._dimension}")

           except Exception as e:
               raise RuntimeError(f"Failed to load model {self.model}: {str(e)}")

       def validate_model(self) -> bool:
           """Check if the model is properly loaded and functional."""
           try:
               # Test with a simple input
               test_result = self.embed_sync("Test input for validation.")
               return test_result is not None and len(test_result.shape) == 2
           except Exception as e:
               logger.error(f"Model validation failed: {e}")
               return False

       def get_dimension(self) -> int:
           """Return the embedding dimension."""
           if self._dimension is None:
               raise RuntimeError("Model not properly initialized")
           return self._dimension

       def embed_sync(self, texts: Union[str, List[str]]) -> np.ndarray:
           """
           Generate embeddings synchronously.

           Args:
               texts: Single text or list of texts to embed

           Returns:
               numpy array of embeddings with shape (n_texts, embedding_dim)
           """
           # Normalize input
           if isinstance(texts, str):
               texts = [texts]

           if not texts:
               return np.array([]).reshape(0, self._dimension)

           try:
               if self._use_sentence_transformers:
                   return self._embed_with_sentence_transformers(texts)
               else:
                   return self._embed_with_transformers(texts)

           except Exception as e:
               logger.error(f"Error generating embeddings: {e}")
               raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

       def _embed_with_sentence_transformers(self, texts: List[str]) -> np.ndarray:
           """Generate embeddings using sentence-transformers."""
           embeddings = self._model.encode(
               texts,
               batch_size=self.batch_size,
               show_progress_bar=len(texts) > 100,
               convert_to_numpy=True,
               normalize_embeddings=self.normalize_embeddings
           )
           return embeddings

       def _embed_with_transformers(self, texts: List[str]) -> np.ndarray:
           """Generate embeddings using transformers library."""
           all_embeddings = []

           # Process in batches
           for i in range(0, len(texts), self.batch_size):
               batch_texts = texts[i:i + self.batch_size]
               batch_embeddings = self._process_batch(batch_texts)
               all_embeddings.append(batch_embeddings)

           embeddings = np.vstack(all_embeddings)

           if self.normalize_embeddings:
               # Normalize to unit vectors
               norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
               embeddings = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero

           return embeddings

       def _process_batch(self, texts: List[str]) -> np.ndarray:
           """Process a batch of texts with the transformers model."""
           # Tokenize
           encoded = self._tokenizer(
               texts,
               padding=True,
               truncation=True,
               max_length=self.max_seq_length or self._tokenizer.model_max_length,
               return_tensors='pt'
           )

           # Move to device
           encoded = {k: v.to(self.device) for k, v in encoded.items()}

           # Generate embeddings
           with torch.no_grad():
               outputs = self._model(**encoded)

               # Use mean pooling of last hidden states
               embeddings = self._mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])

           return embeddings.cpu().numpy()

       @staticmethod
       def _mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
           """Apply mean pooling to get sentence embeddings."""
           # Expand attention mask to match hidden states dimensions
           expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

           # Apply mask and sum
           sum_embeddings = torch.sum(hidden_states * expanded_mask, dim=1)
           sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)

           return sum_embeddings / sum_mask

       async def embed_async(self, texts: Union[str, List[str]]) -> np.ndarray:
           """
           Generate embeddings asynchronously.

           Note: This is a simple implementation that runs sync code in a thread.
           For true async support, you'd want to use asyncio-compatible libraries.
           """
           import asyncio
           import concurrent.futures

           loop = asyncio.get_event_loop()
           with concurrent.futures.ThreadPoolExecutor() as executor:
               result = await loop.run_in_executor(executor, self.embed_sync, texts)

           return result

Registering the Plugin
======================

Now we need to register our custom provider with LocalVectorDB's embedding registry:

.. code-block:: python

   from localvectordb.embeddings import EmbeddingRegistry

   def register_huggingface_provider():
       """Register the Hugging Face provider with LocalVectorDB."""

       def create_huggingface_provider(model: str, **kwargs):
           """Factory function to create Hugging Face provider instances."""
           return HuggingFaceEmbeddingProvider(model=model, **kwargs)

       # Register the provider
       EmbeddingRegistry.register_provider("huggingface", create_huggingface_provider)

       logger.info("Hugging Face embedding provider registered successfully")

   # Register the provider when this module is imported
   register_huggingface_provider()

Using the Custom Provider
=========================

Now let's see how to use our custom Hugging Face provider with LocalVectorDB:

.. code-block:: python

   from localvectordb import LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType

   # Import our custom provider (this also registers it)
   from your_module import HuggingFaceEmbeddingProvider, register_huggingface_provider

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
           embedding_provider="huggingface",
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
           embedding_provider="huggingface",
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
           embedding_provider="huggingface",
           embedding_model="allenai/scibert_scivocab_uncased",  # Scientific domain model
           embedding_config={
               "device": "auto",
               "batch_size": 16,  # Smaller batch for larger model
               "normalize_embeddings": True,
               "max_seq_length": 512,
               "use_sentence_transformers": False  # Use transformers directly
           }
       )

       return db

   def create_legal_db():
       """Create a database optimized for legal texts."""

       db = LocalVectorDB(
           name="legal_kb",
           base_path="./legal_storage",
           embedding_provider="huggingface",
           embedding_model="nlpaueb/legal-bert-base-uncased",
           embedding_config={
               "device": "auto",
               "batch_size": 16,
               "normalize_embeddings": True,
               "max_seq_length": 512,
               "use_sentence_transformers": False
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
           embedding_provider="huggingface",
           embedding_model="microsoft/codebert-base",
           embedding_config={
               "device": "auto",
               "batch_size": 32,
               "normalize_embeddings": True,
               "max_seq_length": 512,
               "use_sentence_transformers": False
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
           embedding_provider="huggingface",
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
           embedding = self.provider.embed_sync(text)

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
           different_embedding = self.provider.embed_sync(different_text)

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

   # Import our custom provider
   from huggingface_provider import HuggingFaceEmbeddingProvider, register_huggingface_provider
   from localvectordb import LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType

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
           embedding_provider="huggingface",
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

               # Test embedding generation
               import time
               start_time = time.time()
               embedding = provider.embed_sync(test_text)
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

   # Enhanced provider with monitoring
   class ProductionHuggingFaceProvider(HuggingFaceEmbeddingProvider):
       """Production-ready version with enhanced error handling."""

       @with_retry_and_monitoring(max_retries=3, backoff_factor=1.0)
       def embed_sync(self, texts):
           return super().embed_sync(texts)

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