Embeddings
==========

LocalVectorDB features a plugin-based embedding system that supports multiple providers with a unified interface. The system is designed for flexibility, allowing easy switching between providers and custom implementations.

Overview
--------

**Embeddings** are dense vector representations of text that capture semantic meaning. LocalVectorDB supports multiple embedding providers:

- **Ollama**: Local embeddings without API costs
- **OpenAI**: Cloud-based embeddings with high quality
- **JinaAI**: Advanced cloud-based embedding models with more control
- **Google**: Cloud-based Gemini Embedding
- **SentenceTransformers**: Local inference with the sentence-transformers library
- **HuggingFace**: Both Inference API and local transformers models
- **Custom Providers**: Plugin system for additional providers

Embedding Providers
-------------------

Ollama Provider (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run embeddings locally without API costs or rate limits.

Setup:

.. code-block:: bash

   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull embedding models
   ollama pull nomic-embed-text        # 137M parameters, good quality
   ollama pull mxbai-embed-large       # 334M parameters, highest quality
   ollama pull all-minilm              # 23M parameters, fastest

Configuration:

.. code-block:: python

   from localvectordb import VectorDB

   # Default Ollama configuration
   db = VectorDB(
       "my_db",
       embedding_provider="ollama",
       embedding_model="nomic-embed-text",
       embedding_config={
           "base_url": "http://localhost:11434"  # Default Ollama URL
       }
   )

   # Custom Ollama configuration
   db = VectorDB(
       "my_db",
       embedding_provider="ollama",
       embedding_model="mxbai-embed-large",
       embedding_config={
           "base_url": "http://remote-ollama:11434",  # Remote Ollama
           "timeout": 60  # Request timeout in seconds
       }
   )

Available Models:

- ``nomic-embed-text``: General-purpose, good balance of speed/quality
- ``mxbai-embed-large``: Highest quality, slower
- ``all-minilm``: Fastest, lower quality
- ``snowflake-arctic-embed``: Optimized for retrieval tasks

OpenAI Provider
^^^^^^^^^^^^^^^

High-quality cloud embeddings with API costs.

Setup:

.. code-block:: bash

   export OPENAI_API_KEY=your_api_key_here

Configuration:

.. code-block:: python

   # Using environment variable
   db = VectorDB(
       "my_db",
       embedding_provider="openai",
       embedding_model="text-embedding-3-small"
   )

   # Explicit API key
   db = VectorDB(
       "my_db",
       embedding_provider="openai",
       embedding_model="text-embedding-3-large",
       embedding_config={
           "api_key": "your_api_key_here"
       }
   )

Available Models:

- ``text-embedding-3-small``: 1536 dimensions, cost-effective
- ``text-embedding-3-large``: 3072 dimensions, highest quality
- ``text-embedding-ada-002``: Legacy model, still good quality

JinaAI Provider
^^^^^^^^^^^^^^^

Advanced cloud-based embedding models with extensive customization options.

.. note::
   The JinaAI provider is built into LocalVectorDB and requires no additional dependencies.
   It uses the standard HTTP client already included with LocalVectorDB.

Setup:

.. code-block:: bash

   # No additional installation required - JinaAI provider is built-in
   export JINA_API_KEY=your_api_key_here
   # Get your free API key at: https://jina.ai/?sui=apikey

Configuration:

.. code-block:: python

   # Basic configuration
   db = VectorDB(
       "my_db",
       embedding_provider="jina",
       embedding_model="jina-embeddings-v4"
   )

   # Advanced configuration with task-specific optimization
   db = VectorDB(
       "my_db",
       embedding_provider="jina",
       embedding_model="jina-embeddings-v4",
       embedding_config={
           "api_key": "your_api_key_here",
           "task": "retrieval.passage",  # Optimize for document retrieval
           "requested_dimensions": 1024,  # Truncate to 1024 dimensions
           "truncate": True,
           "late_chunking": True
       }
   )

   # Code embeddings
   db = VectorDB(
       "my_db",
       embedding_provider="jina",
       embedding_model="jina-code-embeddings-1.5b",
       embedding_config={
           "task": "code2code.passage"  # Code-to-code similarity
       }
   )

Available Models:

- ``jina-embeddings-v4``: 2048 dimensions, multimodal/multilingual
- ``jina-embeddings-v3``: 1024 dimensions, text-only
- ``jina-code-embeddings-1.5b``: 1536 dimensions, code-specialized
- ``jina-code-embeddings-0.5b``: 896 dimensions, code-specialized

Task Types for jina-embeddings-v4:

- ``retrieval.query``: For search queries
- ``retrieval.passage``: For documents being searched
- ``text-matching``: For similarity comparisons
- ``code.query`` / ``code.passage``: For code search

Task Types for code models:

- ``nl2code.query`` / ``nl2code.passage``: Natural language to code
- ``code2code.query`` / ``code2code.passage``: Code-to-code search
- ``code2nl.query`` / ``code2nl.passage``: Code to natural language
- ``code2completion.query`` / ``code2completion.passage``: Code completion
- ``qa.query`` / ``qa.passage``: Question-answering

Google AI Provider
^^^^^^^^^^^^^^^^^^

Google's Gemini embedding models with flexible configuration.

.. note::
   The Google AI provider is built into LocalVectorDB and requires no additional dependencies.
   It uses the standard HTTP client already included with LocalVectorDB.

Setup:

.. code-block:: bash

   # No additional installation required - Google AI provider is built-in
   # Set one of these environment variables
   export GEMINI_API_KEY=your_api_key_here
   export GOOGLE_API_KEY=your_api_key_here

Configuration:

.. code-block:: python

   # Basic configuration
   db = VectorDB(
       "my_db",
       embedding_provider="google",
       embedding_model="gemini-embedding-001"
   )

   # Advanced configuration with task optimization
   db = VectorDB(
       "my_db",
       embedding_provider="google",
       embedding_model="gemini-embedding-001",
       embedding_config={
           "api_key": "your_api_key_here",     # Or better yet, use GEMINI_API_KEY environment variable instead
           "task_type": "retrieval_document",  # Optimize for document storage
           "requested_dimensions": 1536,       # Control output size
           "normalize": True                    # L2-normalize vectors
       }
   )

Available Models:

- ``gemini-embedding-001``: 3072 dimensions (default), stable production model

Task Types:

- ``semantic_similarity``: General text similarity (default)
- ``classification``: Text classification tasks
- ``clustering``: Document clustering
- ``retrieval_document``: For documents being indexed
- ``retrieval_query``: For search queries
- ``code_retrieval_query``: Code search queries
- ``question_answering``: Q&A systems
- ``fact_verification``: Fact-checking tasks

Configuration Options:

- ``requested_dimensions``: Output size (128-3072), defaults to 3072
- ``normalize``: L2-normalize vectors (recommended for non-3072 outputs)
- ``task_type``: Task-specific optimization

SentenceTransformers Provider
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run any SentenceTransformer model locally. Supports Matryoshka dimension truncation.

.. note::
   Requires the ``sentence-transformers`` optional dependency:
   ``pip install localvectordb[sentence-transformers]``

Configuration:

.. code-block:: python

   # Basic usage
   db = VectorDB(
       "my_db",
       embedding_provider="sentence_transformers",
       embedding_model="all-MiniLM-L6-v2"
   )

   # With Matryoshka dimension truncation
   db = VectorDB(
       "my_db",
       embedding_provider="sentence_transformers",
       embedding_model="all-MiniLM-L6-v2",
       embedding_config={
           "requested_dimensions": 128,  # Truncate to 128 dims
           "normalize": True,
           "device": "cuda"  # Use GPU (cpu/cuda/mps/auto)
       }
   )

HuggingFace Inference API Provider
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use HuggingFace's hosted Inference API for embedding models.

Setup:

.. code-block:: bash

   export HF_TOKEN=your_huggingface_token

Configuration:

.. code-block:: python

   # Using HuggingFace Inference API
   db = VectorDB(
       "my_db",
       embedding_provider="huggingface",
       embedding_model="BAAI/bge-small-en-v1.5"
   )

   # With a custom TEI (Text Embeddings Inference) endpoint
   db = VectorDB(
       "my_db",
       embedding_provider="huggingface",
       embedding_model="BAAI/bge-small-en-v1.5",
       embedding_config={
           "base_url": "http://localhost:8080",  # Custom TEI endpoint
           "requested_dimensions": 256,
           "normalize": True
       }
   )

HuggingFace Local Provider
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run HuggingFace transformer models locally with full control over pooling and device.

.. note::
   Requires the ``local-embeddings`` optional dependency:
   ``pip install localvectordb[local-embeddings]``

Configuration:

.. code-block:: python

   db = VectorDB(
       "my_db",
       embedding_provider="huggingface_local",
       embedding_model="BAAI/bge-small-en-v1.5",
       embedding_config={
           "pooling_strategy": "mean",  # mean, cls, or max
           "device": "cuda",
           "normalize": True,
           "requested_dimensions": 256
       }
   )

Matryoshka Dimension Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Several providers support `Matryoshka Representation Learning (MRL) <https://arxiv.org/abs/2205.13147>`_,
which allows you to truncate embeddings to a smaller dimension while preserving most of their quality.
This reduces storage and speeds up similarity search.

**OpenAI** (``text-embedding-3-small`` and ``text-embedding-3-large`` only):

.. code-block:: python

   # Reduce OpenAI embeddings from 1536 to 256 dimensions
   db = VectorDB(
       "my_db",
       embedding_provider="openai",
       embedding_model="text-embedding-3-small",
       embedding_config={
           "requested_dimensions": 256,
           "normalize": True
       }
   )

**Ollama** (model-dependent):

.. code-block:: python

   # Reduce Ollama embeddings with client-side truncation
   db = VectorDB(
       "my_db",
       embedding_provider="ollama",
       embedding_model="nomic-embed-text",
       embedding_config={
           "requested_dimensions": 256,
           "normalize": True
       }
   )

**SentenceTransformers** and **HuggingFace** providers also support ``requested_dimensions``
for Matryoshka truncation (see their sections above).

Custom Provider Example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from localvectordb.embeddings import EmbeddingProvider, EmbeddingRegistry
   import numpy as np

   class CustomEmbeddingProvider(EmbeddingProvider):
       def __init__(self, model: str, **kwargs):
           super().__init__(model, **kwargs)
           self.api_endpoint = kwargs.get('api_endpoint')

       @property
       def provider_name(self) -> str:
           return "custom"

       @property
       def max_batch_size(self) -> int:
           return 100

       def validate_model(self) -> bool:
           # Check if your model/API is available
           return True

       def get_dimension(self) -> int:
           return 768  # Your embedding dimension

       async def _embed_single_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
           # Implement your embedding logic for a single batch
           embeddings = []
           for text in texts:
               # Call your embedding API/model
               embedding = await self._get_embedding(text)
               embeddings.append(embedding)
           return embeddings

       async def _get_embedding(self, text: str) -> List[float]:
           # Your implementation here
           pass

   # Register custom provider
   EmbeddingRegistry.register("custom", CustomEmbeddingProvider)

   # Use custom provider
   db = VectorDB(
       "my_db",
       embedding_provider="custom",
       embedding_model="your-model",
       embedding_config={
           "api_endpoint": "https://your-api.com/embed"
       }
   )

Direct Embedding API
--------------------

Use embedding providers directly without a database:

.. code-block:: python

   from localvectordb.embeddings import EmbeddingRegistry

   # Create provider
   provider = EmbeddingRegistry.create_provider(
       "ollama",
       "nomic-embed-text"
   )

   # Generate embeddings
   texts = ["Hello world", "How are you?", "Goodbye"]

   # Synchronous
   embeddings = provider.embed_sync(texts)
   print(f"Shape: {embeddings.shape}")  # (3, 768)

   # Asynchronous
   import asyncio
   embeddings = await provider.embed_batch(texts)

Provider Comparison
-------------------

Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import time
   from localvectordb.embeddings import EmbeddingRegistry

   def benchmark_provider(provider_name, model, texts):
       provider = EmbeddingRegistry.create_provider(provider_name, model)

       # Validate model
       if not provider.validate_model():
           print(f"{provider_name} model {model} not available")
           return

       # Time embedding generation
       start_time = time.time()
       embeddings = provider.embed_sync(texts)
       duration = time.time() - start_time

       dimension = embeddings.shape[1]
       speed = len(texts) / duration

       print(f"{provider_name}/{model}:")
       print(f"  Dimension: {dimension}")
       print(f"  Speed: {speed:.1f} texts/second")
       print(f"  Total time: {duration:.2f}s")

   # Test different providers
   test_texts = ["Example text " + str(i) for i in range(100)]

   benchmark_provider("ollama", "nomic-embed-text", test_texts)
   benchmark_provider("ollama", "all-minilm", test_texts)
   benchmark_provider("openai", "text-embedding-3-small", test_texts)
   benchmark_provider("jina", "jina-embeddings-v4", test_texts)
   benchmark_provider("google", "gemini-embedding-001", test_texts)

Quality Considerations
^^^^^^^^^^^^^^^^^^^^^^

+----------------------+----------------------------+------------+-----------+-----------------+
| Provider             | Model                      | Dimensions | Speed     | Cost            |
+======================+============================+============+===========+=================+
| Ollama               | nomic-embed-text           | 768        | Medium    | Free            |
+----------------------+----------------------------+------------+-----------+-----------------+
| Ollama               | mxbai-embed-large          | 1024       | Medium    | Free            |
+----------------------+----------------------------+------------+-----------+-----------------+
| Ollama               | all-minilm                 | 384        | Fast      | Free            |
+----------------------+----------------------------+------------+-----------+-----------------+
| OpenAI               | text-embedding-3-small     | 1536       | Fast      | $0.02/1M tokens |
+----------------------+----------------------------+------------+-----------+-----------------+
| OpenAI               | text-embedding-3-large     | 3072       | Fast      | $0.13/1M tokens |
+----------------------+----------------------------+------------+-----------+-----------------+
| JinaAI               | jina-embeddings-v4         | 2048       | Fast      | Free tier       |
+----------------------+----------------------------+------------+-----------+-----------------+
| JinaAI               | jina-embeddings-v3         | 1024       | Fast      | Free tier       |
+----------------------+----------------------------+------------+-----------+-----------------+
| JinaAI               | jina-code-embeddings-1.5b  | 1536       | Fast      | Free tier       |
+----------------------+----------------------------+------------+-----------+-----------------+
| Google AI            | gemini-embedding-001       | 3072       | Fast      | Free tier       |
+----------------------+----------------------------+------------+-----------+-----------------+
| SentenceTransformers | all-MiniLM-L6-v2           | 384        | Fast      | Free (local)    |
+----------------------+----------------------------+------------+-----------+-----------------+
| HuggingFace          | BAAI/bge-small-en-v1.5     | 384        | Fast      | Free tier       |
+----------------------+----------------------------+------------+-----------+-----------------+
| HuggingFace Local    | BAAI/bge-small-en-v1.5     | 384        | Fast      | Free (local)    |
+----------------------+----------------------------+------------+-----------+-----------------+

Advanced Configuration
----------------------

``embedding_config`` Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``embedding_config`` dictionary is passed as keyword arguments to the embedding provider's
constructor. All providers inherit a set of common parameters from the base ``EmbeddingProvider`` class,
plus provider-specific options.

**Common parameters (all providers):**

.. code-block:: python

   embedding_config={
       "timeout": 90,                  # Request timeout in seconds (default: 90, Ollama default: 300)
       "max_retries": 3,               # Number of retries on failure (default: 3)
       "retry_delay": 1.0,             # Initial retry delay in seconds, with exponential backoff (default: 1.0)
       "max_concurrent_requests": 5,   # Max parallel batch requests (default: 5, Ollama default: 3)
   }

**Provider-specific parameters:**

+----------------------+-------------------------------+----------------------------------------------------------+
| Provider             | Parameter                     | Description                                              |
+======================+===============================+==========================================================+
| Ollama               | ``base_url``                  | Ollama server URL (default: ``$OLLAMA_URL`` or           |
|                      |                               | ``http://localhost:11434``)                               |
+----------------------+-------------------------------+----------------------------------------------------------+
| Ollama               | ``requested_dimensions``      | Truncate output to N dims (Matryoshka/MRL)               |
+----------------------+-------------------------------+----------------------------------------------------------+
| Ollama               | ``normalize``                 | L2-normalize output vectors (bool)                       |
+----------------------+-------------------------------+----------------------------------------------------------+
| OpenAI               | ``api_key``                   | API key (default: ``$OPENAI_API_KEY``). Prefix with      |
|                      |                               | ``$`` to read from a custom env var, e.g. ``$MY_KEY``    |
+----------------------+-------------------------------+----------------------------------------------------------+
| OpenAI               | ``base_url``                  | API base URL for OpenAI-compatible endpoints             |
+----------------------+-------------------------------+----------------------------------------------------------+
| OpenAI               | ``requested_dimensions``      | Output dims (MRL, v3 models only)                        |
+----------------------+-------------------------------+----------------------------------------------------------+
| OpenAI               | ``normalize``                 | L2-normalize output vectors (bool)                       |
+----------------------+-------------------------------+----------------------------------------------------------+
| JinaAI               | ``api_key``                   | API key (default: ``$JINA_API_KEY``)                     |
+----------------------+-------------------------------+----------------------------------------------------------+
| JinaAI               | ``task``                      | Task-specific optimization (see JinaAI section above)    |
+----------------------+-------------------------------+----------------------------------------------------------+
| JinaAI               | ``requested_dimensions``      | Truncate output to N dimensions                          |
+----------------------+-------------------------------+----------------------------------------------------------+
| JinaAI               | ``truncate``                  | Whether to truncate long inputs (bool)                   |
+----------------------+-------------------------------+----------------------------------------------------------+
| JinaAI               | ``late_chunking``             | Enable late chunking (bool)                              |
+----------------------+-------------------------------+----------------------------------------------------------+
| Google AI            | ``api_key``                   | API key (default: ``$GEMINI_API_KEY`` or                 |
|                      |                               | ``$GOOGLE_API_KEY``)                                     |
+----------------------+-------------------------------+----------------------------------------------------------+
| Google AI            | ``task_type``                 | Task-specific optimization (see Google AI section above) |
+----------------------+-------------------------------+----------------------------------------------------------+
| Google AI            | ``requested_dimensions``      | Output size (128-3072)                                   |
+----------------------+-------------------------------+----------------------------------------------------------+
| Google AI            | ``normalize``                 | L2-normalize output vectors (bool)                       |
+----------------------+-------------------------------+----------------------------------------------------------+
| SentenceTransformers | ``device``                    | Inference device (cpu/cuda/mps/auto)                     |
+----------------------+-------------------------------+----------------------------------------------------------+
| SentenceTransformers | ``requested_dimensions``      | Truncate output to N dims (Matryoshka)                   |
+----------------------+-------------------------------+----------------------------------------------------------+
| SentenceTransformers | ``normalize``                 | L2-normalize output vectors (bool, default: True)        |
+----------------------+-------------------------------+----------------------------------------------------------+
| SentenceTransformers | ``trust_remote_code``         | Trust remote code when loading model (bool)              |
+----------------------+-------------------------------+----------------------------------------------------------+
| HuggingFace          | ``api_key``                   | API key (default: ``$HF_TOKEN`` or                       |
|                      |                               | ``$HUGGINGFACE_TOKEN``)                                  |
+----------------------+-------------------------------+----------------------------------------------------------+
| HuggingFace          | ``base_url``                  | Custom TEI endpoint URL                                  |
+----------------------+-------------------------------+----------------------------------------------------------+
| HuggingFace          | ``requested_dimensions``      | Truncate output to N dimensions                          |
+----------------------+-------------------------------+----------------------------------------------------------+
| HuggingFace          | ``normalize``                 | L2-normalize output vectors (bool, default: True)        |
+----------------------+-------------------------------+----------------------------------------------------------+
| HuggingFace Local    | ``device``                    | Inference device (cpu/cuda/mps)                          |
+----------------------+-------------------------------+----------------------------------------------------------+
| HuggingFace Local    | ``pooling_strategy``          | Pooling method: mean, cls, or max (default: mean)        |
+----------------------+-------------------------------+----------------------------------------------------------+
| HuggingFace Local    | ``requested_dimensions``      | Truncate output to N dimensions                          |
+----------------------+-------------------------------+----------------------------------------------------------+
| HuggingFace Local    | ``normalize``                 | L2-normalize output vectors (bool, default: True)        |
+----------------------+-------------------------------+----------------------------------------------------------+
| HuggingFace Local    | ``trust_remote_code``         | Trust remote code when loading model (bool)              |
+----------------------+-------------------------------+----------------------------------------------------------+

Retry behavior uses exponential backoff: the delay after attempt *n* is ``retry_delay * 2^n`` seconds.
Retries are triggered by network errors, timeouts, HTTP 429 (rate limit), and 5xx server errors.

Batch Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Configure batch sizes for optimal performance
   db = VectorDB(
       "my_db",
       embedding_provider="ollama",
       embedding_model="nomic-embed-text",
       embedding_config={
           "batch_size": 32,  # Process 32 texts at once
           "timeout": 120     # Longer timeout for large batches
       }
   )

   # Manual batch processing
   large_documents = ["document " + str(i) for i in range(1000)]

   # Insert with custom batch size
   doc_ids = db.upsert(
       documents=large_documents,
       batch_size=50  # Process 50 documents at a time
   )

Error Handling and Retries
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from localvectordb.exceptions import EmbeddingError

   try:
       db = VectorDB(
           "my_db",
           embedding_provider="ollama",
           embedding_model="nonexistent-model"
       )
   except EmbeddingError as e:
       print(f"Embedding error: {e}")

       # Fallback to different model
       db = VectorDB(
           "my_db",
           embedding_provider="ollama",
           embedding_model="all-minilm"  # Smaller, more reliable model
       )

Provider Selection Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def create_db_with_fallback(name, preferred_provider="ollama"):
       """Create database with provider fallback"""

       providers_to_try = [
           ("ollama", "nomic-embed-text"),
           ("ollama", "all-minilm"),
           ("openai", "text-embedding-3-small")
       ]

       if preferred_provider == "openai":
           providers_to_try = providers_to_try[::-1]  # Try OpenAI first

       for provider, model in providers_to_try:
           try:
               # Test provider availability
               test_provider = EmbeddingRegistry.create_provider(provider, model)
               if test_provider.validate_model():
                   return VectorDB(
                       name,
                       embedding_provider=provider,
                       embedding_model=model
                   )
           except Exception as e:
               print(f"Failed to use {provider}/{model}: {e}")
               continue

       raise Exception("No embedding providers available")

   # Use with fallback
   db = create_db_with_fallback("my_db", preferred_provider="ollama")

Plugin Development
------------------

Creating an Embedding Plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a Python package with entry points.

**setup.py**:

.. code-block:: python

   from setuptools import setup

   setup(
       name="my-embedding-provider",
       version="1.0.0",
       packages=["my_embedding_provider"],
       entry_points={
           'localvectordb.embedding_providers': [
               'my_provider = my_embedding_provider:MyEmbeddingProvider',
           ],
       },
       install_requires=[
           "localvectordb>=1.0.0",
           "requests",  # Your dependencies
       ]
   )

**my_embedding_provider/__init__.py**:

.. code-block:: python

   from localvectordb.embeddings import EmbeddingProvider
   import numpy as np
   import requests

   class MyEmbeddingProvider(EmbeddingProvider):
       def __init__(self, model: str, **kwargs):
           super().__init__(model, **kwargs)
           self.api_url = kwargs.get('api_url', 'https://api.example.com')
           self.api_key = kwargs.get('api_key')

       @property
       def provider_name(self) -> str:
           return "my_provider"

       @property
       def max_batch_size(self) -> int:
           return 50

       def validate_model(self) -> bool:
           try:
               response = requests.get(f"{self.api_url}/models/{self.model}")
               return response.status_code == 200
           except:
               return False

       def get_dimension(self) -> int:
           # Return dimension for your model
           return 512

       async def _embed_single_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
           response = requests.post(
               f"{self.api_url}/embed",
               json={
                   "model": self.model,
                   "input": texts
               },
               headers={"Authorization": f"Bearer {self.api_key}"}
           )

           if response.status_code != 200:
               raise RuntimeError(f"API error: {response.text}")

           return response.json()['embeddings']

Installation and Usage:

.. code-block:: bash

   pip install my-embedding-provider

.. code-block:: console

   # Now use in LocalVectorDB
   python -c "
   from localvectordb import VectorDB
   db = VectorDB(
       'test_db',
       embedding_provider='my_provider',
       embedding_model='my-model-v1',
       embedding_config={'api_key': 'your_key'}
   )
   "

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

Ollama connection errors:

.. code-block:: python

   # Test Ollama connection
   from localvectordb.embeddings import EmbeddingRegistry

   try:
       provider = EmbeddingRegistry.create_provider("ollama", "nomic-embed-text")
       if provider.validate_model():
           print("Ollama working correctly")
       else:
           print("Model not available, try: ollama pull nomic-embed-text")
   except Exception as e:
       print(f"Ollama error: {e}")
       print("Check if Ollama is running: ollama list")

OpenAI authentication errors:

.. code-block:: python

   import os

   # Verify API key
   api_key = os.getenv("OPENAI_API_KEY")
   if not api_key:
       print("Set OPENAI_API_KEY environment variable")
   elif not api_key.startswith("sk-"):
       print("Invalid OpenAI API key format")
   else:
       print("API key configured correctly")

Dimension mismatch errors:

.. code-block:: python

   # Check embedding dimensions
   provider = EmbeddingRegistry.create_provider("ollama", "nomic-embed-text")
   dimension = provider.get_dimension()
   print(f"Model dimension: {dimension}")

   # When switching models, ensure dimensions match
   # or create a new database with the new model

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Optimize embedding performance
   import asyncio
   from localvectordb.embeddings import embed_texts

   async def fast_embedding_example():
       texts = ["Text " + str(i) for i in range(1000)]

       # Process in parallel with optimal batch size
       embeddings = await embed_texts(
           texts=texts,
           provider="ollama",
           model="all-minilm",  # Fastest model
           batch_size=64  # Optimize based on your hardware
       )

       return embeddings

   # Run async embedding
   embeddings = asyncio.run(fast_embedding_example())
