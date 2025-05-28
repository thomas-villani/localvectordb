Embeddings
==========

LocalVectorDB v2.0 features a plugin-based embedding system that supports multiple providers with a unified interface. The system is designed for flexibility, allowing easy switching between providers and custom implementations.

Overview
--------

**Embeddings** are dense vector representations of text that capture semantic meaning. LocalVectorDB supports multiple embedding providers:

- **Ollama**: Local embeddings without API costs
- **OpenAI**: Cloud-based embeddings with high quality
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
^^^^^^^^^^^^^^^^

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

       async def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
           # Implement your embedding logic
           embeddings = []
           for text in texts:
               # Call your embedding API/model
               embedding = await self._get_embedding(text)
               embeddings.append(embedding)
           return np.array(embeddings, dtype=np.float32)

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

Quality Considerations
^^^^^^^^^^^^^^^^^^^^^^

+----------------------+------------------------+------------+-----------+-----------------+
| Provider             | Model                  | Dimensions | Speed     | Cost            |
+======================+========================+============+===========+=================+
| Ollama               | nomic-embed-text       | 768        | Medium    | Free            |
+----------------------+------------------------+------------+-----------+-----------------+
| Ollama               | mxbai-embed-large      | 1024       | Medium    | Free            |
+----------------------+------------------------+------------+-----------+-----------------+
| Ollama               | all-minilm             | 384        | Fast      | Free            |
+----------------------+------------------------+------------+-----------+-----------------+
| OpenAI               | text-embedding-3-small | 1536       | Fast      | $0.02/1M tokens |
+----------------------+------------------------+------------+-----------+-----------------+
| OpenAI               | text-embedding-3-large | 3072       | Fast      | $0.13/1M tokens |
+----------------------+------------------------+------------+-----------+-----------------+

Advanced Configuration
----------------------

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
           "localvectordb>=2.0.0",
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

       async def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
           batch_size = batch_size or self.max_batch_size
           all_embeddings = []

           for i in range(0, len(texts), batch_size):
               batch = texts[i:i + batch_size]

               response = requests.post(
                   f"{self.api_url}/embed",
                   json={
                       "model": self.model,
                       "input": batch
                   },
                   headers={"Authorization": f"Bearer {self.api_key}"}
               )

               if response.status_code != 200:
                   raise RuntimeError(f"API error: {response.text}")

               batch_embeddings = response.json()['embeddings']
               all_embeddings.extend(batch_embeddings)

           return np.array(all_embeddings, dtype=np.float32)

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
