# LocalVectorDB

**LocalVectorDB** is a **document-first vector database** that combines the simplicity of SQLite with the power of FAISS for semantic search. Unlike traditional vector databases that require you to manage chunks manually, LocalVectorDB lets you work with complete documents while automatically handling the chunking, embedding, and indexing behind the scenes.

## ✨ Key Features

- **🗂️ Document-First API**: Work with documents, not chunks—chunking handled automatically
- **🔍 Unified Search**: Vector, keyword, and hybrid search with normalized scoring
- **📍 Position Tracking**: Perfect document reconstruction and precise highlighting
- **🗄️ Structured Metadata**: SQLite-backed metadata with indexed columns and schema validation
- **🔌 Plugin Embeddings**: Support for Ollama (free, local), OpenAI, and custom providers
- **🚀 Production Ready**: HTTP server, CLI tools, authentication, and monitoring
- **☁️ Local + Remote**: Identical API for local databases and remote server connections

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install localvectordb

# With server and CLI tools
pip install localvectordb[server]
```

### 5-Minute Example

```python
from localvectordb import VectorDB
from localvectordb.core import MetadataField, MetadataFieldType

# Create a document database with metadata schema
db = VectorDB(
    name="my_documents",
    base_path="./my_vectordb",
    metadata_schema={
        'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        'date': MetadataField(type=MetadataFieldType.DATE, indexed=True),
        'tags': MetadataField(type=MetadataFieldType.JSON)
    },
    embedding_model="nomic-embed-text",
    chunk_size=500
)

# Add documents with metadata
documents = [
    "LocalVectorDB is a document-first vector database...",
    "Python is a powerful programming language...",
    "Machine learning enables computers to learn..."
]

metadata = [
    {"title": "LocalVectorDB Guide", "author": "AI Assistant", "date": "2024-01-01"},
    {"title": "Python Basics", "author": "Developer", "date": "2024-01-02"},
    {"title": "ML Introduction", "author": "Data Scientist", "date": "2024-01-03"}
]

# Insert documents
doc_ids = db.upsert(documents, metadata=metadata)
print(f"Added documents: {doc_ids}")

# Search documents
results = db.query("vector database", search_type="vector", k=3)
for result in results:
    print(f"Score: {result.score:.3f} | {result.content[:100]}...")

# Hybrid search combining vector and keyword search  
results = db.query("python programming", search_type="hybrid", k=2)

# Filter by metadata
python_docs = db.filter(where={"author": "Developer"})

db.close()
```

## 🌐 Server Usage

### Start the Server

```bash
# Start LocalVectorDB server
lvdb serve --host 0.0.0.0 --port 5000

# Create database via CLI
lvdb create my_database --embedding-model nomic-embed-text

# Add documents 
lvdb db my_database add document.txt

# Search documents
lvdb db my_database search "query text" --limit 5
```

### Remote Client

```python
from localvectordb import VectorDB

# Connect to remote server (same API as local!)
db = VectorDB(
    name="my_remote_db", 
    base_path="http://localhost:5000",
    api_key="your_api_key"
)

# Identical API to local database
doc_ids = db.upsert(["Remote document content"])
results = db.query("search query")
```

### REST API

```bash
# Create database
curl -X POST http://localhost:5000/api/v1/databases \
  -H "Content-Type: application/json" \
  -d '{"name": "api_db", "embedding_model": "nomic-embed-text"}'

# Add documents
curl -X POST http://localhost:5000/api/v1/api_db/documents \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Document content"], "metadata": [{"title": "Test"}]}'

# Search
curl -X POST http://localhost:5000/api/v1/api_db/query \
  -H "Content-Type: application/json" \
  -d '{"query": "search text", "search_type": "vector", "k": 5}'
```

## 📋 Use Cases

### 🔬 Research & Academia
```python
# Research paper database with structured metadata
db = VectorDB("research_papers", metadata_schema={
    'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
    'authors': MetadataField(type=MetadataFieldType.JSON),
    'journal': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
    'year': MetadataField(type=MetadataFieldType.INTEGER, indexed=True),
    'doi': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
})

# Search for specific research topics
results = db.query("neural networks", filters={"journal": "Nature", "year": {">=": 2020}})
```

### 💼 Enterprise Knowledge Base
```python
# Company documents with department-based access
db = VectorDB("company_kb", metadata_schema={
    'department': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
    'document_type': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
    'confidentiality': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
    'last_updated': MetadataField(type=MetadataFieldType.DATE, indexed=True)
})

# Secure search with access controls
results = db.query("project requirements", 
                  filters={"department": "engineering", "confidentiality": "internal"})
```

### 🛠️ Code Documentation
```python
# Source code with intelligent chunking
db = VectorDB("codebase", 
              chunking_method="code-blocks",  # Preserves code structure
              metadata_schema={
                  'file_path': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                  'language': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                  'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
              })

# Find code examples
results = db.query("authentication middleware", 
                  filters={"language": "python"}, return_type="chunks")
```

## 🔧 Advanced Features

### Multiple Search Types
```python
# Vector search (semantic similarity)
vector_results = db.query("machine learning", search_type="vector")

# Keyword search (exact matches)
keyword_results = db.query("neural networks", search_type="keyword")

# Hybrid search (best of both)
hybrid_results = db.query("AI algorithms", search_type="hybrid", vector_weight=0.7)
```

### Flexible Embedding Providers
```python
# Local embeddings with Ollama (free, no API keys)
db_local = VectorDB("local_db", 
                    embedding_provider="ollama",
                    embedding_model="nomic-embed-text")

# Cloud embeddings with OpenAI
db_cloud = VectorDB("cloud_db",
                    embedding_provider="openai", 
                    embedding_model="text-embedding-3-small")
```

### Advanced Chunking Strategies
```python
# Different strategies for different content
sentences_db = VectorDB("articles", chunking_method="sentences")
paragraphs_db = VectorDB("essays", chunking_method="paragraphs") 
sections_db = VectorDB("docs", chunking_method="sections")
code_db = VectorDB("source", chunking_method="code-blocks")
```

### Production Deployment
```bash
# Initialize production configuration
lvdb config init --format toml --output production.toml

# Start production server with authentication
lvdb serve --config production.toml --host 0.0.0.0 --port 8080

# Create API keys for authentication
lvdb auth create-key --description "Production API" --expires-days 90

# Monitor server health
curl http://localhost:8080/api/v1/health
```

## 📚 Documentation

- **📖 [Full Documentation](https://localvectordb.readthedocs.io)**: Comprehensive guides and API reference
- **🚀 [Quickstart Guide](https://localvectordb.readthedocs.io/en/latest/quickstart.html)**: Get up and running in 5 minutes
- **🔧 [Installation](https://localvectordb.readthedocs.io/en/latest/installation.html)**: Detailed setup instructions
- **📊 [Examples](https://github.com/your-org/localvectordb/tree/main/examples)**: Real-world usage examples
- **🛠️ [CLI Reference](https://localvectordb.readthedocs.io/en/latest/cli.html)**: Complete command-line guide

## 🛠️ Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, Windows
- **Optional**: [Ollama](https://ollama.ai) for local embeddings (recommended)
- **Optional**: NVIDIA GPU for accelerated vector operations

## 🔗 Installation Options

```bash
# Minimal installation
pip install localvectordb

# With HTTP server and CLI tools  
pip install localvectordb[server]

# Development installation with testing tools
pip install localvectordb[dev]

# Install from source
git clone https://github.com/your-org/localvectordb.git
cd localvectordb
pip install -e .[dev]
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/your-org/localvectordb.git
cd localvectordb
pip install -e .[dev]

# Run tests
pytest

# Run linting
flake8 src/
black src/

# Build documentation
cd docs/
make html
```

## 📊 Performance

LocalVectorDB is designed for both development and production use:

- **🏃‍♂️ Fast**: SQLite + FAISS provide excellent performance for most use cases
- **📈 Scalable**: Handle millions of documents with proper hardware
- **💾 Efficient**: Smart chunking and connection pooling minimize resource usage
- **🔧 Tunable**: Extensive configuration options for optimization

## 🆚 Comparison

| Feature | LocalVectorDB | Pinecone | Weaviate | ChromaDB |
|---------|---------------|----------|----------|----------|
| **Local Development** | ✅ | ❌ | ✅ | ✅ |
| **Document-First API** | ✅ | ❌ | ❌ | ❌ |
| **Position Tracking** | ✅ | ❌ | ❌ | ❌ |
| **Structured Metadata** | ✅ | ✅ | ✅ | ❌ |
| **Hybrid Search** | ✅ | ❌ | ✅ | ❌ |
| **No API Keys Required** | ✅ | ❌ | ✅ | ✅ |
| **Production Ready** | ✅ | ✅ | ✅ | ⚠️ |

## 📄 License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

- ✅ **Permitted**: Personal use, research, education, non-commercial projects
- ❌ **Requires Permission**: Commercial use, redistribution in commercial products
- 📧 **Commercial Licensing**: Contact [thomas.villani@gmail.com](mailto:thomas.villani@gmail.com) for commercial licensing options

## 🙏 Acknowledgments

- **[FAISS](https://github.com/facebookresearch/faiss)**: Meta's vector similarity search library
- **[SQLite](https://www.sqlite.org/)**: The world's most deployed database engine
- **[Ollama](https://ollama.ai)**: Local AI model runtime
- **[tiktoken](https://github.com/openai/tiktoken)**: OpenAI's tokenization library

## 📞 Support

- **🐛 [Bug Reports](https://github.com/your-org/localvectordb/issues)**: Found a bug? Let us know!
- **💡 [Feature Requests](https://github.com/your-org/localvectordb/discussions)**: Have an idea? We'd love to hear it!
- **💬 [Community Discord](https://discord.gg/localvectordb)**: Join our community for help and discussions
- **📧 [Email Support](mailto:support@localvectordb.com)**: Direct support for commercial users

---

**⭐ Star this repo if LocalVectorDB is useful for your projects!**

*Built with ❤️ for the AI and developer community*