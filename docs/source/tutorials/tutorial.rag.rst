=======================================================
LocalVectorDB Tutorial: Building a RAG Chat Application
=======================================================

This tutorial will guide you through building a complete Retrieval-Augmented Generation (RAG) chat application using LocalVectorDB and Ollama. By the end, you'll have a working chatbot that can answer questions based on your own documents.

What We'll Build
================

We'll create a chat application that:

* Uses LocalVectorDB to store and search through documents
* Employs Ollama for both embeddings and chat responses
* Implements a simple RAG pipeline to provide context-aware answers
* Includes document ingestion and real-time querying capabilities

Prerequisites
=============

Before starting, ensure you have:

* Python 3.12 or higher
* Ollama installed and running
* The required Python packages

Installation
============

First, install the required dependencies:

.. code-block:: bash

   pip install localvectordb ollama

Next, ensure Ollama is running and pull the required models:

.. code-block:: bash

   # Start Ollama (if not already running)
   ollama serve

   # Pull the embedding model
   ollama pull nomic-embed-text

   # Pull a chat model
   ollama pull llama3.2

Setting Up the Database
=======================

Let's start by creating our vector database with a proper metadata schema for document management:

.. code-block:: python

   from localvectordb import VectorDB, LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType
   from pathlib import Path
   import logging

   # We create the database with the ``VectorDB`` factory (which returns a
   # ``LocalVectorDB`` for a local path); ``LocalVectorDB`` is imported as well so
   # we can use it in the type annotations of the helper functions below.

   # Configure logging to see what's happening
   logging.basicConfig(level=logging.INFO)

   # Define metadata schema for our documents
   metadata_schema = {
       'title': MetadataField(
           type=MetadataFieldType.TEXT,
           indexed=True,
           required=True
       ),
       'source': MetadataField(
           type=MetadataFieldType.TEXT,
           indexed=True
       ),
       'category': MetadataField(
           type=MetadataFieldType.TEXT,
           indexed=True
       ),
       'created_date': MetadataField(
           type=MetadataFieldType.DATE,
           indexed=True
       ),
       'word_count': MetadataField(
           type=MetadataFieldType.INTEGER
       )
   }

   # Create the database
   db = VectorDB(
       name="rag_knowledge_base",
       base_path="./vector_storage",
       metadata_schema=metadata_schema,
       embedding_provider="ollama",
       embedding_model="nomic-embed-text",
       chunking_method="sentences",
       chunk_size=500,
       chunk_overlap=50,
       enable_fts=True  # Enable full-text search for keyword queries
   )

   print(f"Database created with {db.embedding_dimension} dimensional embeddings")

Document Ingestion
==================

Now let's create a function to ingest documents into our database:

.. code-block:: python

   from datetime import datetime
   from typing import List, Dict, Any

   def ingest_documents(db: LocalVectorDB, documents: List[Dict[str, Any]]) -> List[str]:
       """
       Ingest a list of documents into the vector database.

       Args:
           db: The LocalVectorDB instance
           documents: List of document dictionaries with 'content' and metadata

       Returns:
           List of document IDs that were inserted
       """
       texts = []
       metadata_list = []

       for doc in documents:
           # Extract content
           content = doc.get('content', '')
           if not content.strip():
               continue

           texts.append(content)

           # Prepare metadata
           metadata = {
               'title': doc.get('title', 'Untitled'),
               'source': doc.get('source', 'unknown'),
               'category': doc.get('category', 'general'),
               'created_date': doc.get('created_date', datetime.now().isoformat()),
               'word_count': len(content.split())
           }
           metadata_list.append(metadata)

       # Insert documents in batch
       if texts:
           doc_ids = db.upsert(
               documents=texts,
               metadata=metadata_list,
               batch_size=50,
               similarity_threshold=0.85  # Avoid duplicate content
           )
           print(f"Successfully ingested {len(doc_ids)} documents")
           return doc_ids

       return []

   # Example: Ingest some sample documents
   sample_documents = [
       {
           'content': """
           Python is a high-level programming language known for its simplicity and readability.
           It was created by Guido van Rossum and first released in 1991. Python supports
           multiple programming paradigms including procedural, object-oriented, and functional programming.
           The language emphasizes code readability with its notable use of significant whitespace.
           """,
           'title': 'Introduction to Python',
           'source': 'programming_guide.md',
           'category': 'programming'
       },
       {
           'content': """
           Machine learning is a subset of artificial intelligence that enables systems to learn
           and improve from experience without being explicitly programmed. It focuses on developing
           algorithms that can access data and use it to learn for themselves. The process involves
           training algorithms on data sets to find patterns and make predictions or decisions.
           """,
           'title': 'Machine Learning Basics',
           'source': 'ml_handbook.md',
           'category': 'ai'
       },
       {
           'content': """
           LocalVectorDB is a document-first vector database that combines SQLite for metadata
           storage with FAISS for vector similarity search. It provides a simple API for storing,
           searching, and managing documents with their embeddings. The database supports various
           chunking strategies and embedding providers, making it ideal for RAG applications.
           """,
           'title': 'LocalVectorDB Overview',
           'source': 'documentation.md',
           'category': 'database'
       }
   ]

   # Ingest the sample documents
   ingested_ids = ingest_documents(db, sample_documents)

Building the RAG Pipeline
==========================

Now let's create the core RAG functionality:

.. code-block:: python

   import ollama
   from typing import Optional

   class RAGChatBot:
       """
       A RAG-powered chatbot using LocalVectorDB and Ollama.
       """

       def __init__(self, db: LocalVectorDB, chat_model: str = "llama3.2"):
           self.db = db
           self.chat_model = chat_model
           self.conversation_history = []

       def search_knowledge_base(
           self,
           query: str,
           search_type: str = "hybrid",
           k: int = 5,
           category_filter: Optional[str] = None
       ) -> List[Dict[str, Any]]:
           """
           Search the knowledge base for relevant information.

           Args:
               query: The search query
               search_type: Type of search ('vector', 'keyword', 'hybrid')
               k: Number of results to return
               category_filter: Optional category to filter by

           Returns:
               List of relevant document chunks with metadata
           """
           # Prepare filters
           filters = {}
           if category_filter:
               filters['category'] = category_filter

           # Search the database
           results = self.db.query(
               query=query,
               search_type=search_type,
               return_type="chunks",  # Get specific chunks for better context
               k=k,
               score_threshold=0.3,  # Only return reasonably relevant results
               filters=filters
           )

           # Format results for context
           formatted_results = []
           for result in results:
               formatted_results.append({
                   'content': result.content,
                   'score': result.score,
                   'title': result.metadata.get('title', 'Unknown'),
                   'source': result.metadata.get('source', 'Unknown'),
                   'category': result.metadata.get('category', 'general')
               })

           return formatted_results

       def generate_response(self, user_query: str, context_results: List[Dict[str, Any]]) -> str:
           """
           Generate a response using Ollama with the retrieved context.

           Args:
               user_query: The user's question
               context_results: Relevant information from the knowledge base

           Returns:
               The generated response
           """
           # Build context from search results
           context_parts = []
           for i, result in enumerate(context_results, 1):
               context_parts.append(
                   f"[Source {i}: {result['title']}]\n{result['content']}\n"
               )

           context = "\n".join(context_parts)

           # Create the prompt
           prompt = f"""You are a helpful assistant that answers questions based on the provided context.
   Use the context below to answer the user's question. If the context doesn't contain enough information
   to answer the question, say so clearly.

   Context:
   {context}

   Question: {user_query}

   Answer:"""

           try:
               # Generate response using Ollama
               response = ollama.generate(
                   model=self.chat_model,
                   prompt=prompt,
                   options={
                       'temperature': 0.7,
                       'num_predict': 500,  # Ollama's name for max tokens to generate
                       'top_p': 0.9
                   }
               )

               return response['response'].strip()

           except Exception as e:
               return f"Sorry, I encountered an error generating a response: {str(e)}"

       def chat(self, user_query: str, category_filter: Optional[str] = None) -> Dict[str, Any]:
           """
           Process a user query and return a comprehensive response.

           Args:
               user_query: The user's question or message
               category_filter: Optional category to limit search to

           Returns:
               Dictionary containing the response and metadata
           """
           print(f"🔍 Searching knowledge base for: '{user_query}'")

           # Search for relevant information
           context_results = self.search_knowledge_base(
               query=user_query,
               search_type="hybrid",  # Use hybrid search for best results
               k=3,  # Get top 3 most relevant chunks
               category_filter=category_filter
           )

           print(f"📚 Found {len(context_results)} relevant chunks")

           # Generate response
           if context_results:
               response = self.generate_response(user_query, context_results)
           else:
               response = "I couldn't find any relevant information in my knowledge base to answer your question."

           # Store in conversation history
           chat_entry = {
               'query': user_query,
               'response': response,
               'sources': [r['title'] for r in context_results],
               'timestamp': datetime.now().isoformat()
           }
           self.conversation_history.append(chat_entry)

           return {
               'response': response,
               'sources': context_results,
               'total_sources': len(context_results)
           }

Creating the Chat Interface
===========================

Let's create a simple command-line interface for our RAG chatbot:

.. code-block:: python

   def run_chat_interface():
       """
       Run an interactive chat interface for the RAG chatbot.
       """
       print("RAG ChatBot initialized!")
       print("Knowledge base loaded with LocalVectorDB")
       print("Type 'quit' to exit, 'help' for commands")
       print("=" * 50)

       # Initialize the chatbot
       chatbot = RAGChatBot(db)

       while True:
           try:
               # Get user input
               user_input = input("\nYou: ").strip()

               if not user_input:
                   continue

               # Handle special commands
               if user_input.lower() == 'quit':
                   print("Goodbye!")
                   break

               elif user_input.lower() == 'help':
                   print("""
   Available commands:
   - quit: Exit the chat
   - help: Show this help message
   - stats: Show database statistics
   - categories: List available categories
   - search [category]: Search within a specific category

   You can also ask any question and I'll search my knowledge base!
                   """)
                   continue

               elif user_input.lower() == 'stats':
                   stats = db.get_stats()
                   print(f"""
   Database Statistics:
   - Documents: {stats['documents']}
   - Chunks: {stats['chunks']}
   - Index vectors: {stats['index_vectors']}
   - Embedding model: {stats['embedding_model']}
   - FTS enabled: {stats['fts_enabled']}
                   """)
                   continue

               elif user_input.lower() == 'categories':
                   # Get unique categories
                   docs = db.filter(limit=100)  # Get sample of documents
                   categories = set(doc.metadata.get('category', 'unknown') for doc in docs)
                   print(f"Available categories: {', '.join(sorted(categories))}")
                   continue

               elif user_input.lower().startswith('search '):
                   category = user_input[7:].strip()
                   query = input(f"Search query for '{category}' category: ")
                   result = chatbot.chat(query, category_filter=category)
               else:
                   # Regular chat query
                   result = chatbot.chat(user_input)

               # Display response
               print(f"\nBot: {result['response']}")

               # Show sources if any
               if result['sources']:
                   print(f"\nSources ({result['total_sources']}):")
                   for i, source in enumerate(result['sources'], 1):
                       title = source['title']
                       score = source['score']
                       category = source['category']
                       print(f"  {i}. {title} (score: {score:.3f}, category: {category})")

           except KeyboardInterrupt:
               print("\nGoodbye!")
               break
           except Exception as e:
               print(f"Error: {str(e)}")


Advanced Features
=================

Let's add some advanced features to make our RAG application more powerful:

Document Management
-------------------

.. code-block:: python

   def add_document_from_file(db: LocalVectorDB, file_path: str, category: str = "general") -> str:
       """
       Add a document from a text file.

       Args:
           db: The LocalVectorDB instance
           file_path: Path to the text file
           category: Category for the document

       Returns:
           The document ID if successful
       """
       try:
           with open(file_path, 'r', encoding='utf-8') as f:
               content = f.read()

           # Extract title from filename
           title = Path(file_path).stem.replace('_', ' ').title()

           doc_data = {
               'content': content,
               'title': title,
               'source': str(file_path),
               'category': category,
               'created_date': datetime.now().isoformat(),
               'word_count': len(content.split())
           }

           doc_ids = ingest_documents(db, [doc_data])
           return doc_ids[0] if doc_ids else None

       except Exception as e:
           print(f"Error adding document from {file_path}: {e}")
           return None

   def search_documents(db: LocalVectorDB, **filters) -> List[Dict[str, Any]]:
       """
       Search documents by metadata filters.

       Args:
           db: The LocalVectorDB instance
           **filters: Metadata filters (e.g., category='programming')

       Returns:
           List of matching documents
       """
       documents = db.filter(where=filters, limit=50)

       results = []
       for doc in documents:
           results.append({
               'id': doc.id,
               'title': doc.metadata.get('title', 'Untitled'),
               'category': doc.metadata.get('category', 'general'),
               'word_count': doc.metadata.get('word_count', 0),
               'created_date': doc.metadata.get('created_date', ''),
               'content_preview': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
           })

       return results

Conversation Memory
-------------------

.. code-block:: python

   class EnhancedRAGChatBot(RAGChatBot):
       """
       Enhanced RAG chatbot with conversation memory and context awareness.
       """

       def __init__(self, db: LocalVectorDB, chat_model: str = "llama3.2", max_history: int = 5):
           super().__init__(db, chat_model)
           self.max_history = max_history

       def get_conversation_context(self) -> str:
           """
           Build conversation context from recent history.

           Returns:
               Formatted conversation history
           """
           if not self.conversation_history:
               return ""

           recent_history = self.conversation_history[-self.max_history:]
           context_parts = []

           for entry in recent_history:
               context_parts.append(f"User: {entry['query']}")
               context_parts.append(f"Assistant: {entry['response']}")

           return "\n".join(context_parts)

       def generate_response_with_memory(
           self,
           user_query: str,
           context_results: List[Dict[str, Any]]
       ) -> str:
           """
           Generate response considering conversation history.

           Args:
               user_query: The user's question
               context_results: Relevant information from knowledge base

           Returns:
               The generated response
           """
           # Build knowledge context
           knowledge_context = "\n".join([
               f"[{result['title']}]\n{result['content']}\n"
               for result in context_results
           ])

           # Get conversation context
           conversation_context = self.get_conversation_context()

           # Create enhanced prompt
           prompt = f"""You are a helpful assistant with access to a knowledge base and conversation history.
   Use both the knowledge base and conversation context to provide relevant, coherent responses.

   Previous Conversation:
   {conversation_context}

   Knowledge Base Context:
   {knowledge_context}

   Current Question: {user_query}

   Provide a helpful response that considers both the knowledge base and conversation history:"""

           try:
               response = ollama.generate(
                   model=self.chat_model,
                   prompt=prompt,
                   options={
                       'temperature': 0.7,
                       'num_predict': 500,  # Ollama's name for max tokens to generate
                       'top_p': 0.9
                   }
               )

               return response['response'].strip()

           except Exception as e:
               return f"Sorry, I encountered an error: {str(e)}"

Running the Complete Application
================================

Here's how to tie everything together:

.. code-block:: python

   def main():
       """
       Main function to run the RAG chat application.
       """
       print("🚀 Starting LocalVectorDB RAG Chat Application")
       print("=" * 60)

       try:
           # Check if Ollama is available
           try:
               ollama.list()
               print("Ollama connection successful")
           except Exception as e:
               print(f"Ollama connection failed: {e}")
               print("Please ensure Ollama is running and accessible")
               return

           # Initialize database (reuse existing if available)
           print("Initializing knowledge base...")

           # Add more sample documents if database is empty
           if db.get_stats()['documents'] == 0:
               print("Adding sample documents...")
               ingested_ids = ingest_documents(db, sample_documents)
               print(f"Added {len(ingested_ids)} documents to knowledge base")

           # Show database stats
           stats = db.get_stats()
           print(f"Knowledge base ready: {stats['documents']} documents, {stats['chunks']} chunks")

           # Start chat interface
           run_chat_interface()

       except Exception as e:
           print(f"Application error: {e}")

       finally:
           # Clean up
           if 'db' in locals():
               db.close()
               print("Database connection closed")

   if __name__ == "__main__":
       main()


Complete Example Script
========================

Here's the complete script you can run:

.. code-block:: python

   #!/usr/bin/env python3
   """
   LocalVectorDB RAG Chat Application

   A complete example of building a RAG chatbot using LocalVectorDB and Ollama.
   """

   import logging
   import ollama
   from datetime import datetime
   from pathlib import Path
   from typing import List, Dict, Any, Optional

   from localvectordb import VectorDB, LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType

   # Configure logging
   logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

   # [Include all the class definitions and functions from above]

   if __name__ == "__main__":
       main()

Next Steps
==========

Now that you have a working RAG chatbot, here are some ideas for enhancement:

**Data Sources**
- Add support for PDF, Word, and other document formats
- Implement web scraping for dynamic content
- Connect to APIs for real-time data

**Search Improvements**
- Implement query expansion and reformulation
- Add semantic filtering and ranking
- Support for multi-modal search (text + images)

**User Interface**
- Build a web interface with Flask/FastAPI
- Add real-time streaming responses
- Implement user authentication and session management

**Performance Optimization**
- Implement caching for frequent queries
- Add async processing for large document batches
- Optimize embedding generation and storage

**Advanced Features**
- Add citation and source tracking
- Implement fact-checking and confidence scoring
- Support for multiple languages

Conclusion
==========

You've successfully built a complete RAG chat application using LocalVectorDB and Ollama! This tutorial covered:

- Setting up a vector database with proper metadata schema
- Implementing document ingestion and management
- Building a hybrid search system combining vector and keyword search
- Creating a conversational AI interface with context awareness
- Adding advanced features like conversation memory

The modular design makes it easy to extend and customize for your specific use cases. LocalVectorDB's document-first approach simplifies the complexity of managing embeddings and chunks, while Ollama provides powerful local AI capabilities.

Happy building! 🚀