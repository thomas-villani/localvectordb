========================
Metadata Filtering Guide
========================

Overview
--------

LocalVectorDB provides a powerful and secure metadata filtering system that allows you to query documents based on
their metadata fields. The filtering system supports MongoDB-style operators and generates safe parameterized
SQL queries to prevent injection attacks.

.. important::

   **Metadata fields must be declared in the database's** ``metadata_schema`` **to be stored and
   queryable.** Each declared field becomes a real column on the ``documents`` table. Any key you pass
   in a document's metadata that is **not** part of the schema is dropped on insert with a logged
   warning (it is not stored, and you cannot filter on it). Likewise, filtering on a field that is not
   in the schema raises an error. Declare every field you intend to store, filter, or sort on when you
   create the database (or add it later with ``update_metadata_schema()``).


Key Features
^^^^^^^^^^^^

* **Secure by design** - All queries use parameterized SQL
* **MongoDB-style operators** for intuitive querying
* **Logical operators** for complex conditions
* **Type-safe operations** with schema validation
* **JSON/Array support** for complex data structures
* **Performance optimized** with proper indexing
* **Multi-column embeddings** - Enable vector search on metadata fields
* **Full-text search** - FTS5 support for metadata fields

Metadata Field Attributes
-------------------------

When defining metadata fields, you can specify several attributes:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Attribute
     - Default
     - Description
   * - ``type``
     - Required
     - Field type: TEXT, INTEGER, REAL, BOOLEAN, DATE, or JSON
   * - ``indexed``
     - False
     - Create database index for faster filtering
   * - ``required``
     - False
     - Field must be provided when adding documents
   * - ``default_value``
     - None
     - Default value for optional fields
   * - ``embedding_enabled``
     - False
     - Generate embeddings for vector search (TEXT/JSON only)
   * - ``fts_enabled``
     - False
     - Enable full-text search with FTS5 (TEXT only)

Predefined Schema Templates
---------------------------

Instead of hand-writing a schema, you can start from one of several built-in
templates via :func:`~localvectordb.get_common_metadata_schemas`. Pass the
template name directly as the ``metadata_schema`` argument when creating a
database:

.. code-block:: python

   from localvectordb import VectorDB

   # Use the built-in "research_papers" schema (title, authors, abstract,
   # publication_date, journal, doi, keywords, citation_count)
   db = VectorDB("papers", "./storage", metadata_schema="research_papers")

The available templates are ``files``, ``documents``, ``research_papers``,
``code_repository``, and ``customer_support``. You can also inspect or extend a
template programmatically:

.. code-block:: python

   from localvectordb import get_common_metadata_schemas
   from localvectordb.core import MetadataField, MetadataFieldType

   schema = dict(get_common_metadata_schemas("documents"))   # a copy to extend
   schema["language"] = MetadataField(type=MetadataFieldType.TEXT, indexed=True)
   db = VectorDB("docs", "./storage", metadata_schema=schema)

Calling ``get_common_metadata_schemas()`` with no argument returns a dict of all
templates; an unknown name raises ``KeyError``.

Supported Operators
-------------------

Comparison Operators
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Operator
     - Description
   * - ``$eq``
     - Equal to (default for simple values)
   * - ``$ne``
     - Not equal to
   * - ``$gt``
     - Greater than
   * - ``$lt``
     - Less than
   * - ``$gte``
     - Greater than or equal to
   * - ``$lte``
     - Less than or equal to
   * - ``$in``
     - Value in list
   * - ``$nin``
     - Value not in list

String Operators
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Operator
     - Description
   * - ``$like``
     - SQL ``LIKE`` pattern matching. This uses SQLite's built-in ``LIKE``,
       which is case-insensitive for ASCII characters by default.
   * - ``$ilike``
     - Case-insensitive ``LIKE`` matching. Lowercases both the column and the
       value (``LOWER(field) LIKE ...``) for reliable case-insensitive
       matching regardless of SQLite's ``LIKE`` settings.
   * - ``$contains``
     - Contains substring
   * - ``$startswith``
     - Starts with substring
   * - ``$endswith``
     - Ends with substring

Existence Operators
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Operator
     - Description
   * - ``$exists``
     - Field exists and is not null
   * - ``$not_exists``
     - Field does not exist or is null

Type Operators
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Operator
     - Description
   * - ``$type``
     - Check field type: ``null``, ``string``, ``number``, ``integer``, ``real``, ``boolean``, ``array``, ``object``

Logical Operators
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Operator
     - Description
   * - ``$and``
     - All conditions must be true
   * - ``$or``
     - At least one condition must be true
   * - ``$not``
     - Condition must be false

JSON Operators
^^^^^^^^^^^^^^

For metadata fields with JSON type:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Operator
     - Description
   * - ``$contains``
     - JSON array contains value
   * - ``$not_contains``
     - JSON array does not contain value

Basic Usage Examples
--------------------

Simple Equality
^^^^^^^^^^^^^^^

Filter documents with exact field matches:

.. code-block:: python

   # Find documents by a specific author
   docs = db.filter(where={"author": "John Doe"})

   # Find published documents by John Doe from 2023
   docs = db.filter(where={
       "author": "John Doe",
       "year": 2023,
       "published": True
   })

Range Queries
^^^^^^^^^^^^^

Filter with numeric or date ranges:

.. code-block:: python

   # Find documents between 2020-2024 (inclusive) with a rating > 4.0
   docs = db.filter(where={
       "year": {"$gte": 2020, "$lte": 2024},
       "rating": {"$gt": 4.0}
   })

   # Find documents created in 2024 but updated before December 31st
   docs = db.filter(where={
       "created_at": {"$gte": "2024-01-01"},
       "updated_at": {"$lt": "2024-12-31"}
   })

String Operations
^^^^^^^^^^^^^^^^^

Search and match text content:

.. code-block:: python

   # Find documents with "python" in title, authors starting with "Dr.", and case-insensitive machine learning in description
   docs = db.filter(where={
       "title": {"$contains": "python"},
       "author": {"$startswith": "Dr."},
       "description": {"$ilike": "%machine learning%"}
   })

   # Find documents with company emails and tutorial categories
   docs = db.filter(where={
       "email": {"$like": "%@company.com"},
       "category": {"$endswith": "_tutorial"}
   })

List Operations
^^^^^^^^^^^^^^^

Work with lists and arrays:

.. code-block:: python

   # Find documents in tech, science, or education categories, excluding drafts and archived documents
   docs = db.filter(where={
       "category": {"$in": ["tech", "science", "education"]},
       "status": {"$nin": ["draft", "archived"]}
   })

   # Find documents with "python" in tags array but without "deprecated" in keywords
   docs = db.filter(where={
       "tags": {"$contains": "python"},  # JSON array contains "python"
       "keywords": {"$not_contains": "deprecated"}
   })

Advanced Usage Examples
-----------------------

Logical Operations
^^^^^^^^^^^^^^^^^^

Combine multiple conditions with logical operators:

.. code-block:: python

   # Find tech documents from 2020+ with rating >= 4.0 (explicit AND)
   docs = db.filter(where={
       "$and": [
           {"year": {"$gte": 2020}},
           {"rating": {"$gte": 4.0}},
           {"category": "tech"}
       ]
   })

   # Find documents by John Doe, Jane Smith, or in the featured category
   docs = db.filter(where={
       "$or": [
           {"author": "John Doe"},
           {"author": "Jane Smith"},
           {"category": "featured"}
       ]
   })

   # Find recent tech documents (by category OR programming tags) that are NOT archived
   docs = db.filter(where={
       "$and": [
           {"year": {"$gte": 2020}},
           {"$or": [
               {"category": "tech"},
               {"tags": {"$contains": "programming"}}
           ]},
           {"$not": {"status": "archived"}}
       ]
   })

Existence and Type Checking
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validate field presence and types:

.. code-block:: python

   # Find documents missing optional fields but having required fields
   docs = db.filter(where={
       "optional_field": {"$exists": False},
       "required_field": {"$exists": True}
   })

   # Find documents with specific data types (numeric ratings, array tags, object metadata, string descriptions)
   docs = db.filter(where={
       "rating": {"$type": "number"},
       "tags": {"$type": "array"},
       "metadata": {"$type": "object"},
       "description": {"$type": "string"}
   })

Ordering and Pagination
^^^^^^^^^^^^^^^^^^^^^^^

Sort and paginate filtered results:

.. code-block:: python

   # Find tech documents, order by newest first, get page 3 (items 21-30)
   docs = db.filter(
       where={"category": "tech"},
       order_by="created_at DESC",
       limit=10,
       offset=20
   )

   # Find recent documents with highest ratings first, limit to 50 results
   docs = db.filter(
       where={"year": {"$gte": 2020}},
       order_by="rating DESC",
       limit=50
   )

API Usage Examples
------------------

Local Database (LocalVectorDB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct usage with the local database:

.. code-block:: python

   from localvectordb import LocalVectorDB

   # Initialize database
   db = LocalVectorDB("my_database")

   # Find recent documents by John Doe or Jane Smith from 2020 onwards
   docs = db.filter(where={
       "author": {"$in": ["John Doe", "Jane Smith"]},
       "year": {"$gte": 2020}
   })

   # Find recent research documents: approved peer-reviewed papers OR highly cited papers, newest first
   docs = db.filter(
       where={
           "$and": [
               {"category": "research"},
               {"$or": [
                   {"peer_reviewed": True},
                   {"citation_count": {"$gte": 100}}
               ]}
           ]
       },
       order_by="publication_date DESC",
       limit=25
   )

Remote Database (RemoteVectorDB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Client usage for remote databases:

.. code-block:: python

   from localvectordb.client import RemoteVectorDB

   # Connect to remote database
   db = RemoteVectorDB(
       "my_database",
       "http://localhost:8000",
       api_key="your_api_key"
   )

   # Find high-quality tech documents in English or Spanish
   docs = db.filter(where={
       "$and": [
           {"category": "tech"},
           {"rating": {"$gte": 4.0}},
           {"language": {"$in": ["en", "es"]}}
       ]
   })

REST API Usage
^^^^^^^^^^^^^^

HTTP API examples using curl:

Simple Filtering
~~~~~~~~~~~~~~~~

Find documents by John Doe from 2020 onwards:

.. code-block:: bash

   curl -X POST "http://localhost:8000/api/v1/databases/my_database/filter" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "where": {
         "author": "John Doe",
         "year": {"$gte": 2020}
       }
     }'

Complex Filtering with Pagination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find high-quality tech/science documents that are featured OR trending, newest first:

.. code-block:: bash

   curl -X POST "http://localhost:8000/api/v1/databases/my_database/filter" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "where": {
         "$and": [
           {"category": {"$in": ["tech", "science"]}},
           {"rating": {"$gte": 4.0}},
           {"$or": [
             {"featured": true},
             {"trending": true}
           ]}
         ]
       },
       "order_by": "created_at DESC",
       "limit": 10,
       "offset": 0
     }'

String Search Operations
~~~~~~~~~~~~~~~~~~~~~~~~

Find machine learning articles by doctors with AI in the description:

.. code-block:: bash

   curl -X POST "http://localhost:8000/api/v1/databases/my_database/filter" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "where": {
         "title": {"$contains": "machine learning"},
         "author": {"$startswith": "Dr."},
         "description": {"$ilike": "%artificial intelligence%"}
       },
       "limit": 20
     }'

JSON Array Operations
~~~~~~~~~~~~~~~~~~~~~

Find Python documents without deprecated content, with specific skill requirements:

.. code-block:: bash

   curl -X POST "http://localhost:8000/api/v1/databases/my_database/filter" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "where": {
         "tags": {"$contains": "python"},
         "categories": {"$not_contains": "deprecated"},
         "skills_required": {"$in": [["python"], ["javascript"], ["python", "sql"]]}
       }
     }'

Search Integration
------------------

Enhanced filtering works seamlessly with search operations:

Vector Search with Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Search for machine learning tutorials: recent, tech/education category, intermediate difficulty or below
   results = db.query(
       "machine learning tutorial",
       search_type="vector",
       filters={
           "year": {"$gte": 2020},
           "category": {"$in": ["tech", "education"]},
           "difficulty": {"$lte": "intermediate"}
       }
   )

Hybrid Search with Complex Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Search for Python programming content: English language, beginner-friendly OR tutorial content, high quality
   results = db.query(
       "python programming",
       search_type="hybrid",
       filters={
           "$and": [
               {"language": "en"},
               {"$or": [
                   {"level": "beginner"},
                   {"tags": {"$contains": "tutorial"}}
               ]},
               {"rating": {"$gte": 4.0}}
           ]
       },
       vector_weight=0.5
   )

Search API with Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^

Search for recent NLP content in AI category OR with NLP tags:

.. code-block:: bash

   curl -X POST "http://localhost:8000/api/v1/databases/my_database/query" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key" \
     -d '{
       "query": "natural language processing",
       "search_type": "hybrid",
       "filters": {
         "$and": [
           {"year": {"$gte": 2020}},
           {"$or": [
             {"category": "ai"},
             {"tags": {"$contains": "nlp"}}
           ]}
         ]
       },
       "k": 10
     }'


Performance and Best Practices
------------------------------

Indexing Strategy
^^^^^^^^^^^^^^^^^

Create indexes on frequently filtered fields:

.. code-block:: python

   from localvectordb.core import MetadataField, MetadataFieldType

   # Define schema with indexed fields and optional embeddings
   metadata_schema = {
       "author": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       "title": MetadataField(type=MetadataFieldType.TEXT, indexed=True, 
                            embedding_enabled=True,  # Enable vector search on title
                            fts_enabled=True),        # Enable full-text search
       "abstract": MetadataField(type=MetadataFieldType.TEXT, 
                               embedding_enabled=True),  # Enable embeddings without indexing
       "year": MetadataField(type=MetadataFieldType.INTEGER, indexed=True),
       "category": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       "rating": MetadataField(type=MetadataFieldType.REAL, indexed=True),
       "tags": MetadataField(type=MetadataFieldType.JSON, indexed=False)  # JSON operations are slower
   }

Query Optimization Tips
^^^^^^^^^^^^^^^^^^^^^^^

1. **Use indexed fields** for frequently filtered columns
2. **Combine simple conditions** before using logical operators
3. **Place selective filters first** in ``$and`` operations
4. **Use appropriate operators** for data types
5. **Limit result sets** with ``limit`` parameter

.. code-block:: python

   # Good: Place selective filter first, then broader filter
   # Find documents by a specific author (selective), then filter by broader categories
   docs = db.filter(where={
       "$and": [
           {"author": "Specific Author"},      # Very selective
           {"category": {"$in": ["tech", "science"]}}  # Less selective
       ]
   })

   # Good: Use appropriate operators for data types
   # Find high-rated AI documents with boolean exact match
   docs = db.filter(where={
       "rating": {"$gte": 4.0},        # Numeric comparison
       "title": {"$contains": "AI"},   # String search
       "published": True               # Boolean exact match
   })

Performance Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   **JSON Operations**: ``$contains`` and ``$not_contains`` on JSON fields may be slower on large datasets. Consider restructuring data if these operations are frequent.

.. tip::
   **Complex Queries**: Test complex logical operations with small datasets first to understand performance characteristics.

Error Handling
--------------

The filtering system provides clear error messages for common issues:

Invalid Field Names
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   try:
       docs = db.filter(where={"invalid_field": "value"})
   except DatabaseError as e:
       print(f"Filter error: {e}")
       # Error: Field 'invalid_field' not found in metadata schema

Unsupported Operators
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   try:
       docs = db.filter(where={"author": {"$invalid": "value"}})
   except DatabaseError as e:
       print(f"Filter error: {e}")
       # Error: Unsupported operator: $invalid

Operator/Type Mismatches
^^^^^^^^^^^^^^^^^^^^^^^^^

Using a string operator such as ``$contains`` against a non-text field does not
raise; it falls back to a substring ``LIKE`` match on the value's text form,
which is rarely what you want. Prefer the numeric comparison operators for
numeric fields:

.. code-block:: python

   # Silently does a LIKE '%2020%' match — usually not intended:
   docs = db.filter(where={"year": {"$contains": "2020"}})

   # Use a numeric comparison instead:
   docs = db.filter(where={"year": 2020})
   docs = db.filter(where={"year": {"$gte": 2020, "$lt": 2025}})

Common Error Messages
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error Type
     - Description
   * - ``Field 'field_name' not found in metadata schema``
     - Field doesn't exist in the database schema
   * - ``Unsupported operator: $operator``
     - Operator not recognized or not supported
   * - ``Invalid field name: field_name``
     - Field name contains invalid characters
   * - ``Operator $op requires a list/array value``
     - Wrong value type for operator (e.g., ``$in`` with non-list)
   * - ``JSON operations only supported on JSON fields``
     - Attempting JSON operations on non-JSON field

Complete Examples
-----------------

Research Paper Database
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from localvectordb import LocalVectorDB
   from localvectordb.core import MetadataField, MetadataFieldType

   # Create database with research paper schema
   metadata_schema = {
       "title": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       "authors": MetadataField(type=MetadataFieldType.JSON),
       "year": MetadataField(type=MetadataFieldType.INTEGER, indexed=True),
       "journal": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       "citations": MetadataField(type=MetadataFieldType.INTEGER, indexed=True),
       "keywords": MetadataField(type=MetadataFieldType.JSON),
       "peer_reviewed": MetadataField(type=MetadataFieldType.BOOLEAN, indexed=True)
   }

   db = LocalVectorDB("research_papers", metadata_schema=metadata_schema)

   # Find highly cited machine learning papers from top journals (100+ citations, recent, peer-reviewed)
   papers = db.filter(where={
       "$and": [
           {"keywords": {"$contains": "machine learning"}},
           {"citations": {"$gte": 100}},
           {"journal": {"$in": ["Nature", "Science", "ICML", "NeurIPS"]}},
           {"year": {"$gte": 2020}},
           {"peer_reviewed": True}
       ]
   }, order_by="citations DESC", limit=20)

E-commerce Product Database
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Find affordable electronics from premium brands OR with wireless features, in stock, not discontinued
   product_filters = {
       "$and": [
           {"category": {"$in": ["electronics", "computers"]}},
           {"price": {"$gte": 100, "$lte": 1000}},
           {"rating": {"$gte": 4.0}},
           {"in_stock": True},
           {"$or": [
               {"brand": {"$in": ["Apple", "Samsung", "Sony"]}},
               {"features": {"$contains": "wireless"}}
           ]},
           {"$not": {"discontinued": True}}
       ]
   }

   products = db.filter(
       where=product_filters,
       order_by="rating DESC",
       limit=50
   )

Document Management System
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Find recent approved documents by key authors/departments OR urgent items, excluding confidential drafts
   documents = db.filter(where={
       "$and": [
           {"status": "approved"},
           {"created_at": {"$gte": "2024-01-01"}},
           {"$or": [
               {"author": {"$in": ["John Doe", "Jane Smith"]}},
               {"department": {"$in": ["engineering", "research"]}},
               {"tags": {"$contains": "urgent"}}
           ]},
           {"document_type": {"$nin": ["draft", "template"]}},
           {"confidential": {"$ne": True}}
       ]
   }, order_by="created_at DESC")
