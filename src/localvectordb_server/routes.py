# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb_server/routes.py
"""
localvectordb_server/routes.py - LocalVectorDB v1.0 API Routes

Updated API routes for LocalVectorDB v1.0 with document-first architecture,
unified query interface, and structured metadata support.
"""

import logging
from math import ceil
from typing import Dict, Any

from flask import Blueprint, request, jsonify, current_app
from werkzeug.exceptions import NotFound, BadRequest, Unauthorized, UnsupportedMediaType

from localvectordb.exceptions import (
    DatabaseNotFoundError, DuplicateDocumentIDError,
    EmbeddingError, BaseLocalVectorDBException
)
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.utils import get_system_version
from localvectordb_server._auth import require_api_key
from localvectordb_server._checkdeps import check_ollama_service
from localvectordb_server.config import DatabaseSettings, EmbeddingSettings
from localvectordb_server._cache import cache


logger = logging.getLogger(__name__)
api = Blueprint('api', __name__)


def serialize_document(doc) -> Dict[str, Any]:
    """Serialize a Document object for JSON response"""
    return {
        "id": doc.id,
        "content": doc.content,
        "metadata": doc.metadata,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        "content_hash": doc.content_hash
    }


def serialize_query_result(result) -> Dict[str, Any]:
    """Serialize a QueryResult object for JSON response"""
    data = {
        "id": result.id,
        "score": result.score,
        "type": result.type,
        "content": result.content,
        "metadata": result.metadata
    }

    # Add chunk-specific fields if applicable
    if result.type == 'chunk' and result.document_id:
        data["document_id"] = result.document_id

    if result.position:
        data["position"] = result.position.to_dict()

    # if result.highlights:
    #     data["highlights"] = result.highlights

    return data


def parse_metadata_schema(schema_data: Dict[str, Any]) -> Dict[str, MetadataField]:
    """Parse metadata schema from request data"""
    if not schema_data:
        return {}

    parsed_schema = {}
    for field_name, field_config in schema_data.items():
        if isinstance(field_config, str):
            # Simple string type
            field_type = MetadataFieldType(field_config)
            parsed_schema[field_name] = MetadataField(type=field_type)
        elif isinstance(field_config, dict):
            # Full field configuration
            field_type = MetadataFieldType(field_config.get('type', 'text'))
            parsed_schema[field_name] = MetadataField(
                type=field_type,
                indexed=field_config.get('indexed', False),
                required=field_config.get('required', False),
                default_value=field_config.get('default_value')
            )
        else:
            raise BadRequest(f"Invalid metadata field configuration for '{field_name}'")

    return parsed_schema


# Error Handlers
@api.errorhandler(BadRequest)
@api.errorhandler(NotFound)
def handle_error(error):
    """Handle common errors"""
    return jsonify({"error": str(error)}), error.code


@api.errorhandler(Unauthorized)
def handle_unauthorized_error(error):
    return jsonify({
        "error": str(error),
        "type": "authentication_error"
    }), error.code


@api.errorhandler(DatabaseNotFoundError)
def handle_database_not_found(error):
    return jsonify({"error": str(error), "type": "database_not_found"}), 404


@api.errorhandler(DuplicateDocumentIDError)
def handle_duplicate_document_id(error):
    return jsonify({"error": str(error), "type": "duplicate_document_id"}), 409


@api.errorhandler(EmbeddingError)
def handle_embedding_error(error):
    return jsonify({"error": str(error), "type": "embedding_error"}), 503


@api.errorhandler(BaseLocalVectorDBException)
def handle_database_error(error):
    return jsonify({"error": str(error), "type": "database_error"}), 500

@api.errorhandler(UnsupportedMediaType)
def handle_unsupported_media_type_error(error):
    return jsonify({"error": str(error), "type": "server_error"}), 415

@api.errorhandler(Exception)
def handle_unexpected_error(error):
    """Handle unexpected errors"""
    logger.exception("Unexpected error occurred")
    return jsonify({"error": f"An unexpected error occurred: {str(repr(error))}"}), 500


# Database Management Routes
@api.route("/api/v1/databases", methods=["POST"])
@require_api_key
@cache.cached()
def create_database():
    """Create a new vector database with optional metadata schema.

    Expected payload::

        {
            "name": "name-of-db",
            "metadata_schema": { ... },  // optional
            "database": {                // optional if different from default
                "chunking_method": "<chunk-method>",
                "chunk_size": 500, // integer,
                "enable_fts": true, // must be true for hybrid/keyword search to work
            },
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text",
                "
            }
        }
    """
    data = request.json if request.data else None
    if not data:
        raise BadRequest("No configuration provided")

    try:
        # Required parameters
        name = data.get("name")
        if not name:
            raise BadRequest("Database name is required")

        if hasattr(current_app, "config_obj"):
            db_config = current_app.config_obj.database.copy()
            embedding_config = current_app.config_obj.embedding.copy()
        else:
            db_config = DatabaseSettings()
            embedding_config = EmbeddingSettings()

        # Parse metadata schema if provided
        metadata_schema = None
        if "metadata_schema" in data:
            metadata_schema = parse_metadata_schema(data.get("metadata_schema"))
        else:
            metadata_schema = db_config.default_metadata_schema

        if "database" in data:
                database_settings = data.get("database")
                db_config.update_from_dict(database_settings)

        if "embedding" in data:
            embedding_settings = data.get("embedding")
            embedding_config.update_from_dict(embedding_settings)

        db = current_app.db_manager.create_db(name, metadata_schema, db_config, embedding_config)

        # Store in database manager
        current_app.db_manager.databases[name] = db

        return jsonify({
            "message": f"Successfully created database '{name}'",
            "status": "success",
            "config": {
                "name": db.name,
                "embedding_provider": db.embedding_provider.provider_name,
                "embedding_model": db.embedding_provider.model,
                "embedding_dimension": db.embedding_dimension,
                "chunking_method": db.chunking_method,
                "chunk_size": db.chunk_size,
                "chunk_overlap": db.chunk_overlap,
                "metadata_schema": {
                    field_name: {
                        "type": field.type.value,
                        "indexed": field.indexed,
                        "required": field.required,
                        "default_value": field.default_value
                    }
                    for field_name, field in (db.metadata_schema or {}).items()
                },
                "fts_enabled": db.fts_enabled
            }
        })

    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise


@api.route("/api/v1/databases", methods=["GET"])
@require_api_key
@cache.cached()
def list_databases():
    """List all available databases"""
    try:
        databases = current_app.db_manager.list_databases()
        return jsonify({
            "databases": databases,
            "count": len(databases)
        })
    except Exception as e:
        logger.error(f"Error listing databases: {e}")
        raise


@api.route("/api/v1/<db_name>/info", methods=["GET"])
@require_api_key
@cache.cached()
def get_database_info(db_name):
    """Get information about a specific database"""
    try:
        db = current_app.db_manager.get_db(db_name)

        stats = db.stats

        return jsonify({
            "name": db.name,
            "stats": stats,
            "config": {
                "embedding_provider": db.embedding_provider.provider_name,
                "embedding_model": db.embedding_provider.model,
                "embedding_dimension": db.embedding_dimension,
                "chunking_method": db.chunking_method,
                "chunk_size": db.chunk_size,
                "chunk_overlap": db.chunk_overlap,
                "metadata_schema": {
                    field_name: {
                        "type": field.type.value,
                        "indexed": field.indexed,
                        "required": field.required,
                        "default_value": field.default_value
                    }
                    for field_name, field in (db.metadata_schema or {}).items()
                },
                "fts_enabled": db.fts_enabled
            }
        })

    except Exception as e:
        logger.error(f"Error getting database info for {db_name}: {e}")
        raise e


@api.route("/api/v1/<db_name>", methods=["DELETE"])
@require_api_key
@cache.cached()
def delete_database(db_name):
    """Delete a database"""
    try:
        import os
        from pathlib import Path

        # Close the database connection if it's open
        if db_name in current_app.db_manager.databases:
            db = current_app.db_manager.databases[db_name]
            db.close()
            del current_app.db_manager.databases[db_name]

        # Get the database path from the app config
        db_path = Path(current_app.config.get("DB_ROOT_DIR", ".lvdb"))

        # Check if database exists (look for the .sqlite file)
        db_sqlite_file = db_path / f"{db_name}.sqlite"
        db_faiss_file = db_path / f"{db_name}.faiss"

        if not db_sqlite_file.exists():
            raise NotFound(f"Database '{db_name}' not found")

        # Delete the database files
        if db_sqlite_file.exists():
            os.remove(db_sqlite_file)

        if db_faiss_file.exists():
            os.remove(db_faiss_file)

        return jsonify({
            "message": f"Successfully deleted database '{db_name}'",
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error deleting database: {e}")
        raise


# Document Management Routes
@api.route("/api/v1/<db_name>/documents", methods=["POST"])
@require_api_key
@cache.cached()
def upsert_documents(db_name):
    """Upsert (insert or update) documents"""
    data = request.json if request.data else None
    if not data:
        raise BadRequest("No data provided")

    try:
        documents = data.get("documents")
        if not documents:
            raise BadRequest("No documents provided")

        # Convert to list if single document
        if isinstance(documents, str):
            documents = [documents]

        metadata = data.get("metadata", data.get("metadatas"))
        if isinstance(metadata, dict):
            metadata = [metadata]

        ids = data.get("ids")
        if isinstance(ids, str):
            ids = [ids]

        batch_size = data.get("batch_size", 100)

        db = current_app.db_manager.get_db(db_name)
        result_ids = db.upsert(
            documents=documents,
            metadata=metadata,
            ids=ids,
            batch_size=batch_size
        )

        return jsonify({
            "message": f"Successfully processed {len(documents)} documents",
            "ids": result_ids,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error upserting documents: {e}")
        raise


@api.route("/api/v1/<db_name>/documents/insert", methods=["POST"])
@require_api_key
@cache.cached()
def insert_documents(db_name):
    """Insert new documents (fails if ID already exists)"""
    data = request.json if request.data else None
    if not data:
        raise BadRequest("No data provided")

    try:
        documents = data.get("documents")
        if not documents:
            raise BadRequest("No documents provided")

        # Convert to list if single document
        if isinstance(documents, str):
            documents = [documents]

        metadata = data.get("metadata", data.get("metadatas"))
        if isinstance(metadata, dict):
            metadata = [metadata]

        ids = data.get("ids")
        if isinstance(ids, str):
            ids = [ids]

        batch_size = data.get("batch_size", 100)
        errors = data.get("errors", "raise")  # "raise" or "ignore"
        similarity_threshold = data.get("similarity_threshold")

        db = current_app.db_manager.get_db(db_name)
        result_ids = db.insert(
            documents=documents,
            metadata=metadata,
            ids=ids,
            batch_size=batch_size,
            errors=errors,
            similarity_threshold=similarity_threshold
        )

        return jsonify({
            "message": f"Successfully inserted {len(result_ids)} documents",
            "ids": result_ids,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error inserting documents: {e}")
        raise


@api.route("/api/v1/<db_name>/documents/<doc_id>", methods=["GET"])
@require_api_key
@cache.cached()
def get_document(db_name, doc_id):
    """Get a document by ID"""
    try:
        db = current_app.db_manager.get_db(db_name)
        doc = db.get(doc_id)

        if doc is None:
            raise NotFound(f"Document {doc_id} not found")

        return jsonify(serialize_document(doc))

    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise


@api.route("/api/v1/<db_name>/documents/<doc_id>", methods=["PUT"])
@require_api_key
@cache.cached()
def update_document(db_name, doc_id):
    """Update a document's content and/or metadata"""
    try:
        data = request.json
        if not data:
            raise BadRequest("No data provided")

        content = data.get("content")
        metadata = data.get("metadata")

        if not content and not metadata:
            raise BadRequest("Either content or metadata must be provided")

        db = current_app.db_manager.get_db(db_name)
        was_updated = db.update(doc_id, content=content, metadata=metadata)

        if not was_updated:
            raise NotFound(f"Document {doc_id} not found")

        return jsonify({
            "message": f"Successfully updated document {doc_id}",
            "status": "success",
            "updated": was_updated
        })

    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise


@api.route("/api/v1/<db_name>/documents/<doc_id>", methods=["DELETE"])
@require_api_key
@cache.cached()
def delete_document(db_name, doc_id):
    """Delete a document by ID"""
    try:
        db = current_app.db_manager.get_db(db_name)

        if not db.exists(doc_id):
            raise NotFound(f"Document {doc_id} not found")

        deleted_count = db.delete(doc_id)

        return jsonify({
            "message": f"Successfully deleted document {doc_id}",
            "status": "success",
            "deleted_count": deleted_count
        })

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise


@api.route("/api/v1/<db_name>/documents/exists", methods=["POST"])
@require_api_key
@cache.cached()
def check_documents_exist(db_name):
    """Check if documents exist by ID"""
    data = request.json
    if not data:
        raise BadRequest("JSON request body is required")

    ids = data.get("ids")
    if not ids:
        raise BadRequest("`ids` is required")

    try:
        db = current_app.db_manager.get_db(db_name)
        exists = db.exists(ids)

        return jsonify({
            "exists": exists,
            "ids": ids if isinstance(ids, list) else [ids]
        })

    except Exception as e:
        logger.error(f"Error checking document existence: {e}")
        raise


@api.route("/api/v1/<db_name>/documents", methods=["GET"])
@require_api_key
@cache.cached()
def list_documents(db_name):
    """List documents with pagination and filtering"""
    try:
        # Get query parameters
        page = max(1, int(request.args.get('page', 1)))
        limit = max(1, min(1000, int(request.args.get('limit', 100))))

        # Simple filtering parameters
        filters = {}
        for key, value in request.args.items():
            if key not in ['page', 'limit']:
                filters[key] = value

        db = current_app.db_manager.get_db(db_name)

        # Calculate offset
        offset = (page - 1) * limit

        # Get filtered documents
        documents = db.filter(
            where=filters if filters else None,
            limit=limit,
            offset=offset
        )

        # Get total count (this is inefficient, but works for now)
        total_count = len(db.filter(where=filters if filters else None))
        total_pages = ceil(total_count / limit)

        # Serialize documents
        serialized_docs = [serialize_document(doc) for doc in documents]

        return jsonify({
            "documents": serialized_docs,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "page_size": limit,
                "total_count": total_count,
                "has_previous": page > 1,
                "has_next": page < total_pages
            }
        })

    except ValueError:
        raise BadRequest("`page` and `limit` must be valid integers")
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise


def search_handler(db_name, search_params):
    """Unified query interface for all search types"""
    if not search_params:
        raise BadRequest("No query provided")

    query_text = search_params.get("query")
    if not query_text:
        raise BadRequest("No query text provided")

    # Search parameters
    search_type = search_params.get("search_type", "vector")  # vector, keyword, hybrid
    return_type = search_params.get("return_type", "documents")  # documents, chunks
    k = search_params.get("k", 10)
    score_threshold = search_params.get("score_threshold", 0.0)
    filters = search_params.get("filters", search_params.get("metadata_filters"))
    vector_weight = search_params.get("vector_weight", 0.7)  # For hybrid search

    db = current_app.db_manager.get_db(db_name)
    results = db.query(
        query=query_text,
        search_type=search_type,
        return_type=return_type,
        k=k,
        score_threshold=score_threshold,
        filters=filters,
        vector_weight=vector_weight
    )

    # Serialize results
    serialized_results = [serialize_query_result(result) for result in results]

    return jsonify({
        "results": serialized_results,
        "search_type": search_type,
        "return_type": return_type,
        "total_results": len(serialized_results)
    })



# Search Routes
@api.route("/api/v1/<db_name>/query", methods=["POST"])
@require_api_key
@cache.cached()
def query_documents(db_name):
    """Unified query interface for all search types"""
    return search_handler(db_name, request.json)


@api.route("/api/v1/<db_name>/search/vector", methods=["POST"])
@require_api_key
@cache.cached()
def vector_search(db_name):
    """Vector similarity search (convenience endpoint)"""
    data = request.json
    if not data:
        raise BadRequest("No query provided")

    # Add search_type to data
    data["search_type"] = "vector"

    return search_handler(db_name, data)


@api.route("/api/v1/<db_name>/search/keyword", methods=["POST"])
@require_api_key
@cache.cached()
def keyword_search(db_name):
    """Keyword search (convenience endpoint)"""
    data = request.json
    if not data:
        raise BadRequest("No query provided")

    # Add search_type to data
    data["search_type"] = "keyword"

    return search_handler(db_name, data)


@api.route("/api/v1/<db_name>/search/hybrid", methods=["POST"])
@require_api_key
@cache.cached()
def hybrid_search(db_name):
    """Hybrid search (convenience endpoint)"""
    data = request.json
    if not data:
        raise BadRequest("No query provided")

    # Add search_type to data
    data["search_type"] = "hybrid"

    return search_handler(db_name, data)


# Filter and Metadata Routes
@api.route("/api/v1/<db_name>/filter", methods=["POST"])
@require_api_key
@cache.cached()
def filter_documents(db_name):
    """Filter documents by metadata and other criteria"""
    data = request.json
    if not data:
        raise BadRequest("No filter criteria provided")

    try:
        # Filter parameters
        where = data.get("where")
        sql = data.get("sql")
        order_by = data.get("order_by")
        limit = data.get("limit")
        offset = data.get("offset", 0)

        if not where and not sql:
            raise BadRequest("Either 'where' or 'sql' must be provided")

        db = current_app.db_manager.get_db(db_name)
        documents = db.filter(
            where=where,
            sql=sql,
            order_by=order_by,
            limit=limit,
            offset=offset
        )

        # Serialize results
        serialized_docs = [serialize_document(doc) for doc in documents]

        return jsonify({
            "documents": serialized_docs,
            "count": len(serialized_docs)
        })

    except Exception as e:
        logger.error(f"Error filtering documents: {e}")
        raise


# Global Search Route
@api.route("/api/v1/search", methods=["POST"])
@require_api_key
@cache.cached()
def global_search():
    """Search across multiple databases"""
    try:
        data = request.json
        if not data:
            raise BadRequest("No query provided")

        query = data.get("query")
        if not query:
            raise BadRequest("No query text provided")

        # Get search parameters
        search_type = data.get("search_type", "vector")
        return_type = data.get("return_type", "documents")
        k = data.get("k", 10)
        score_threshold = data.get("score_threshold", 0.0)
        filters = data.get("filters")
        databases = data.get("databases")  # Optional list of databases to search
        vector_weight = data.get("vector_weight", 0.7)

        # Get databases to search
        if databases is None:
            databases = current_app.db_manager.list_databases()

        # Search each database
        results = {}
        for db_name in databases:
            try:
                db = current_app.db_manager.get_db(db_name)
                db_results = db.query(
                    query=query,
                    search_type=search_type,
                    return_type=return_type,
                    k=k,
                    score_threshold=score_threshold,
                    filters=filters,
                    vector_weight=vector_weight
                )

                # Serialize results
                results[db_name] = [serialize_query_result(result) for result in db_results]

            except Exception as e:
                logger.error(f"Error searching database {db_name}: {e}")
                results[db_name] = f"Search failed: {str(e)}"

        return jsonify({
            "results": results,
            "search_type": search_type,
            "return_type": return_type
        })

    except Exception as e:
        logger.error(f"Error in global search: {e}")
        raise


# Health and System Routes
@api.route("/api/v1/health", methods=["GET"])
def health_check():
    """System health check endpoint"""
    status = {
        "status": "healthy",
        "version": get_system_version(),
        "ollama_available": check_ollama_service()
    }
    return jsonify(status)


# Embedding Routes
@api.route("/api/v1/<db_name>/embeddings", methods=["POST"])
@require_api_key
@cache.cached()
def get_embeddings_for_db(db_name):
    """Get embeddings using the database's embedding provider"""
    data = request.json
    if not data:
        raise BadRequest("Missing request body")

    texts = data.get("texts")
    if not texts:
        raise BadRequest("`texts` required")

    try:
        db = current_app.db_manager.get_db(db_name)

        if isinstance(texts, str):
            texts = [texts]

        embeddings = db.embedding_provider.embed_sync(texts)

        return jsonify({"embeddings": embeddings.tolist()})

    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise


@api.route("/api/v1/embeddings", methods=["POST"])
@require_api_key
@cache.cached()
def get_embeddings():
    """Get embeddings from specified provider and model"""
    data = request.json
    if not data:
        raise BadRequest("Missing request body")

    texts = data.get("texts")
    provider = data.get("provider")
    model = data.get("model")

    if not texts:
        raise BadRequest("`texts` required")
    if not provider:
        raise BadRequest("`provider` required")
    if not model:
        raise BadRequest("`model` required")

    try:
        from localvectordb.embeddings import EmbeddingRegistry

        embedding_provider = EmbeddingRegistry.create_provider(provider, model)

        if isinstance(texts, str):
            texts = [texts]

        embeddings = embedding_provider.embed_sync(texts)

        return jsonify({"embeddings": embeddings.tolist()})

    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise