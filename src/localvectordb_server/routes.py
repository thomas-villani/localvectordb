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

Enhanced API routes with structured logging, comprehensive error handling,
input validation, and performance monitoring.
"""

import logging
from math import ceil
from typing import Dict, Any

from flask import Blueprint, request, jsonify, current_app
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.utils import get_system_version
from localvectordb_server._auth import require_api_key
from localvectordb_server._checkdeps import check_ollama_service
from localvectordb_server.config import DatabaseSettings, EmbeddingSettings
from localvectordb_server._cache import cache
from localvectordb_server._error_handlers import (
    handle_errors, APIError, ValidationError,
    validate_required_fields, validate_field_type, validate_pagination_params,
    validate_search_params, validate_database_creation_params
)
from localvectordb_server._logcfg import log_performance, request_context, DatabaseLogger

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
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

    return data


def parse_metadata_schema(schema_data: Dict[str, Any]) -> Dict[str, MetadataField]:
    """Parse metadata schema from request data with validation"""
    if not schema_data:
        return {}

    if not isinstance(schema_data, dict):
        raise ValidationError("Metadata schema must be an object", field="metadata_schema")

    parsed_schema = {}
    for field_name, field_config in schema_data.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValidationError(
                "Metadata field names must be non-empty strings",
                field=f"metadata_schema.{field_name}"
            )

        try:
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
                raise ValidationError(
                    f"Invalid metadata field configuration for '{field_name}'. Must be string or object.",
                    field=f"metadata_schema.{field_name}"
                )
        except ValueError as e:
            raise ValidationError(
                f"Invalid metadata field type for '{field_name}': {str(e)}",
                field=f"metadata_schema.{field_name}"
            )

    return parsed_schema


# Database Management Routes
@api.route("/api/v1/databases", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("create_database")
def create_database():
    """Create a new vector database with optional metadata schema."""

    with request_context("create_database"):
        # Validate request has JSON body
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate input parameters
        data = validate_database_creation_params(data)

        name = data["name"]

        # Check if database already exists
        existing_dbs = current_app.db_manager.list_databases()
        if name in existing_dbs:
            raise APIError(
                message=f"Database '{name}' already exists",
                error_code="DATABASE_ALREADY_EXISTS",
                status_code=409,
                recoverable=True
            )

        # Get configuration with defaults
        if hasattr(current_app, "config_obj"):
            db_config = current_app.config_obj.database.copy()
            embedding_config = current_app.config_obj.embedding.copy()
        else:
            db_config = DatabaseSettings()
            embedding_config = EmbeddingSettings()

        # Parse metadata schema if provided
        metadata_schema = None
        if "metadata_schema" in data:
            metadata_schema = parse_metadata_schema(data["metadata_schema"])
        else:
            metadata_schema = db_config.default_metadata_schema

        # Update configurations from request
        if "database" in data:
            database_settings = data["database"]
            try:
                db_config.update_from_dict(database_settings)
            except Exception as e:
                raise ValidationError(f"Invalid database configuration: {str(e)}")

        if "embedding" in data:
            embedding_settings = data["embedding"]
            try:
                embedding_config.update_from_dict(embedding_settings)
            except Exception as e:
                raise ValidationError(f"Invalid embedding configuration: {str(e)}")

        # Create database
        db_logger.log_query("create_database", database_name=name)

        try:
            db = current_app.db_manager.create_db(name, metadata_schema, db_config, embedding_config)

            db_logger.log_query("create_database_success", database_name=name)

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
            db_logger.log_error("create_database", e, database_name=name)
            raise


@api.route("/api/v1/databases", methods=["GET"])
@require_api_key
@handle_errors
@cache.cached(timeout=60)  # Cache for 1 minute
@log_performance("list_databases")
def list_databases():
    """List all available databases"""

    with request_context("list_databases"):
        try:
            databases = current_app.db_manager.list_databases()
            return jsonify({
                "databases": databases,
                "count": len(databases)
            })

        except Exception as e:
            db_logger.log_error("list_databases", e)
            raise


@api.route("/api/v1/<db_name>/info", methods=["GET"])
@require_api_key
@handle_errors
@cache.cached(timeout=300)  # Cache for 5 minutes
@log_performance("get_database_info")
def get_database_info(db_name):
    """Get information about a specific database"""

    with request_context("get_database_info"):
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
            db_logger.log_error("get_database_info", e, database_name=db_name)
            raise


@api.route("/api/v1/<db_name>", methods=["DELETE"])
@require_api_key
@handle_errors
@log_performance("delete_database")
def delete_database(db_name):
    """Delete a database"""

    with request_context("delete_database"):

        success, errors = current_app.db_manager.delete_db(db_name)
        if not success:
            raise APIError(errors, "DATABASE_DELETE_ERROR", 500, {"db_name": db_name}, True)

        return jsonify({
            "message": f"Successfully deleted database '{db_name}'",
            "status": "success",
        })


# Document Management Routes
@api.route("/api/v1/<db_name>/documents", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("upsert_documents")
def upsert_documents(db_name):
    """Upsert (insert or update) documents"""

    with request_context("upsert_documents"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate required fields
        validate_required_fields(data, ['documents'])

        documents = data["documents"]
        if not documents:
            raise ValidationError("Documents array cannot be empty", field="documents")

        # Convert to list if single document
        if isinstance(documents, str):
            documents = [documents]
        elif not isinstance(documents, list):
            raise ValidationError("Documents must be a string or array of strings", field="documents")

        # Validate document content
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise ValidationError(f"Document at index {i} must be a string", field=f"documents[{i}]")
            if not doc.strip():
                raise ValidationError(f"Document at index {i} cannot be empty", field=f"documents[{i}]")

        # Handle metadata
        metadata = data.get("metadata", data.get("metadatas"))
        if metadata is not None:
            if isinstance(metadata, dict):
                metadata = [metadata]
            elif not isinstance(metadata, list):
                raise ValidationError("Metadata must be an object or array of objects", field="metadata")

            if len(metadata) != len(documents):
                raise ValidationError(
                    f"Number of metadata entries ({len(metadata)}) must match number of documents ({len(documents)})",
                    field="metadata"
                )

        # Handle IDs
        ids = data.get("ids")
        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            elif not isinstance(ids, list):
                raise ValidationError("IDs must be a string or array of strings", field="ids")

            if len(ids) != len(documents):
                raise ValidationError(
                    f"Number of IDs ({len(ids)}) must match number of documents ({len(documents)})",
                    field="ids"
                )

        # Validate batch size
        batch_size = data.get("batch_size", 100)
        validate_field_type(data, "batch_size", int)
        if batch_size < 1 or batch_size > 1000:
            raise ValidationError("Batch size must be between 1 and 1000", field="batch_size", value=batch_size)

        try:
            db = current_app.db_manager.get_db(db_name)

            db_logger.log_query("upsert_documents",
                                database_name=db_name,
                                document_count=len(documents),
                                batch_size=batch_size)

            result_ids = db.upsert(
                documents=documents,
                metadata=metadata,
                ids=ids,
                batch_size=batch_size
            )

            db_logger.log_query("upsert_documents_success",
                                database_name=db_name,
                                result_count=len(result_ids))

            return jsonify({
                "message": f"Successfully processed {len(documents)} documents",
                "ids": result_ids,
                "status": "success"
            })

        except Exception as e:
            db_logger.log_error("upsert_documents", e, database_name=db_name, document_count=len(documents))
            raise


@api.route("/api/v1/<db_name>/documents/insert", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("insert_documents")
def insert_documents(db_name):
    """Insert new documents (fails if ID already exists)"""

    with request_context("insert_documents"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Similar validation as upsert_documents...
        validate_required_fields(data, ['documents'])

        documents = data["documents"]
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

        # Validate parameters
        if errors not in ["raise", "ignore"]:
            raise ValidationError("Errors parameter must be 'raise' or 'ignore'", field="errors", value=errors)

        if similarity_threshold is not None:
            validate_field_type(data, "similarity_threshold", (int, float))
            if similarity_threshold < 0 or similarity_threshold > 1:
                raise ValidationError(
                    "Similarity threshold must be between 0 and 1",
                    field="similarity_threshold",
                    value=similarity_threshold
                )

        try:
            db = current_app.db_manager.get_db(db_name)

            db_logger.log_query("insert_documents",
                                database_name=db_name,
                                document_count=len(documents),
                                similarity_threshold=similarity_threshold)

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
            db_logger.log_error("insert_documents", e, database_name=db_name)
            raise


@api.route("/api/v1/<db_name>/documents/<doc_id>", methods=["GET"])
@require_api_key
@handle_errors
@cache.cached(timeout=300)
@log_performance("get_document")
def get_document(db_name, doc_id):
    """Get a document by ID"""

    with request_context("get_document"):
        try:
            db = current_app.db_manager.get_db(db_name)
            doc = db.get(doc_id)

            if doc is None:
                raise APIError(
                    message=f"Document '{doc_id}' not found in database '{db_name}'",
                    error_code="DOCUMENT_NOT_FOUND",
                    status_code=404,
                    recoverable=True
                )

            return jsonify(serialize_document(doc))

        except Exception as e:
            db_logger.log_error("get_document", e, database_name=db_name, document_id=doc_id)
            raise


@api.route("/api/v1/<db_name>/documents/<doc_id>", methods=["PUT"])
@require_api_key
@handle_errors
@log_performance("update_document")
def update_document(db_name, doc_id):
    """Update a document's content and/or metadata"""

    with request_context("update_document"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        content = data.get("content")
        metadata = data.get("metadata")

        if not content and not metadata:
            raise ValidationError("Either content or metadata must be provided")

        if content is not None and not isinstance(content, str):
            raise ValidationError("Content must be a string", field="content")

        if metadata is not None and not isinstance(metadata, dict):
            raise ValidationError("Metadata must be an object", field="metadata")

        try:
            db = current_app.db_manager.get_db(db_name)

            db_logger.log_query("update_document",
                                database_name=db_name,
                                document_id=doc_id,
                                has_content=content is not None,
                                has_metadata=metadata is not None)

            was_updated = db.update(doc_id, content=content, metadata=metadata)

            if not was_updated:
                raise APIError(
                    message=f"Document '{doc_id}' not found in database '{db_name}'",
                    error_code="DOCUMENT_NOT_FOUND",
                    status_code=404,
                    recoverable=True
                )

            return jsonify({
                "message": f"Successfully updated document {doc_id}",
                "status": "success",
                "updated": was_updated
            })

        except Exception as e:
            db_logger.log_error("update_document", e, database_name=db_name, document_id=doc_id)
            raise


@api.route("/api/v1/<db_name>/documents/<doc_id>", methods=["DELETE"])
@require_api_key
@handle_errors
@log_performance("delete_document")
def delete_document(db_name, doc_id):
    """Delete a document by ID"""

    with request_context("delete_document"):
        try:
            db = current_app.db_manager.get_db(db_name)

            if not db.exists(doc_id):
                raise APIError(
                    message=f"Document '{doc_id}' not found in database '{db_name}'",
                    error_code="DOCUMENT_NOT_FOUND",
                    status_code=404,
                    recoverable=True
                )

            deleted_count = db.delete(doc_id)

            db_logger.log_query("delete_document_success",
                                database_name=db_name,
                                document_id=doc_id,
                                deleted_count=deleted_count)

            return jsonify({
                "message": f"Successfully deleted document {doc_id}",
                "status": "success",
                "deleted_count": deleted_count
            })

        except Exception as e:
            db_logger.log_error("delete_document", e, database_name=db_name, document_id=doc_id)
            raise


@api.route("/api/v1/<db_name>/documents/exists", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("check_documents_exist")
def check_documents_exist(db_name):
    """Check if documents exist by ID"""

    with request_context("check_documents_exist"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ['ids'])

        ids = data["ids"]
        if isinstance(ids, str):
            ids = [ids]
        elif not isinstance(ids, list):
            raise ValidationError("IDs must be a string or array of strings", field="ids")

        try:
            db = current_app.db_manager.get_db(db_name)
            exists = db.exists(ids)

            return jsonify({
                "exists": exists,
                "ids": ids
            })

        except Exception as e:
            db_logger.log_error("check_documents_exist", e, database_name=db_name)
            raise


@api.route("/api/v1/<db_name>/documents", methods=["GET"])
@require_api_key
@handle_errors
@cache.cached(timeout=60)
@log_performance("list_documents")
def list_documents(db_name):
    """List documents with pagination and filtering"""

    with request_context("list_documents"):
        try:
            # Validate pagination parameters
            page, limit = validate_pagination_params(
                request.args.get('page'),
                request.args.get('limit')
            )

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

        except Exception as e:
            db_logger.log_error("list_documents", e, database_name=db_name)
            raise


def search_handler(db_name, search_params):
    """Unified query interface for all search types with enhanced validation"""

    # Validate search parameters
    search_params = validate_search_params(search_params)

    query_text = search_params["query"]
    search_type = search_params.get("search_type", "vector")
    return_type = search_params.get("return_type", "documents")
    k = search_params.get("k", 10)
    score_threshold = search_params.get("score_threshold", 0.0)
    filters = search_params.get("filters", search_params.get("metadata_filters"))
    vector_weight = search_params.get("vector_weight", 0.7)

    try:
        db = current_app.db_manager.get_db(db_name)

        db_logger.log_query("search",
                            database_name=db_name,
                            search_type=search_type,
                            return_type=return_type,
                            k=k,
                            query_length=len(query_text))

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

        db_logger.log_query("search_success",
                            database_name=db_name,
                            search_type=search_type,
                            result_count=len(serialized_results))

        return jsonify({
            "results": serialized_results,
            "search_type": search_type,
            "return_type": return_type,
            "total_results": len(serialized_results)
        })

    except Exception as e:
        db_logger.log_error("search", e, database_name=db_name, search_type=search_type)
        raise


# Search Routes
@api.route("/api/v1/<db_name>/query", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("query_documents")
def query_documents(db_name):
    """Unified query interface for all search types"""

    with request_context("query_documents"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        return search_handler(db_name, data)


@api.route("/api/v1/<db_name>/search/vector", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("vector_search")
def vector_search(db_name):
    """Vector similarity search (convenience endpoint)"""

    with request_context("vector_search"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Add search_type to data
        data["search_type"] = "vector"
        return search_handler(db_name, data)


@api.route("/api/v1/<db_name>/search/keyword", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("keyword_search")
def keyword_search(db_name):
    """Keyword search (convenience endpoint)"""

    with request_context("keyword_search"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Add search_type to data
        data["search_type"] = "keyword"
        return search_handler(db_name, data)


@api.route("/api/v1/<db_name>/search/hybrid", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("hybrid_search")
def hybrid_search(db_name):
    """Hybrid search (convenience endpoint)"""

    with request_context("hybrid_search"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Add search_type to data
        data["search_type"] = "hybrid"
        return search_handler(db_name, data)


# Filter and Metadata Routes
@api.route("/api/v1/<db_name>/filter", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("filter_documents")
def filter_documents(db_name):
    """Filter documents by metadata and other criteria"""

    with request_context("filter_documents"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        where = data.get("where")
        sql = data.get("sql")
        order_by = data.get("order_by")
        limit = data.get("limit")
        offset = data.get("offset", 0)

        if not where and not sql:
            raise ValidationError("Either 'where' or 'sql' must be provided")

        # Validate types
        if where is not None:
            validate_field_type(data, "where", dict)
        if sql is not None:
            validate_field_type(data, "sql", str)
        if order_by is not None:
            validate_field_type(data, "order_by", str)
        if limit is not None:
            validate_field_type(data, "limit", int)
            if limit < 1 or limit > 10000:
                raise ValidationError("Limit must be between 1 and 10000", field="limit", value=limit)

        validate_field_type(data, "offset", int)
        if offset < 0:
            raise ValidationError("Offset must be >= 0", field="offset", value=offset)

        try:
            db = current_app.db_manager.get_db(db_name)

            db_logger.log_query("filter_documents",
                                database_name=db_name,
                                has_where=where is not None,
                                has_sql=sql is not None,
                                limit=limit)

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
            db_logger.log_error("filter_documents", e, database_name=db_name)
            raise


# Global Search Route
@api.route("/api/v1/search", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("global_search")
def global_search():
    """Search across multiple databases"""

    with request_context("global_search"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate search parameters
        data = validate_search_params(data)

        query = data["query"]
        search_type = data.get("search_type", "vector")
        return_type = data.get("return_type", "documents")
        k = data.get("k", 10)
        score_threshold = data.get("score_threshold", 0.0)
        filters = data.get("filters")
        databases = data.get("databases")  # Optional list of databases to search
        vector_weight = data.get("vector_weight", 0.7)

        try:

            results = current_app.db_manager.search_databases(
                query=query,
                database_names=databases,
                search_type=search_type,
                return_type=return_type,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
                vector_weight=vector_weight
            )
            for db_name, db_results in results.items():
                results[db_name] = [serialize_query_result(result) for result in db_results]

            return jsonify({
                "results": results,
                "search_type": search_type,
                "return_type": return_type
            })

        except Exception as e:
            db_logger.log_error("global_search", e, search_type=search_type)
            raise


# Health and System Routes
@api.route("/api/v1/health", methods=["GET"])
@handle_errors
def health_check():
    """System health check endpoint"""

    try:
        status = {
            "status": "healthy",
            "version": get_system_version(),
            "ollama_available": check_ollama_service(),
            "timestamp": logger.manager.loggerDict.get('timestamp', 'unknown')
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


# Embedding Routes
@api.route("/api/v1/<db_name>/embeddings", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("get_embeddings_for_db")
def get_embeddings_for_db(db_name):
    """Get embeddings using the database's embedding provider"""

    with request_context("get_embeddings_for_db"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ['texts'])

        texts = data["texts"]
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValidationError("Texts must be a string or array of strings", field="texts")

        try:
            db = current_app.db_manager.get_db(db_name)
            embeddings = db.embedding_provider.embed_sync(texts)

            return jsonify({"embeddings": embeddings.tolist()})

        except Exception as e:
            db_logger.log_error("get_embeddings_for_db", e, database_name=db_name)
            raise


@api.route("/api/v1/embeddings", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("get_embeddings")
def get_embeddings():
    """Get embeddings from specified provider and model"""

    with request_context("get_embeddings"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ['texts', 'provider', 'model'])

        texts = data["texts"]
        provider = data["provider"]
        model = data["model"]

        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValidationError("Texts must be a string or array of strings", field="texts")

        from localvectordb.embeddings import EmbeddingRegistry
        if provider not in EmbeddingRegistry.list():
            raise ValidationError(f"Provider must be one of: {', '.join(EmbeddingRegistry.list())}", field="provider", value=provider)

        try:
            embedding_provider = EmbeddingRegistry.create_provider(provider, model)
            embeddings = embedding_provider.embed_sync(texts)

            return jsonify({"embeddings": embeddings.tolist()})

        except Exception as e:
            logger.error(f"Error getting embeddings with {provider}/{model}: {e}")
            raise APIError(
                message=f"Failed to get embeddings: {str(e)}",
                error_code="EMBEDDING_GENERATION_FAILED",
                status_code=503,
                recoverable=True,
                details={"provider": provider, "model": model}
            )