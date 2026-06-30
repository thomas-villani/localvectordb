"""
Search and query routes for the LocalVectorDB FastAPI server.

Replaces the Flask search routes from routes.py with a FastAPI APIRouter.
Provides unified query, convenience search endpoints, query builder execution,
multi-column search, metadata filtering, and global cross-database search.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request

from localvectordb._filters import FilterQueryBuilder
from localvectordb._schema import DatabaseSchema
from localvectordb_server._auth import require_read_permission
from localvectordb_server._error_handlers import (
    ValidationError,
    validate_field_type,
    validate_required_fields,
    validate_search_params,
)
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.routers._deps import get_db, get_db_manager

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["search"])


def serialize_query_result(result) -> Dict[str, Any]:
    """Serialize a QueryResult object for JSON response."""
    data = {
        "id": result.id,
        "score": result.score,
        "type": result.type,
        "content": result.content,
        "metadata": result.metadata,
    }
    if result.type == "chunk" and result.document_id:
        data["document_id"] = result.document_id
    if result.position:
        data["position"] = result.position.to_dict()
    return data


def serialize_document(doc) -> Dict[str, Any]:
    """Serialize a Document object for JSON response."""
    return {
        "id": doc.id,
        "content": doc.content,
        "metadata": doc.metadata,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        "content_hash": doc.content_hash,
    }


def search_handler(db, db_name: str, search_params: Dict[str, Any]) -> Dict[str, Any]:
    """Unified search handler for all search types with semantic filtering.

    Parameters
    ----------
    db : LocalVectorDB
        The database instance to search.
    db_name : str
        Name of the database (for logging).
    search_params : dict
        Validated search parameters.

    Returns
    -------
    dict
        Search results with metadata.
    """
    # Validate search parameters
    search_params = validate_search_params(search_params)

    query_text = search_params["query"]
    search_type = search_params.get("search_type", "hybrid")
    return_type = search_params.get("return_type", "documents")
    search_level = search_params.get("search_level", "chunks")
    k = search_params.get("k", 10)
    score_threshold = search_params.get("score_threshold", 0.0)
    filters = search_params.get("filters", search_params.get("metadata_filters"))
    vector_weight = search_params.get("vector_weight", 0.7)

    context_window = search_params.get("context_window", 2)
    semantic_dedup_threshold = search_params.get("semantic_dedup_threshold")
    document_scoring_method = search_params.get("document_scoring_method", "frequency_boost")
    document_scoring_options = search_params.get("document_scoring_options", None)
    reranker_config = search_params.get("reranker_config")
    semantic_filters = search_params.get("semantic_filters")

    try:
        db_logger.log_query(
            "search",
            database_name=db_name,
            search_type=search_type,
            return_type=return_type,
            k=k,
            query_length=len(query_text),
        )

        results = db.query(
            query=query_text,
            search_type=search_type,
            return_type=return_type,
            search_level=search_level,
            k=k,
            score_threshold=score_threshold,
            filters=filters,
            vector_weight=vector_weight,
            context_window=context_window,
            semantic_dedup_threshold=semantic_dedup_threshold,
            document_scoring_method=document_scoring_method,
            document_scoring_options=document_scoring_options,
            reranker_config=reranker_config,
        )

        # Apply semantic filters server-side if provided
        if semantic_filters:
            from localvectordb.core import Document

            # Convert results to documents for semantic filtering
            documents = []
            for result in results:
                doc = Document(id=result.id, content=result.content, metadata=result.metadata)
                doc.metadata["_original_score"] = result.score
                documents.append(doc)

            # Apply each semantic filter
            for sem_filter in semantic_filters:
                field = sem_filter["field"]
                concept = sem_filter["concept"]
                threshold = sem_filter.get("threshold", 0.7)
                metric = sem_filter.get("metric", "cosine")

                # Get field contents for documents
                field_contents = []
                valid_docs = []
                for doc in documents:
                    if field == "content":
                        field_contents.append(doc.content)
                        valid_docs.append(doc)
                    elif field in doc.metadata and doc.metadata[field]:
                        field_contents.append(str(doc.metadata[field]))
                        valid_docs.append(doc)

                if field_contents:
                    # Generate embeddings using database's embedding provider
                    embedding_provider = db.embedding_provider
                    concept_embedding = embedding_provider.embed_sync([concept])[0]
                    field_embeddings = embedding_provider.embed_sync(field_contents)

                    # Apply similarity filtering based on metric
                    import numpy as np

                    filtered_docs = []

                    for doc, field_emb in zip(valid_docs, field_embeddings, strict=False):
                        if metric == "cosine":
                            # Cosine similarity
                            similarity = np.dot(concept_embedding, field_emb) / (
                                np.linalg.norm(concept_embedding) * np.linalg.norm(field_emb)
                            )
                        elif metric == "euclidean":
                            # Euclidean distance (convert to similarity)
                            distance = np.linalg.norm(concept_embedding - field_emb)
                            similarity = 1 / (1 + distance)
                        elif metric == "dot":
                            # Dot product
                            similarity = np.dot(concept_embedding, field_emb)
                        else:
                            similarity = 0.0

                        if similarity >= threshold:
                            filtered_docs.append(doc)

                    documents = filtered_docs

            # Convert filtered documents back to QueryResult format
            from localvectordb.core import QueryResult

            filtered_results = []
            for doc in documents:
                score = doc.metadata.pop("_original_score", 0.0)
                filtered_results.append(
                    QueryResult(id=doc.id, score=score, type="document", content=doc.content, metadata=doc.metadata)
                )

            results = filtered_results

        # Serialize results
        serialized_results = [serialize_query_result(result) for result in results]

        db_logger.log_query(
            "search_success", database_name=db_name, search_type=search_type, result_count=len(serialized_results)
        )

        return {
            "results": serialized_results,
            "search_type": search_type,
            "return_type": return_type,
            "total_results": len(serialized_results),
            # Include processing info in response
            "processing_info": {
                "context_window": context_window if return_type in ("context", "enriched") else None,
                "semantic_dedup_applied": semantic_dedup_threshold is not None,
                "document_scoring_method": document_scoring_method if return_type == "documents" else None,
                "semantic_filters_applied": len(semantic_filters) if semantic_filters else 0,
            },
        }

    except Exception as e:
        db_logger.log_error("search", e, database_name=db_name, search_type=search_type)
        raise


# ---------------------------------------------------------------------------
# Search Routes
# ---------------------------------------------------------------------------


@router.post("/{db_name}/query", dependencies=[Depends(require_read_permission)])
@log_performance("query_documents")
async def query_documents(db_name: str, request: Request):
    """Unified query interface for all search types."""

    with request_context("query_documents"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        db = get_db(db_name, request)
        return search_handler(db, db_name, data)


@router.post("/{db_name}/search/vector", dependencies=[Depends(require_read_permission)])
@log_performance("vector_search")
async def vector_search(db_name: str, request: Request):
    """Vector similarity search (convenience endpoint)."""

    with request_context("vector_search"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Force search_type to vector
        data["search_type"] = "vector"
        db = get_db(db_name, request)
        return search_handler(db, db_name, data)


@router.post("/{db_name}/search/keyword", dependencies=[Depends(require_read_permission)])
@log_performance("keyword_search")
async def keyword_search(db_name: str, request: Request):
    """Keyword search (convenience endpoint)."""

    with request_context("keyword_search"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Force search_type to keyword
        data["search_type"] = "keyword"
        db = get_db(db_name, request)
        return search_handler(db, db_name, data)


@router.post("/{db_name}/search/hybrid", dependencies=[Depends(require_read_permission)])
@log_performance("hybrid_search")
async def hybrid_search(db_name: str, request: Request):
    """Hybrid search (convenience endpoint)."""

    with request_context("hybrid_search"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Force search_type to hybrid
        data["search_type"] = "hybrid"
        db = get_db(db_name, request)
        return search_handler(db, db_name, data)


@router.post("/{db_name}/query_builder", dependencies=[Depends(require_read_permission)])
@log_performance("query_builder")
async def query_builder_execute(db_name: str, request: Request):
    """Execute a QueryBuilder query with full state from client.

    This endpoint allows RemoteVectorDB to send a complete QueryBuilder state
    for server-side execution, including semantic filters.
    """

    with request_context("query_builder"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        try:
            db = get_db(db_name, request)

            # Reconstruct QueryBuilder from state
            from localvectordb.query_builder import QueryBuilder

            builder = QueryBuilder(db)

            # Apply search clauses
            search_clauses = data.get("search_clauses", [])
            for clause in search_clauses:
                builder = builder.search(clause["text"], search_type=clause.get("search_type", "hybrid"))

            # Apply exact filters
            exact_filters = data.get("exact_filters", [])
            for filter_item in exact_filters:
                builder = builder.filter(filter_item["field"], **filter_item["conditions"])

            # Apply semantic filters
            semantic_filters = data.get("semantic_filters", [])
            for sem_filter in semantic_filters:
                builder = builder.semantic_filter(
                    sem_filter["field"],
                    sem_filter["concept"],
                    threshold=sem_filter.get("threshold", 0.7),
                    metric=sem_filter.get("metric", "cosine"),
                )

            # Apply other builder settings
            if "search_type" in data:
                builder._search_type = data["search_type"]

            if "vector_weight" in data:
                builder._vector_weight = data["vector_weight"]

            if "return_type" in data:
                builder._return_type = data["return_type"]

            if "order_by" in data:
                for order in data["order_by"]:
                    builder = builder.order_by(order["field"], order.get("direction", "asc"))

            if "limit" in data:
                builder = builder.limit(data["limit"])

            if "offset" in data:
                builder = builder.offset(data["offset"])

            if "group_by" in data:
                builder = builder.group_by(*data["group_by"])

            if "aggregations" in data:
                for agg in data["aggregations"]:
                    builder = builder.aggregate(agg["field"], agg["function"], agg.get("alias"))

            # Execute the query
            results = builder.execute()

            # Serialize results
            serialized_results = [serialize_query_result(result) for result in results]

            db_logger.log_query("query_builder_success", database_name=db_name, result_count=len(serialized_results))

            return {"results": serialized_results, "total_results": len(serialized_results)}

        except Exception as e:
            db_logger.log_error("query_builder", e, database_name=db_name)
            raise


@router.post("/{db_name}/query-multi-column", dependencies=[Depends(require_read_permission)])
@log_performance("query_multi_column")
async def query_multi_column(db_name: str, request: Request):
    """Query across multiple columns (main content + embedding-enabled metadata fields)."""

    with request_context("query_multi_column"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate required fields
        validate_required_fields(data, ["query"])

        query_text = data["query"]
        columns = data.get("columns")
        search_type = data.get("search_type", "hybrid")
        return_type = data.get("return_type", "documents")
        k = data.get("k", 10)
        score_threshold = data.get("score_threshold", 0.0)
        filters = data.get("filters", data.get("metadata_filters"))
        vector_weight = data.get("vector_weight", 0.7)
        document_scoring_method = data.get("document_scoring_method", "frequency_boost")
        document_scoring_options = data.get("document_scoring_options")

        # Validate parameters
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValidationError("Query must be a non-empty string", field="query")

        if columns is not None:
            if not isinstance(columns, list):
                raise ValidationError("Columns must be a list of strings", field="columns")
            for i, col in enumerate(columns):
                if not isinstance(col, str):
                    raise ValidationError(f"Column at index {i} must be a string", field=f"columns[{i}]")

        if search_type not in ["vector", "keyword", "hybrid"]:
            raise ValidationError("Search type must be 'vector', 'keyword', or 'hybrid'", field="search_type")

        if return_type not in ["documents", "chunks", "context", "enriched"]:
            raise ValidationError(
                "Return type must be 'documents', 'chunks', 'context', or 'enriched'", field="return_type"
            )

        validate_field_type(data, "k", int)
        # Upper bound is hardcoded; could be made configurable via server settings.
        if k < 1 or k > 1000:
            raise ValidationError("k must be between 1 and 1000", field="k", value=k)

        validate_field_type(data, "score_threshold", (int, float))
        if score_threshold < 0 or score_threshold > 1:
            raise ValidationError(
                "Score threshold must be between 0 and 1", field="score_threshold", value=score_threshold
            )

        validate_field_type(data, "vector_weight", (int, float))
        if vector_weight < 0 or vector_weight > 1:
            raise ValidationError("Vector weight must be between 0 and 1", field="vector_weight", value=vector_weight)

        if filters is not None and not isinstance(filters, dict):
            raise ValidationError("Filters must be a dictionary", field="filters")

        try:
            db = get_db(db_name, request)

            db_logger.log_query(
                "query_multi_column",
                database_name=db_name,
                search_type=search_type,
                return_type=return_type,
                k=k,
                query_length=len(query_text),
                columns_count=len(columns) if columns else None,
            )

            results = db.query_multi_column(
                query=query_text,
                columns=columns,
                search_type=search_type,
                return_type=return_type,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
                vector_weight=vector_weight,
                document_scoring_method=document_scoring_method,
                document_scoring_options=document_scoring_options,
            )

            # Serialize results
            serialized_results = [serialize_query_result(result) for result in results]

            db_logger.log_query(
                "query_multi_column_success",
                database_name=db_name,
                search_type=search_type,
                result_count=len(serialized_results),
            )

            return {
                "results": serialized_results,
                "search_type": search_type,
                "return_type": return_type,
                "total_results": len(serialized_results),
                "columns_searched": columns,
                "processing_info": {
                    "document_scoring_method": document_scoring_method if return_type == "documents" else None
                },
            }

        except Exception as e:
            db_logger.log_error("query_multi_column", e, database_name=db_name, search_type=search_type)
            raise


@router.post("/{db_name}/filter", dependencies=[Depends(require_read_permission)])
@log_performance("filter_documents")
async def filter_documents(db_name: str, request: Request):
    """Filter documents by metadata with enhanced filtering capabilities.

    Request body supports::

        {
            "where": {
                // Simple format
                "author": "John Doe",
                "year": 2023,

                // Advanced format with operators
                "rating": {"$gte": 4.0, "$lte": 5.0},
                "tags": {"$contains": "python"},
                "category": {"$in": ["tech", "science"]},
                "title": {"$ilike": "%tutorial%"},

                // Logical operators
                "$and": [
                    {"year": {"$gte": 2020}},
                    {"$or": [
                        {"author": "John Doe"},
                        {"category": "featured"}
                    ]}
                ]
            },
            "order_by": "created_at DESC",
            "limit": 100,
            "offset": 0
        }

    Supported operators:

    - Comparison: $eq, $ne, $gt, $lt, $gte, $lte, $in, $nin
    - String: $like, $ilike, $contains, $startswith, $endswith
    - Existence: $exists, $not_exists
    - Type: $type
    - Logical: $and, $or, $not
    - JSON: $contains, $not_contains (for JSON fields)
    """

    with request_context("filter_documents"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        where = data.get("where")
        order_by = data.get("order_by")
        limit = data.get("limit")
        offset = data.get("offset", 0)

        # Validate types
        if where is not None:
            validate_field_type(data, "where", dict)
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
            db = get_db(db_name, request)

            # Perform secure ORDER BY validation with schema access
            if order_by is not None:
                try:
                    # Create FilterQueryBuilder with the database's metadata schema
                    filter_builder = FilterQueryBuilder(db.metadata_schema)

                    # Build valid columns set (base columns + metadata columns)
                    base_columns = DatabaseSchema.BASE_COLUMNS
                    metadata_columns = set(db.metadata_schema.keys())
                    valid_columns = set(base_columns).union(metadata_columns)

                    # Validate the ORDER BY clause using secure builder
                    # This will raise DatabaseError if invalid
                    filter_builder.build_order_by_clause(order_by, valid_columns)
                except Exception as e:
                    raise ValidationError(f"Invalid ORDER BY clause: {str(e)}", field="order_by", value=order_by) from e

            db_logger.log_query(
                "filter_documents",
                database_name=db_name,
                has_where=where is not None,
                filter_complexity=len(str(where)) if where else 0,
                limit=limit,
            )

            documents = db.filter(where=where, order_by=order_by, limit=limit, offset=offset)

            # Serialize results
            serialized_docs = [serialize_document(doc) for doc in documents]

            db_logger.log_query("filter_documents_success", database_name=db_name, result_count=len(serialized_docs))

            return {
                "documents": serialized_docs,
                "count": len(serialized_docs),
                "filter_info": {
                    "where_provided": where is not None,
                    "order_by_provided": order_by is not None,
                    "limit": limit,
                    "offset": offset,
                },
            }

        except Exception as e:
            db_logger.log_error("filter_documents", e, database_name=db_name)
            raise


# ---------------------------------------------------------------------------
# Global Search Route
# ---------------------------------------------------------------------------


@router.post("/search", dependencies=[Depends(require_read_permission)])
@log_performance("global_search")
async def global_search(request: Request):
    """Search across multiple databases.

    Request body supports::

        {
            "query": "The search query",        // required
            "databases": ["db_one", "db_two"],   // optional list of databases (defaults to all)
            "search_type": "hybrid",             // or "vector" or "keyword", default is hybrid
            "return_type": "documents",          // or "chunks", or "context"
            "k": 10,                             // number of documents to return from each database
            "score_threshold": 0.4,              // optionally limit results by similarity score
            "filters": { ... },                  // MongoDB-style filters
            "vector_weight": 0.7,                // balance of vector:keyword for hybrid search
            "context_window": 2                  // for return_type="context"
        }

    Returns::

        {
            "results": {
                "db_one": [ ... ],
                "db_two": [ ... ]
            },
            "search_type": "hybrid",
            "return_type": "documents"
        }
    """

    with request_context("global_search"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate search parameters
        data = validate_search_params(data)

        query = data["query"]
        search_type = data.get("search_type", "hybrid")
        return_type = data.get("return_type", "documents")
        k = data.get("k", 10)
        score_threshold = data.get("score_threshold", 0.0)
        filters = data.get("filters")
        databases = data.get("databases")  # Optional list of databases to search
        vector_weight = data.get("vector_weight", 0.7)
        context_window = data.get("context_window", 2)

        try:
            results = get_db_manager(request).search_databases(
                query=query,
                database_names=databases,
                search_type=search_type,
                return_type=return_type,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
                vector_weight=vector_weight,
                context_window=context_window,
            )
            for db_name, db_results in results.items():
                results[db_name] = [serialize_query_result(result) for result in db_results]

            return {"results": results, "search_type": search_type, "return_type": return_type}

        except Exception as e:
            db_logger.log_error("global_search", e, search_type=search_type)
            raise
