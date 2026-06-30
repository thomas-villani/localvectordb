"""
Search and query routes for the LocalVectorDB FastAPI server.

Replaces the Flask search routes from routes.py with a FastAPI APIRouter.
Provides unified query, convenience search endpoints, query builder execution,
multi-column search, metadata filtering, and global cross-database search.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import ConfigDict, Field

from localvectordb._filters import FilterQueryBuilder
from localvectordb._schema import DatabaseSchema
from localvectordb_server._auth import require_read_permission
from localvectordb_server._error_handlers import ValidationError
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server._serializers import serialize_document, serialize_query_result
from localvectordb_server.routers._deps import get_db, get_db_manager
from localvectordb_server.routers._models import QueryBody, StrictModel

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["search"])


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class SearchBody(QueryBody):
    """Unified ``query()`` body plus optional server-side semantic filters."""

    semantic_filters: Optional[List[Dict[str, Any]]] = None


class MultiColumnBody(QueryBody):
    """Body for multi-column search (main content + embedding-enabled fields)."""

    columns: Optional[List[str]] = None


class FilterDocumentsBody(StrictModel):
    """Metadata-only filtering with limit/offset pagination and ORDER BY."""

    filters: Optional[Dict[str, Any]] = None
    order_by: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)


class QueryBuilderStateBody(StrictModel):
    """Dynamic QueryBuilder state sent by ``RemoteVectorDB`` for execution.

    Unlike the other models this one *ignores* unknown fields (``extra="ignore"``)
    because the builder state is open-ended; every field is optional.
    """

    model_config = ConfigDict(extra="ignore")

    search_clauses: List[Dict[str, Any]] = []
    exact_filters: List[Dict[str, Any]] = []
    semantic_filters: List[Dict[str, Any]] = []
    search_type: Optional[Literal["vector", "keyword", "hybrid"]] = None
    vector_weight: Optional[float] = None
    return_type: Optional[Literal["documents", "chunks", "sections", "context"]] = None
    order_by: List[Dict[str, Any]] = []
    limit: Optional[int] = None
    offset: Optional[int] = None
    group_by: Optional[List[str]] = None
    aggregations: List[Dict[str, Any]] = []


class GlobalSearchBody(SearchBody):
    """Cross-database search body (adds an optional target database list)."""

    databases: Optional[List[str]] = None


def search_handler(db, db_name: str, search_params: Dict[str, Any]) -> Dict[str, Any]:
    """Unified search handler for all search types with semantic filtering.

    Parameters
    ----------
    db : LocalVectorDB
        The database instance to search.
    db_name : str
        Name of the database (for logging).
    search_params : dict
        Validated search parameters (Pydantic-validated upstream).

    Returns
    -------
    dict
        Search results with metadata.
    """
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
async def query_documents(db_name: str, body: SearchBody, db=Depends(get_db)):
    """Unified query interface for all search types."""
    with request_context("query_documents"):
        return search_handler(db, db_name, body.model_dump())


@router.post("/{db_name}/search/vector", dependencies=[Depends(require_read_permission)])
@log_performance("vector_search")
async def vector_search(db_name: str, body: SearchBody, db=Depends(get_db)):
    """Vector similarity search (convenience endpoint)."""
    with request_context("vector_search"):
        params = body.model_dump()
        params["search_type"] = "vector"
        return search_handler(db, db_name, params)


@router.post("/{db_name}/search/keyword", dependencies=[Depends(require_read_permission)])
@log_performance("keyword_search")
async def keyword_search(db_name: str, body: SearchBody, db=Depends(get_db)):
    """Keyword search (convenience endpoint)."""
    with request_context("keyword_search"):
        params = body.model_dump()
        params["search_type"] = "keyword"
        return search_handler(db, db_name, params)


@router.post("/{db_name}/search/hybrid", dependencies=[Depends(require_read_permission)])
@log_performance("hybrid_search")
async def hybrid_search(db_name: str, body: SearchBody, db=Depends(get_db)):
    """Hybrid search (convenience endpoint)."""
    with request_context("hybrid_search"):
        params = body.model_dump()
        params["search_type"] = "hybrid"
        return search_handler(db, db_name, params)


@router.post("/{db_name}/query_builder", dependencies=[Depends(require_read_permission)])
@log_performance("query_builder")
async def query_builder_execute(db_name: str, body: QueryBuilderStateBody, db=Depends(get_db)):
    """Execute a QueryBuilder query with full state from client.

    This endpoint allows RemoteVectorDB to send a complete QueryBuilder state
    for server-side execution, including semantic filters.
    """
    with request_context("query_builder"):
        try:
            # Reconstruct QueryBuilder from state
            from localvectordb.query_builder import QueryBuilder

            builder = QueryBuilder(db)

            # Apply search clauses
            for clause in body.search_clauses:
                builder = builder.search(clause["text"], search_type=clause.get("search_type", "hybrid"))

            # Apply exact filters
            for filter_item in body.exact_filters:
                builder = builder.filter(filter_item["field"], **filter_item["conditions"])

            # Apply semantic filters
            for sem_filter in body.semantic_filters:
                builder = builder.semantic_filter(
                    sem_filter["field"],
                    sem_filter["concept"],
                    threshold=sem_filter.get("threshold", 0.7),
                    metric=sem_filter.get("metric", "cosine"),
                )

            # Apply other builder settings
            if body.search_type is not None:
                builder._search_type = body.search_type

            if body.vector_weight is not None:
                builder._vector_weight = body.vector_weight

            if body.return_type is not None:
                builder._return_type = body.return_type

            for order in body.order_by:
                builder = builder.order_by(order["field"], order.get("direction", "asc"))

            if body.limit is not None:
                builder = builder.limit(body.limit)

            if body.offset is not None:
                builder = builder.offset(body.offset)

            if body.group_by is not None:
                builder = builder.group_by(*body.group_by)

            for agg in body.aggregations:
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
async def query_multi_column(db_name: str, body: MultiColumnBody, db=Depends(get_db)):
    """Query across multiple columns (main content + embedding-enabled metadata fields)."""
    with request_context("query_multi_column"):
        try:
            db_logger.log_query(
                "query_multi_column",
                database_name=db_name,
                search_type=body.search_type,
                return_type=body.return_type,
                k=body.k,
                query_length=len(body.query),
                columns_count=len(body.columns) if body.columns else None,
            )

            results = db.query_multi_column(
                query=body.query,
                columns=body.columns,
                search_type=body.search_type,
                return_type=body.return_type,
                k=body.k,
                score_threshold=body.score_threshold,
                filters=body.filters,
                vector_weight=body.vector_weight,
                document_scoring_method=body.document_scoring_method,
                document_scoring_options=body.document_scoring_options,
            )

            # Serialize results
            serialized_results = [serialize_query_result(result) for result in results]

            db_logger.log_query(
                "query_multi_column_success",
                database_name=db_name,
                search_type=body.search_type,
                result_count=len(serialized_results),
            )

            return {
                "results": serialized_results,
                "search_type": body.search_type,
                "return_type": body.return_type,
                "total_results": len(serialized_results),
                "columns_searched": body.columns,
                "processing_info": {
                    "document_scoring_method": (
                        body.document_scoring_method if body.return_type == "documents" else None
                    )
                },
            }

        except Exception as e:
            db_logger.log_error("query_multi_column", e, database_name=db_name, search_type=body.search_type)
            raise


@router.post("/{db_name}/filter", dependencies=[Depends(require_read_permission)])
@log_performance("filter_documents")
async def filter_documents(db_name: str, body: FilterDocumentsBody, db=Depends(get_db)):
    """Filter documents by metadata with enhanced filtering capabilities.

    Request body supports::

        {
            "filters": {
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
        filters = body.filters
        order_by = body.order_by
        limit = body.limit
        offset = body.offset

        try:
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
                has_where=filters is not None,
                filter_complexity=len(str(filters)) if filters else 0,
                limit=limit,
            )

            documents = db.filter(where=filters, order_by=order_by, limit=limit, offset=offset)

            # Serialize results
            serialized_docs = [serialize_document(doc) for doc in documents]

            db_logger.log_query("filter_documents_success", database_name=db_name, result_count=len(serialized_docs))

            return {
                "documents": serialized_docs,
                "count": len(serialized_docs),
                "filter_info": {
                    "where_provided": filters is not None,
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
async def global_search(body: GlobalSearchBody, db_manager=Depends(get_db_manager)):
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
        try:
            results = db_manager.search_databases(
                query=body.query,
                database_names=body.databases,
                search_type=body.search_type,
                return_type=body.return_type,
                k=body.k,
                score_threshold=body.score_threshold,
                filters=body.filters,
                vector_weight=body.vector_weight,
                context_window=body.context_window,
            )
            for db_name, db_results in results.items():
                results[db_name] = [serialize_query_result(result) for result in db_results]

            return {"results": results, "search_type": body.search_type, "return_type": body.return_type}

        except Exception as e:
            db_logger.log_error("global_search", e, search_type=body.search_type)
            raise
