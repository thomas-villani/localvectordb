# src/localvectordb_server/routers/streaming.py
"""SSE streaming endpoint for query results."""

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from localvectordb_server._auth import require_read_permission
from localvectordb_server._error_handlers import ValidationError, validate_search_params
from localvectordb_server._logcfg import DatabaseLogger
from localvectordb_server._serializers import serialize_query_result
from localvectordb_server.routers._deps import get_db

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["streaming"])


@router.post("/{db_name}/query/stream", dependencies=[Depends(require_read_permission)])
async def query_stream(db_name: str, request: Request):
    """Stream query results via Server-Sent Events.

    Uses cursor-based async iteration for lazy FAISS+SQLite streaming.
    Event types: 'result' (individual QueryResult), 'done' (completion), 'error'.
    """
    data = await request.json()
    if not data:
        raise ValidationError("Request body cannot be empty")

    data = validate_search_params(data)

    query_text = data["query"]
    search_type = data.get("search_type", "hybrid")
    return_type = data.get("return_type", "documents")
    k = data.get("k", 10)
    score_threshold = data.get("score_threshold", 0.0)
    filters = data.get("filters", data.get("metadata_filters"))
    vector_weight = data.get("vector_weight", 0.7)
    batch_size = data.get("batch_size", 10)
    context_window = data.get("context_window", 2)
    document_scoring_method = data.get("document_scoring_method", "frequency_boost")
    document_scoring_options = data.get("document_scoring_options")

    db = get_db(db_name, request)

    async def event_generator() -> AsyncIterator[dict]:
        try:
            # Check if the database supports async cursor-based streaming
            if hasattr(db, "query_cursor_async"):
                cursor = db.query_cursor_async(
                    query=query_text,
                    search_type=search_type,
                    return_type=return_type,
                    k=k,
                    score_threshold=score_threshold,
                    filters=filters,
                    vector_weight=vector_weight,
                    batch_size=batch_size,
                    context_window=context_window,
                    document_scoring_method=document_scoring_method,
                    document_scoring_options=document_scoring_options,
                )
                count = 0
                async for result in cursor:
                    serialized = serialize_query_result(result)
                    yield {"event": "result", "data": json.dumps(serialized)}
                    count += 1

                yield {"event": "done", "data": json.dumps({"total_results": count})}

            elif hasattr(db, "query_stream"):
                # Fallback to sync streaming via query_stream
                count = 0
                for batch in db.query_stream(
                    query=query_text,
                    search_type=search_type,
                    return_type=return_type,
                    k=k,
                    score_threshold=score_threshold,
                    filters=filters,
                    vector_weight=vector_weight,
                    batch_size=batch_size,
                    context_window=context_window,
                    document_scoring_method=document_scoring_method,
                    document_scoring_options=document_scoring_options,
                ):
                    for result in batch:
                        serialized = serialize_query_result(result)
                        yield {"event": "result", "data": json.dumps(serialized)}
                        count += 1

                yield {"event": "done", "data": json.dumps({"total_results": count})}

            else:
                # Final fallback: run regular query and stream results one by one
                results = db.query(
                    query=query_text,
                    search_type=search_type,
                    return_type=return_type,
                    k=k,
                    score_threshold=score_threshold,
                    filters=filters,
                    vector_weight=vector_weight,
                    context_window=context_window,
                    document_scoring_method=document_scoring_method,
                    document_scoring_options=document_scoring_options,
                )
                for result in results:
                    serialized = serialize_query_result(result)
                    yield {"event": "result", "data": json.dumps(serialized)}

                yield {"event": "done", "data": json.dumps({"total_results": len(results)})}

        except Exception as e:
            logger.error(f"Streaming error for {db_name}: {e}", exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())
