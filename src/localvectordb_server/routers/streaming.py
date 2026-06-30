# src/localvectordb_server/routers/streaming.py
"""SSE streaming endpoint for query results."""

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from pydantic import Field
from sse_starlette.sse import EventSourceResponse

from localvectordb_server._auth import require_read_permission
from localvectordb_server._error_handlers import ValidationError
from localvectordb_server._logcfg import DatabaseLogger
from localvectordb_server._serializers import serialize_query_result
from localvectordb_server.routers._deps import get_db
from localvectordb_server.routers._models import QueryBody

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["streaming"])


class StreamQueryBody(QueryBody):
    batch_size: int = Field(default=10, ge=1, le=1000)


@router.post("/{db_name}/query/stream", dependencies=[Depends(require_read_permission)])
async def query_stream(db_name: str, body: StreamQueryBody, db=Depends(get_db)):
    """Stream query results via Server-Sent Events.

    Uses cursor-based async iteration for lazy FAISS+SQLite streaming.
    Event types: 'result' (individual QueryResult), 'done' (completion), 'error'.
    """
    # Reranking needs the fully materialized result set, which is incompatible
    # with streaming (mirrors the library cursor/stream behavior).
    if body.reranker_config:
        raise ValidationError("Reranking is not supported with streaming queries", field="reranker_config")

    common = dict(
        query=body.query,
        search_type=body.search_type,
        return_type=body.return_type,
        k=body.k,
        score_threshold=body.score_threshold,
        filters=body.filters,
        vector_weight=body.vector_weight,
        context_window=body.context_window,
        document_scoring_method=body.document_scoring_method,
        document_scoring_options=body.document_scoring_options,
    )

    async def event_generator() -> AsyncIterator[dict]:
        try:
            if hasattr(db, "query_cursor_async"):
                cursor = db.query_cursor_async(batch_size=body.batch_size, **common)
                count = 0
                async for result in cursor:
                    yield {"event": "result", "data": json.dumps(serialize_query_result(result))}
                    count += 1
                yield {"event": "done", "data": json.dumps({"total_results": count})}

            elif hasattr(db, "query_stream"):
                count = 0
                for batch in db.query_stream(batch_size=body.batch_size, **common):
                    for result in batch:
                        yield {"event": "result", "data": json.dumps(serialize_query_result(result))}
                        count += 1
                yield {"event": "done", "data": json.dumps({"total_results": count})}

            else:
                results = db.query(**common)
                for result in results:
                    yield {"event": "result", "data": json.dumps(serialize_query_result(result))}
                yield {"event": "done", "data": json.dumps({"total_results": len(results)})}

        except Exception as e:
            logger.error(f"Streaming error for {db_name}: {e}", exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())
