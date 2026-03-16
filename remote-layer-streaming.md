# Remote Layer Streaming (Deferred)

## Context
As part of the streaming query results feature, we designed a three-layer approach:
1. **Core layer**: `QueryCursor` with cached FAISS results + lazy SQLite metadata loading
2. **API layer**: Async generator interface (`query_stream` / `query_stream_async`)
3. **Remote layer**: SSE streaming for RemoteVectorDB

## Decision
Layer 3 (remote streaming) is deferred — it will be implemented during the upcoming FastAPI backend refactor.

## What to implement during FastAPI refactor
- **Server-side**: SSE (Server-Sent Events) endpoint that streams `QueryResult` objects as they're processed, rather than building the full JSON response
- **Client-side**: `RemoteVectorDB.query_stream()` / `query_stream_async()` that consumes the SSE stream and exposes the same async generator API as the local implementation
- FastAPI has native SSE support via `StreamingResponse` + `async def event_generator()`, which maps cleanly to this design
- The goal is a unified interface: `async for result in db.query_stream(...)` works identically for local and remote

## Local streaming API to match
The local `query_stream` / `query_stream_async` methods on `LocalVectorDB` and `QueryBuilder` will already exist by the time of the refactor. The remote implementation should mirror their signatures exactly.
