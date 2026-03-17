# src/localvectordb_server/routers/__init__.py
"""FastAPI router aggregation."""

from fastapi import FastAPI


def register_routers(app: FastAPI) -> None:
    """Register all API routers with the FastAPI app."""
    from localvectordb_server.routers.comparison import router as comparison_router
    from localvectordb_server.routers.databases import router as databases_router
    from localvectordb_server.routers.documents import router as documents_router
    from localvectordb_server.routers.embeddings import router as embeddings_router
    from localvectordb_server.routers.factcheck import router as factcheck_router
    from localvectordb_server.routers.health import router as health_router
    from localvectordb_server.routers.schema import router as schema_router
    from localvectordb_server.routers.search import router as search_router
    from localvectordb_server.routers.streaming import router as streaming_router
    from localvectordb_server.routers.tuning import router as tuning_router
    from localvectordb_server.routers.upload import router as upload_router

    prefix = "/api/v1"
    app.include_router(health_router, prefix=prefix)
    app.include_router(databases_router, prefix=prefix)
    app.include_router(documents_router, prefix=prefix)
    app.include_router(search_router, prefix=prefix)
    app.include_router(upload_router, prefix=prefix)
    app.include_router(schema_router, prefix=prefix)
    app.include_router(embeddings_router, prefix=prefix)
    app.include_router(tuning_router, prefix=prefix)
    app.include_router(streaming_router, prefix=prefix)
    app.include_router(comparison_router, prefix=prefix)
    app.include_router(factcheck_router, prefix=prefix)
