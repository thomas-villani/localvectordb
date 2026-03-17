# src/localvectordb_server/routers/embeddings.py
"""Embedding generation routes."""

import logging

from fastapi import APIRouter, Depends, Request

from localvectordb_server._auth import require_read_permission
from localvectordb_server._error_handlers import (
    APIError,
    ValidationError,
    validate_required_fields,
)
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.routers._deps import get_db

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["embeddings"])


@router.post("/{db_name}/embeddings", dependencies=[Depends(require_read_permission)])
@log_performance("get_embeddings_for_db")
async def get_embeddings_for_db(db_name: str, request: Request):
    """Get embeddings using the database's embedding provider, or for existing chunks by id.

    Request body to get embeddings for existing chunks from the database::

        {
           "ids": ["doc_1:0", "doc_1:1", ...]
        }

    Request body to get embeddings for custom texts::

        {
            "texts": ["The first text", "The second text", ...]
        }

    Returns::

        {
            "embeddings": [[0.1234, 0.5678, ...], ...],
            "provider": "embedding-provider",
            "model": "embedding-model"
        }
    """
    with request_context("get_embeddings_for_db"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        id_list = data.get("ids")
        if id_list:
            if isinstance(id_list, str):
                id_list = [id_list]
            elif not isinstance(id_list, list):
                raise ValidationError("`ids` must be a string or array of strings", field="ids")
        else:
            validate_required_fields(data, ["texts"])
            texts = data["texts"]
            if isinstance(texts, str):
                texts = [texts]
            elif not isinstance(texts, list):
                raise ValidationError("`texts` must be a string or array of strings", field="texts")

        try:
            db = get_db(db_name, request)
            if id_list:
                embeddings = db.get_chunk_embeddings(id_list)
            else:
                embeddings = db.embedding_provider.embed_sync(texts).tolist()

            return {
                "embeddings": embeddings,
                "provider": db.embedding_provider.provider_name,
                "model": db.embedding_provider.model,
            }

        except Exception as e:
            db_logger.log_error("get_embeddings_for_db", e, database_name=db_name)
            raise


@router.post("/embeddings", dependencies=[Depends(require_read_permission)])
@log_performance("get_embeddings")
async def get_embeddings(request: Request):
    """Get embeddings from specified provider and model.

    Request body::

        {
            "provider": "ollama",
            "model": "nomic-embed-text",
            "texts": ["The first text", "The second text", ...]
        }

    Returns::

        {
            "embeddings": [[0.1234, 0.5678, ...], ...]
        }
    """
    with request_context("get_embeddings"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ["texts", "provider", "model"])

        texts = data["texts"]
        provider = data["provider"]
        model = data["model"]

        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValidationError("Texts must be a string or array of strings", field="texts")

        from localvectordb.embeddings import EmbeddingRegistry

        if provider not in EmbeddingRegistry.list():
            raise ValidationError(
                f"Provider must be one of: {', '.join(EmbeddingRegistry.list())}",
                field="provider",
                value=provider,
            )

        try:
            embedding_provider = EmbeddingRegistry.create_provider(provider, model)
            embeddings = embedding_provider.embed_sync(texts)

            return {"embeddings": embeddings.tolist()}

        except Exception as e:
            logger.error(f"Error getting embeddings with {provider}/{model}: {e}")
            raise APIError(
                message=f"Failed to get embeddings: {str(e)}",
                error_code="EMBEDDING_GENERATION_FAILED",
                status_code=503,
                recoverable=True,
                details={"provider": provider, "model": model},
            ) from e
