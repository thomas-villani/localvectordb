# src/localvectordb_server/routers/embeddings.py
"""Embedding generation routes (Pydantic request/response models + dependency injection)."""

import logging
from typing import List, Optional, Union

from fastapi import APIRouter, Depends

from localvectordb_server._auth import require_read_permission
from localvectordb_server._error_handlers import APIError, ValidationError
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.routers._deps import get_db
from localvectordb_server.routers._models import StrictModel

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["embeddings"])


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class DbEmbeddingsBody(StrictModel):
    """Get embeddings for existing chunk ``ids`` *or* for custom ``texts``."""

    ids: Optional[Union[str, List[str]]] = None
    texts: Optional[Union[str, List[str]]] = None


class DbEmbeddingsResponse(StrictModel):
    embeddings: List[List[float]]
    provider: str
    model: str


class EmbeddingsBody(StrictModel):
    """Get embeddings from an explicitly named provider and model."""

    provider: str
    model: str
    texts: Union[str, List[str]]


class EmbeddingsResponse(StrictModel):
    embeddings: List[List[float]]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post(
    "/{db_name}/embeddings",
    response_model=DbEmbeddingsResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("get_embeddings_for_db")
async def get_embeddings_for_db(db_name: str, body: DbEmbeddingsBody, db=Depends(get_db)):
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
        try:
            if body.ids:
                id_list = [body.ids] if isinstance(body.ids, str) else body.ids
                embeddings = db.get_chunk_embeddings(id_list).tolist()
            else:
                if not body.texts:
                    raise ValidationError(
                        "Missing required fields: texts",
                        field="texts",
                        details={"missing_fields": ["texts"]},
                    )
                texts = [body.texts] if isinstance(body.texts, str) else body.texts
                embeddings = db.embedding_provider.embed_sync(texts).tolist()

            return {
                "embeddings": embeddings,
                "provider": db.embedding_provider.provider_name,
                "model": db.embedding_provider.model,
            }

        except Exception as e:
            db_logger.log_error("get_embeddings_for_db", e, database_name=db_name)
            raise


@router.post(
    "/embeddings",
    response_model=EmbeddingsResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("get_embeddings")
async def get_embeddings(body: EmbeddingsBody):
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
        provider = body.provider
        model = body.model
        texts = [body.texts] if isinstance(body.texts, str) else body.texts

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
