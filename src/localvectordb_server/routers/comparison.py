# src/localvectordb_server/routers/comparison.py
"""Document comparison endpoints exposing ComparisonMixin methods.

Migrated to the model-driven router pattern: Pydantic request/response models,
``db=Depends(get_db)`` dependency injection, and inline ``response_model``
declarations. Response keys are preserved exactly so existing clients keep
working.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from localvectordb.exceptions import MetadataFilterError
from localvectordb_server._auth import require_read_permission
from localvectordb_server._error_handlers import APIError
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server._serializers import serialize_query_result
from localvectordb_server.routers._deps import get_db
from localvectordb_server.routers._models import StrictModel

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["comparison"])


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class CompareBody(StrictModel):
    doc_id_1: str
    doc_id_2: str


class CompareDetailedBody(StrictModel):
    doc_id_1: str
    doc_id_2: str
    chunk_threshold: float = 0.7


class NearestNeighborsBody(StrictModel):
    doc_id: str
    # Floor only (L6/M2): reject k<=0 but impose no ceiling -- the result pool is
    # bounded by the index's ntotal, so an oversized k can't amplify allocation.
    k: int = Field(default=5, ge=1)
    score_threshold: float = 0.0
    filters: Optional[Dict[str, Any]] = None


class SimilarityMatrixBody(StrictModel):
    """Optional body; an empty/absent body means "use defaults"."""

    doc_ids: Optional[List[str]] = None


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class CompareResponse(StrictModel):
    doc_id_1: str
    doc_id_2: str
    similarity: float
    status: str


class ChunkAlignmentModel(StrictModel):
    chunk_index_1: int
    chunk_index_2: int
    similarity: float


class CompareDetailedResponse(StrictModel):
    doc_id_1: str
    doc_id_2: str
    overall_similarity: float
    chunk_alignments: List[ChunkAlignmentModel]
    matched_ratio_1: float
    matched_ratio_2: float
    unmatched_chunks_1: List[int]
    unmatched_chunks_2: List[int]
    status: str


class NearestNeighborsResponse(StrictModel):
    doc_id: str
    k: int
    results: List[Dict[str, Any]]
    total_results: int
    status: str


class SimilarityMatrixResponse(StrictModel):
    doc_ids: List[str]
    matrix: List[List[float]]
    embeddings: List[List[float]]
    status: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post(
    "/databases/{db_name}/compare",
    response_model=CompareResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("compare_documents")
async def compare_documents(db_name: str, body: CompareBody, db=Depends(get_db)):
    """Compare two documents by their IDs and return similarity score."""
    with request_context("compare_documents"):
        if not hasattr(db, "compare_documents"):
            raise APIError(
                message="Document comparison is not available for this database",
                error_code="FEATURE_NOT_AVAILABLE",
                status_code=501,
            )

        try:
            similarity = await run_in_threadpool(db.compare_documents, body.doc_id_1, body.doc_id_2)
            return {
                "doc_id_1": body.doc_id_1,
                "doc_id_2": body.doc_id_2,
                "similarity": similarity,
                "status": "success",
            }
        except ValueError as e:
            # A missing doc raises DocumentNotFoundError (-> global 404); a bare
            # ValueError here means the document exists but has no embeddable
            # content (e.g. whitespace-only). That is unprocessable, not a 500.
            raise APIError(
                message=str(e), error_code="UNPROCESSABLE_DOCUMENT", status_code=422, recoverable=False
            ) from e
        except Exception as e:
            db_logger.log_error("compare_documents", e, database_name=db_name)
            raise


@router.post(
    "/databases/{db_name}/compare/detailed",
    response_model=CompareDetailedResponse,
    response_model_exclude_unset=True,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("compare_documents_detailed")
async def compare_documents_detailed(db_name: str, body: CompareDetailedBody, db=Depends(get_db)):
    """Compare two documents with detailed chunk-level analysis."""
    with request_context("compare_documents_detailed"):
        if not hasattr(db, "compare_documents_detailed"):
            raise APIError(
                message="Detailed document comparison is not available for this database",
                error_code="FEATURE_NOT_AVAILABLE",
                status_code=501,
            )

        try:
            result = await run_in_threadpool(
                db.compare_documents_detailed, body.doc_id_1, body.doc_id_2, chunk_threshold=body.chunk_threshold
            )
            return CompareDetailedResponse(**result.to_dict(), status="success")

        except ValueError as e:
            raise APIError(
                message=str(e), error_code="UNPROCESSABLE_DOCUMENT", status_code=422, recoverable=False
            ) from e
        except Exception as e:
            db_logger.log_error("compare_documents_detailed", e, database_name=db_name)
            raise


@router.post(
    "/databases/{db_name}/nearest-neighbors",
    response_model=NearestNeighborsResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("nearest_neighbors")
async def nearest_neighbors(db_name: str, body: NearestNeighborsBody, db=Depends(get_db)):
    """Find nearest neighbors for a document."""
    with request_context("nearest_neighbors"):
        if not hasattr(db, "nearest_neighbors"):
            raise APIError(
                message="Nearest neighbors search is not available for this database",
                error_code="FEATURE_NOT_AVAILABLE",
                status_code=501,
            )

        try:
            results = await run_in_threadpool(
                db.nearest_neighbors,
                body.doc_id,
                k=body.k,
                score_threshold=body.score_threshold,
                filters=body.filters,
            )
            serialized = [serialize_query_result(r) for r in results]
            return {
                "doc_id": body.doc_id,
                "k": body.k,
                "results": serialized,
                "total_results": len(serialized),
                "status": "success",
            }
        except MetadataFilterError:
            # A bad filter spec is a ValueError subclass but must stay a 400
            # INVALID_FILTER (via the global handler), not the 422 below.
            raise
        except ValueError as e:
            raise APIError(
                message=str(e), error_code="UNPROCESSABLE_DOCUMENT", status_code=422, recoverable=False
            ) from e
        except Exception as e:
            db_logger.log_error("nearest_neighbors", e, database_name=db_name)
            raise


@router.post(
    "/databases/{db_name}/similarity-matrix",
    response_model=SimilarityMatrixResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("pairwise_similarity_matrix")
async def pairwise_similarity_matrix(db_name: str, body: Optional[SimilarityMatrixBody] = None, db=Depends(get_db)):
    """Compute pairwise similarity matrix for documents."""
    with request_context("pairwise_similarity_matrix"):
        doc_ids = body.doc_ids if body else None

        if not hasattr(db, "pairwise_similarity_matrix"):
            raise APIError(
                message="Similarity matrix computation is not available for this database",
                error_code="FEATURE_NOT_AVAILABLE",
                status_code=501,
            )

        try:
            matrix = await run_in_threadpool(db.pairwise_similarity_matrix, doc_ids=doc_ids)
            return SimilarityMatrixResponse(**matrix.to_dict(), status="success")

        except Exception as e:
            db_logger.log_error("pairwise_similarity_matrix", e, database_name=db_name)
            raise
