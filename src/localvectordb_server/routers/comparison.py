# src/localvectordb_server/routers/comparison.py
"""Document comparison endpoints exposing ComparisonMixin methods."""

import logging

from fastapi import APIRouter, Depends, Request

from localvectordb_server._auth import require_read_permission
from localvectordb_server._error_handlers import APIError, ValidationError, validate_required_fields
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server._serializers import serialize_query_result
from localvectordb_server.routers._deps import get_db

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["comparison"])


@router.post("/{db_name}/compare", dependencies=[Depends(require_read_permission)])
@log_performance("compare_documents")
async def compare_documents(db_name: str, request: Request):
    """Compare two documents by their IDs and return similarity score."""
    with request_context("compare_documents"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ["doc_id_1", "doc_id_2"])

        doc_id_1 = data["doc_id_1"]
        doc_id_2 = data["doc_id_2"]

        db = get_db(db_name, request)

        if not hasattr(db, "compare_documents"):
            raise APIError(
                message="Document comparison is not available for this database",
                error_code="FEATURE_NOT_AVAILABLE",
                status_code=501,
            )

        try:
            similarity = db.compare_documents(doc_id_1, doc_id_2)
            return {
                "doc_id_1": doc_id_1,
                "doc_id_2": doc_id_2,
                "similarity": similarity,
                "status": "success",
            }
        except Exception as e:
            db_logger.log_error("compare_documents", e, database_name=db_name)
            raise


@router.post("/{db_name}/compare/detailed", dependencies=[Depends(require_read_permission)])
@log_performance("compare_documents_detailed")
async def compare_documents_detailed(db_name: str, request: Request):
    """Compare two documents with detailed chunk-level analysis."""
    with request_context("compare_documents_detailed"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ["doc_id_1", "doc_id_2"])

        doc_id_1 = data["doc_id_1"]
        doc_id_2 = data["doc_id_2"]
        chunk_threshold = data.get("chunk_threshold", 0.7)

        db = get_db(db_name, request)

        if not hasattr(db, "compare_documents_detailed"):
            raise APIError(
                message="Detailed document comparison is not available for this database",
                error_code="FEATURE_NOT_AVAILABLE",
                status_code=501,
            )

        try:
            result = db.compare_documents_detailed(doc_id_1, doc_id_2, chunk_threshold=chunk_threshold)

            # Serialize the DocumentComparisonResult
            response = {
                "doc_id_1": doc_id_1,
                "doc_id_2": doc_id_2,
                "overall_similarity": result.overall_similarity,
                "chunk_similarities": result.chunk_similarities if hasattr(result, "chunk_similarities") else [],
                "status": "success",
            }

            # Include additional fields if available
            if hasattr(result, "common_themes"):
                response["common_themes"] = result.common_themes
            if hasattr(result, "unique_to_doc1"):
                response["unique_to_doc1"] = result.unique_to_doc1
            if hasattr(result, "unique_to_doc2"):
                response["unique_to_doc2"] = result.unique_to_doc2

            return response

        except Exception as e:
            db_logger.log_error("compare_documents_detailed", e, database_name=db_name)
            raise


@router.post("/{db_name}/nearest-neighbors", dependencies=[Depends(require_read_permission)])
@log_performance("nearest_neighbors")
async def nearest_neighbors(db_name: str, request: Request):
    """Find nearest neighbors for a document."""
    with request_context("nearest_neighbors"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ["doc_id"])

        doc_id = data["doc_id"]
        k = data.get("k", 5)

        db = get_db(db_name, request)

        if not hasattr(db, "nearest_neighbors"):
            raise APIError(
                message="Nearest neighbors search is not available for this database",
                error_code="FEATURE_NOT_AVAILABLE",
                status_code=501,
            )

        try:
            results = db.nearest_neighbors(doc_id, k=k)
            serialized = [serialize_query_result(r) for r in results]
            return {
                "doc_id": doc_id,
                "k": k,
                "results": serialized,
                "total_results": len(serialized),
                "status": "success",
            }
        except Exception as e:
            db_logger.log_error("nearest_neighbors", e, database_name=db_name)
            raise


@router.post("/{db_name}/similarity-matrix", dependencies=[Depends(require_read_permission)])
@log_performance("pairwise_similarity_matrix")
async def pairwise_similarity_matrix(db_name: str, request: Request):
    """Compute pairwise similarity matrix for documents."""
    with request_context("pairwise_similarity_matrix"):
        data = await request.json()
        if not data:
            data = {}

        doc_ids = data.get("doc_ids")

        db = get_db(db_name, request)

        if not hasattr(db, "pairwise_similarity_matrix"):
            raise APIError(
                message="Similarity matrix computation is not available for this database",
                error_code="FEATURE_NOT_AVAILABLE",
                status_code=501,
            )

        try:
            matrix = db.pairwise_similarity_matrix(doc_ids=doc_ids)

            # Serialize the DocumentSimilarityMatrix
            response = {
                "doc_ids": matrix.doc_ids if hasattr(matrix, "doc_ids") else doc_ids,
                "status": "success",
            }

            if hasattr(matrix, "matrix"):
                # Convert numpy array to list if needed
                mat = matrix.matrix
                if hasattr(mat, "tolist"):
                    response["matrix"] = mat.tolist()
                else:
                    response["matrix"] = mat
            if hasattr(matrix, "similarity_pairs"):
                response["similarity_pairs"] = matrix.similarity_pairs

            return response

        except Exception as e:
            db_logger.log_error("pairwise_similarity_matrix", e, database_name=db_name)
            raise
