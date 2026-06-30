# src/localvectordb_server/routers/factcheck.py
"""Fact-checking endpoints using LLM-based validation."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import Field

from localvectordb_server._auth import require_read_permission
from localvectordb_server._error_handlers import APIError, ValidationError
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.routers._deps import get_db, get_db_manager
from localvectordb_server.routers._models import StrictModel

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["factcheck"])


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class FactCheckBody(StrictModel):
    """Request body for fact-checking text against a single database."""

    text: str = Field(..., description="The text to fact-check")
    llm_provider: str = Field(default="anthropic", description="LLM provider (anthropic/openai/gemini)")
    llm_api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    model: Optional[str] = Field(default=None, description="Model name")
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Min similarity for evidence")
    min_grounding_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Min grounding score for claims")
    search_type: str = Field(default="hybrid", description="Search type for evidence retrieval")
    # Convention (M11): the rest of the HTTP surface uses ``k`` for top-k; accept
    # ``k`` on the wire and forward it to the library's ``top_k`` parameter.
    k: int = Field(default=10, ge=1, description="Number of evidence documents to retrieve")


class MultiFactCheckBody(FactCheckBody):
    """Request body for fact-checking text across multiple databases."""

    databases: Optional[List[str]] = Field(
        default=None, description="Database names to check against (defaults to all)"
    )


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class FactCheckEvidence(StrictModel):
    content: str
    score: Optional[float] = None
    source_id: Optional[Any] = None


class FactCheckClaim(StrictModel):
    text: str
    supported: Optional[bool] = None
    score: Optional[float] = None
    evidence: List[FactCheckEvidence] = Field(default_factory=list)


class FactCheckResponse(StrictModel):
    """Serialized single-database fact-check result (mirrors ``_serialize_factcheck_result``)."""

    grounding_score: Optional[float] = None
    verdict: Optional[Any] = None
    claims: Optional[List[FactCheckClaim]] = None
    evidence: Optional[List[FactCheckEvidence]] = None
    explanation: Optional[str] = None
    database: str
    status: str


class MultiFactCheckResponse(StrictModel):
    """Per-database fact-check results across multiple databases."""

    results: Dict[str, Any]
    databases_checked: List[str]
    status: str


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.post(
    "/{db_name}/factcheck",
    response_model=FactCheckResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("factcheck_single_db")
async def factcheck_single_db(db_name: str, body: FactCheckBody, db=Depends(get_db)):
    """Check text against a specific database for factual grounding."""
    with request_context("factcheck_single_db"):
        try:
            from localvectordb.validation import FactChecker

            # Build the LLM identifier (provider or api_key)
            llm_id = body.llm_api_key or body.llm_provider

            checker = FactChecker(
                databases=db,
                llm=llm_id,
                model=body.model,
                similarity_threshold=body.similarity_threshold,
                min_grounding_score=body.min_grounding_score,
                search_type=body.search_type,
                top_k=body.k,
            )

            result = await checker.check_async(body.text)

            # Serialize result
            response = _serialize_factcheck_result(result)
            response["database"] = db_name
            response["status"] = "success"
            return response

        except ImportError as exc:
            raise APIError(
                message="Fact-checking requires the validation module",
                error_code="FEATURE_NOT_AVAILABLE",
                status_code=501,
            ) from exc
        except Exception as e:
            db_logger.log_error("factcheck_single_db", e, database_name=db_name)
            raise


@router.post(
    "/factcheck",
    response_model=MultiFactCheckResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("factcheck_multi_db")
async def factcheck_multi_db(body: MultiFactCheckBody, db_manager=Depends(get_db_manager)):
    """Check text against multiple databases for factual grounding."""
    with request_context("factcheck_multi_db"):
        # Get list of databases to check
        if body.databases:
            db_names = body.databases
        else:
            db_names = db_manager.list_databases()

        if not db_names:
            raise ValidationError("No databases available for fact-checking")

        try:
            from localvectordb.validation import FactChecker

            results_by_db: Dict[str, Any] = {}
            llm_id = body.llm_api_key or body.llm_provider

            for db_name in db_names:
                try:
                    db = db_manager.get_db(db_name)
                    checker = FactChecker(
                        databases=db,
                        llm=llm_id,
                        model=body.model,
                        similarity_threshold=body.similarity_threshold,
                        min_grounding_score=body.min_grounding_score,
                        search_type=body.search_type,
                        top_k=body.k,
                    )

                    result = await checker.check_async(body.text)

                    results_by_db[db_name] = _serialize_factcheck_result(result)

                except Exception as e:
                    logger.error(f"Fact-check error for database '{db_name}': {e}")
                    results_by_db[db_name] = {"error": str(e)}

            return {
                "results": results_by_db,
                "databases_checked": list(results_by_db.keys()),
                "status": "success",
            }

        except ImportError as exc:
            raise APIError(
                message="Fact-checking requires the validation module",
                error_code="FEATURE_NOT_AVAILABLE",
                status_code=501,
            ) from exc
        except Exception as e:
            db_logger.log_error("factcheck_multi_db", e)
            raise


def _serialize_factcheck_result(result) -> dict:
    """Serialize a FactCheckResult to a JSON-compatible dict."""
    response = {}

    if hasattr(result, "grounding_score"):
        response["grounding_score"] = result.grounding_score
    if hasattr(result, "verdict"):
        response["verdict"] = result.verdict
    if hasattr(result, "claims"):
        response["claims"] = [
            {
                "text": getattr(c, "text", str(c)),
                "supported": getattr(c, "supported", None),
                "score": getattr(c, "score", None),
                "evidence": [
                    {
                        "content": getattr(e, "content", str(e)),
                        "score": getattr(e, "score", None),
                        "source_id": getattr(e, "source_id", getattr(e, "id", None)),
                    }
                    for e in getattr(c, "evidence", [])
                ],
            }
            for c in result.claims
        ]
    if hasattr(result, "evidence"):
        response["evidence"] = [
            {
                "content": getattr(e, "content", str(e)),
                "score": getattr(e, "score", None),
                "source_id": getattr(e, "source_id", getattr(e, "id", None)),
            }
            for e in result.evidence
        ]
    if hasattr(result, "explanation"):
        response["explanation"] = result.explanation

    return response
