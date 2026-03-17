# src/localvectordb_server/routers/factcheck.py
"""Fact-checking endpoints using LLM-based validation."""

import logging

from fastapi import APIRouter, Depends, Request

from localvectordb_server._auth import require_read_permission
from localvectordb_server._error_handlers import APIError, ValidationError, validate_required_fields
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.routers._deps import get_db, get_db_manager

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["factcheck"])


@router.post("/{db_name}/factcheck", dependencies=[Depends(require_read_permission)])
@log_performance("factcheck_single_db")
async def factcheck_single_db(db_name: str, request: Request):
    """Check text against a specific database for factual grounding.

    Request body:
        text: str - The text to fact-check
        llm_provider: str - LLM provider (anthropic/openai/gemini)
        llm_api_key: str - API key for the LLM provider (optional, can use server env)
        model: str - Model name (optional)
        similarity_threshold: float - Min similarity for evidence (default 0.3)
        min_grounding_score: float - Min grounding score (default 0.5)
        search_type: str - Search type (default "hybrid")
        top_k: int - Number of evidence documents (default 10)
    """
    with request_context("factcheck_single_db"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ["text"])

        text = data["text"]
        llm_provider = data.get("llm_provider", "anthropic")
        llm_api_key = data.get("llm_api_key")
        model = data.get("model")
        similarity_threshold = data.get("similarity_threshold", 0.3)
        min_grounding_score = data.get("min_grounding_score", 0.5)
        search_type = data.get("search_type", "hybrid")
        top_k = data.get("top_k", 10)

        db = get_db(db_name, request)

        try:
            from localvectordb.validation import FactChecker

            # Build the LLM identifier (provider or api_key)
            llm_id = llm_api_key or llm_provider

            checker = FactChecker(
                databases=db,
                llm=llm_id,
                model=model,
                similarity_threshold=similarity_threshold,
                min_grounding_score=min_grounding_score,
                search_type=search_type,
                top_k=top_k,
            )

            result = await checker.check_async(text)

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


@router.post("/factcheck", dependencies=[Depends(require_read_permission)])
@log_performance("factcheck_multi_db")
async def factcheck_multi_db(request: Request):
    """Check text against multiple databases for factual grounding.

    Request body:
        text: str - The text to fact-check
        databases: list[str] - Database names to check against (optional, defaults to all)
        llm_provider: str - LLM provider
        llm_api_key: str - API key for the LLM provider
        model: str - Model name
        similarity_threshold: float - Min similarity for evidence
        min_grounding_score: float - Min grounding score
        search_type: str - Search type
        top_k: int - Number of evidence documents per database
    """
    with request_context("factcheck_multi_db"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ["text"])

        text = data["text"]
        database_names = data.get("databases")
        llm_provider = data.get("llm_provider", "anthropic")
        llm_api_key = data.get("llm_api_key")
        model = data.get("model")
        similarity_threshold = data.get("similarity_threshold", 0.3)
        min_grounding_score = data.get("min_grounding_score", 0.5)
        search_type = data.get("search_type", "hybrid")
        top_k = data.get("top_k", 10)

        db_manager = get_db_manager(request)

        # Get list of databases to check
        if database_names:
            db_names = database_names
        else:
            db_names = db_manager.list_databases()

        if not db_names:
            raise ValidationError("No databases available for fact-checking")

        try:
            from localvectordb.validation import FactChecker

            results_by_db = {}
            llm_id = llm_api_key or llm_provider

            for db_name in db_names:
                try:
                    db = db_manager.get_db(db_name)
                    checker = FactChecker(
                        databases=db,
                        llm=llm_id,
                        model=model,
                        similarity_threshold=similarity_threshold,
                        min_grounding_score=min_grounding_score,
                        search_type=search_type,
                        top_k=top_k,
                    )

                    result = await checker.check_async(text)

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
