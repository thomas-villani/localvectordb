"""Shared Pydantic request/response models for the FastAPI routers.

Conventions adopted in the model-driven server pass (v0.1.0):

- **Request bodies are Pydantic models.** Handlers declare a typed body instead
  of ``await request.json()`` + hand-rolled checks; invalid input yields a 422
  rendered through the standard ``{"error": {...}}`` envelope.
- **Filters** are always named ``filters`` (never ``where``) on the wire.
- **Pagination** is ``limit`` + ``offset`` (never ``page``), with a single cap.
- **Top-k** is always ``k`` (never ``top_k``).
- **Response keys** consumed by the client stay stable: ``ids``, ``count``,
  ``deleted_count``, ``documents``, ``document_ids``.
- Models ``forbid`` unknown fields so client/server drift surfaces as a 422
  rather than a silently-ignored payload key.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from localvectordb.core import DocumentScoringMethod

# Shared pagination bounds (was inconsistent: 1000 for list, 10000 for filter).
MAX_PAGE_LIMIT = 1000
DEFAULT_PAGE_LIMIT = 100


class StrictModel(BaseModel):
    """Base model that rejects unknown fields (surfaces typos as 422)."""

    model_config = ConfigDict(extra="forbid")


class MessageResponse(StrictModel):
    """Generic success message."""

    message: str


class WriteResponse(StrictModel):
    """Response for write operations that return affected document IDs."""

    ids: List[str]
    message: str


class DeleteResponse(StrictModel):
    deleted_count: int
    message: str


class UpdateResponse(StrictModel):
    updated: bool
    message: str
    # Populated by content updates and by patch: SHA-256 of the resulting content.
    new_hash: Optional[str] = None
    # Populated by patch only: number of ops applied.
    ops_applied: Optional[int] = None


class BatchDeleteResponse(StrictModel):
    deleted_count: int
    failed_ids: List[str]
    message: str


class CountResponse(StrictModel):
    count: int


class ExistsResponse(StrictModel):
    exists: Union[bool, List[bool]]
    ids: List[str]


class DocumentResponse(StrictModel):
    """Serialized Document (mirrors ``_serializers.serialize_document``)."""

    id: str
    content: str
    metadata: Dict[str, Any]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    content_hash: Optional[str] = None


class PageInfo(StrictModel):
    """limit/offset pagination metadata returned alongside list results."""

    limit: int
    offset: int
    total: int
    has_more: bool


class DocumentListResponse(StrictModel):
    documents: List[DocumentResponse]
    pagination: PageInfo


class DocumentsByIdResponse(StrictModel):
    documents: List[DocumentResponse]
    returned_ids: List[str]
    missing_ids: List[str]


class FilterBody(StrictModel):
    """Optional body carrying only metadata filters."""

    filters: Optional[Dict[str, Any]] = Field(default=None, description="MongoDB-style metadata filters")


class QueryBody(StrictModel):
    """Shared request body for the unified ``query()`` surface (query + streaming).

    Mirrors the library ``query()`` parameters and replaces the hand-rolled
    ``validate_search_params`` checks. ``filters`` accepts the legacy
    ``metadata_filters`` alias. ``return_type`` includes ``sections`` (which the
    engine supports but the old allow-list rejected).
    """

    query: str = Field(min_length=1)
    search_type: Literal["vector", "keyword", "hybrid"] = "hybrid"
    return_type: Literal["documents", "chunks", "sections", "context", "enriched"] = "documents"
    search_level: Literal["chunks", "sections", "documents"] = "chunks"
    k: int = Field(default=10, ge=1, le=1000)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = Field(
        default=None, validation_alias=AliasChoices("filters", "metadata_filters")
    )
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    context_window: int = Field(default=2, ge=0, le=1_000_000)
    context_unit: Literal["chunks", "tokens", "words", "characters"] = "chunks"
    context_truncate: bool = False
    semantic_dedup_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    document_scoring_method: DocumentScoringMethod = "frequency_boost"
    document_scoring_options: Optional[Dict[str, Any]] = None
    reranker_config: Optional[Dict[str, Any]] = None
    # Candidate-pool size fetched before reranking (only used when reranker_config
    # is set). Clamped server-side to <= 200; see database `query(rerank_k=...)`.
    rerank_k: Optional[int] = Field(default=None, ge=1, le=1000)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _check_context_window(self) -> "QueryBody":
        # ``context_window`` counts chunks when the unit is 'chunks' (keep the old
        # small ceiling); for token/word/character budgets it may be much larger.
        if self.context_unit == "chunks" and self.context_window > 20:
            raise ValueError("context_window must be between 0 and 20 when context_unit='chunks'")
        return self
