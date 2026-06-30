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

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

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
