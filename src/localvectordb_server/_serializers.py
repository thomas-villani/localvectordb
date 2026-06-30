"""Shared response serializers for the LocalVectorDB FastAPI server.

Single source of truth for converting core dataclasses (``QueryResult``,
``Document``) into JSON-compatible dicts, so router responses stay consistent.
"""

from typing import Any, Dict

from localvectordb.core import Document, QueryResult


def serialize_query_result(result: QueryResult) -> Dict[str, Any]:
    """Serialize a QueryResult object for JSON response."""
    data: Dict[str, Any] = {
        "id": result.id,
        "score": result.score,
        "type": result.type,
        "content": result.content,
        "metadata": result.metadata,
    }
    if result.type == "chunk" and result.document_id:
        data["document_id"] = result.document_id
    if result.position:
        data["position"] = result.position.to_dict()
    return data


def serialize_document(doc: Document) -> Dict[str, Any]:
    """Serialize a Document object for JSON response."""
    return {
        "id": doc.id,
        "content": doc.content,
        "metadata": doc.metadata,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        "content_hash": doc.content_hash,
    }
