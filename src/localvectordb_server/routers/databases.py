# src/localvectordb_server/routers/databases.py
"""Database management routes (Pydantic request/response models + dependency injection)."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import ConfigDict, Field

from localvectordb_server._auth import require_read_permission, require_write_permission
from localvectordb_server._error_handlers import (
    APIError,
    ValidationError,
    validate_database_creation_params,
)
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.config import Config
from localvectordb_server.routers._deps import get_config, get_db, get_db_manager
from localvectordb_server.routers._models import StrictModel
from localvectordb_server.utils.schema import parse_metadata_schema

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["databases"])


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class CreateDatabaseBody(StrictModel):
    # NOTE: extra="ignore" (not the StrictModel default "forbid"). The SDK's
    # _create_database still posts a legacy flat payload (embedding_provider,
    # chunk_size, ...) that this endpoint has always ignored in favor of server
    # defaults. Forbidding extras would 400 every remote create. Honoring that
    # config end-to-end is tracked as a follow-up; until then we preserve the
    # lenient behavior. (The provider whitelist itself now defers to
    # EmbeddingRegistry — see _error_handlers.validate_database_creation_params.)
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1)
    metadata_schema: Optional[Dict[str, Any]] = None
    database: Optional[Dict[str, Any]] = None
    embedding: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class MetadataFieldInfo(StrictModel):
    type: str
    indexed: bool
    required: bool
    default_value: Any = None


class DatabaseConfigInfo(StrictModel):
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    chunking_method: str
    chunk_size: int
    chunk_overlap: int
    metadata_schema: Dict[str, MetadataFieldInfo]
    fts_enabled: bool


class CreateDatabaseConfigInfo(DatabaseConfigInfo):
    name: str


class CreateDatabaseResponse(StrictModel):
    message: str
    status: str
    config: CreateDatabaseConfigInfo


class DatabaseInfoResponse(StrictModel):
    name: str
    stats: Dict[str, Any]
    config: DatabaseConfigInfo


class DatabaseListResponse(StrictModel):
    databases: List[str]
    count: int


class DeleteDatabaseResponse(StrictModel):
    message: str
    status: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post(
    "/databases",
    response_model=CreateDatabaseResponse,
    dependencies=[Depends(require_write_permission)],
)
async def create_database(
    body: CreateDatabaseBody,
    db_manager=Depends(get_db_manager),
    config: Config = Depends(get_config),
):
    """Create a new vector database with optional metadata schema."""
    with request_context("create_database"):
        # Reconstruct the payload dict so the existing semantic validation (name
        # character checks, chunk_size bounds, embedding provider whitelist) keeps
        # running unchanged. Free-form config dicts stay untyped on the wire.
        data: Dict[str, Any] = {"name": body.name}
        if body.metadata_schema is not None:
            data["metadata_schema"] = body.metadata_schema
        if body.database is not None:
            data["database"] = body.database
        if body.embedding is not None:
            data["embedding"] = body.embedding

        data = validate_database_creation_params(data)
        name = data["name"]

        existing_dbs = db_manager.list_databases()
        if name in existing_dbs:
            raise APIError(
                message=f"Database '{name}' already exists",
                error_code="DATABASE_ALREADY_EXISTS",
                status_code=409,
                recoverable=True,
            )

        db_config = config.database.copy()
        embedding_config = config.embedding.copy()

        if "metadata_schema" in data:
            metadata_schema = parse_metadata_schema(data["metadata_schema"])
        else:
            metadata_schema = db_config.default_metadata_schema

        if "database" in data:
            try:
                db_config.update_from_dict(data["database"])
            except Exception as e:
                raise ValidationError(f"Invalid database configuration: {str(e)}") from e

        if "embedding" in data:
            try:
                embedding_config.update_from_dict(data["embedding"])
            except Exception as e:
                raise ValidationError(f"Invalid embedding configuration: {str(e)}") from e

        db_logger.log_query("create_database", database_name=name)

        try:
            db = db_manager.create_db(name, metadata_schema, db_config, embedding_config)
            db_logger.log_query("create_database_success", database_name=name)

            return {
                "message": f"Successfully created database '{name}'",
                "status": "success",
                "config": {
                    "name": db.name,
                    "embedding_provider": db.embedding_provider.provider_name,
                    "embedding_model": db.embedding_provider.model,
                    "embedding_dimension": db.embedding_dimension,
                    "chunking_method": db.chunking_method,
                    "chunk_size": db.chunk_size,
                    "chunk_overlap": db.chunk_overlap,
                    "metadata_schema": {
                        field_name: {
                            "type": field.type.value,
                            "indexed": field.indexed,
                            "required": field.required,
                            "default_value": field.default_value,
                        }
                        for field_name, field in (db.metadata_schema or {}).items()
                    },
                    "fts_enabled": db.fts_enabled,
                },
            }
        except Exception as e:
            db_logger.log_error("create_database", e, database_name=name)
            raise


@router.get(
    "/databases",
    response_model=DatabaseListResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("list_databases")
def list_databases(db_manager=Depends(get_db_manager)):
    """List all available databases."""
    with request_context("list_databases"):
        try:
            databases = db_manager.list_databases()
            return {"databases": databases, "count": len(databases)}
        except Exception as e:
            db_logger.log_error("list_databases", e)
            raise


@router.get(
    "/{db_name}/info",
    response_model=DatabaseInfoResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("get_database_info")
def get_database_info(db_name: str, db=Depends(get_db)):
    """Get information about a specific database."""
    with request_context("get_database_info"):
        try:
            stats = db.get_stats()
            return {
                "name": db.name,
                "stats": stats,
                "config": {
                    "embedding_provider": db.embedding_provider.provider_name,
                    "embedding_model": db.embedding_provider.model,
                    "embedding_dimension": db.embedding_dimension,
                    "chunking_method": db.chunking_method,
                    "chunk_size": db.chunk_size,
                    "chunk_overlap": db.chunk_overlap,
                    "metadata_schema": {
                        field_name: {
                            "type": field.type.value,
                            "indexed": field.indexed,
                            "required": field.required,
                            "default_value": field.default_value,
                        }
                        for field_name, field in (db.metadata_schema or {}).items()
                    },
                    "fts_enabled": db.fts_enabled,
                },
            }
        except Exception as e:
            db_logger.log_error("get_database_info", e, database_name=db_name)
            raise


@router.delete(
    "/{db_name}",
    response_model=DeleteDatabaseResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("delete_database")
def delete_database(db_name: str, db_manager=Depends(get_db_manager)):
    """Delete a database."""
    with request_context("delete_database"):
        success = db_manager.delete_database(db_name)
        return {
            "message": (
                f"Successfully deleted database '{db_name}'"
                if success
                else f"Database '{db_name}' not found. No action taken."
            ),
            "status": "success",
        }
