# src/localvectordb_server/routers/databases.py
"""Database management routes."""

import logging

from fastapi import APIRouter, Depends, Request

from localvectordb_server._auth import require_read_permission, require_write_permission
from localvectordb_server._error_handlers import (
    APIError,
    ValidationError,
    validate_database_creation_params,
)
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.routers._deps import get_db, get_db_manager
from localvectordb_server.utils.schema import parse_metadata_schema

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["databases"])


@router.post("/databases", dependencies=[Depends(require_write_permission)])
async def create_database(request: Request):
    """Create a new vector database with optional metadata schema."""
    with request_context("create_database"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        data = validate_database_creation_params(data)
        name = data["name"]

        db_manager = get_db_manager(request)
        existing_dbs = db_manager.list_databases()
        if name in existing_dbs:
            raise APIError(
                message=f"Database '{name}' already exists",
                error_code="DATABASE_ALREADY_EXISTS",
                status_code=409,
                recoverable=True,
            )

        config = request.app.state.config
        db_config = config.database.copy()
        embedding_config = config.embedding.copy()

        metadata_schema = None
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


@router.get("/databases", dependencies=[Depends(require_read_permission)])
@log_performance("list_databases")
def list_databases(request: Request):
    """List all available databases."""
    with request_context("list_databases"):
        try:
            db_manager = get_db_manager(request)
            databases = db_manager.list_databases()
            return {"databases": databases, "count": len(databases)}
        except Exception as e:
            db_logger.log_error("list_databases", e)
            raise


@router.get("/{db_name}/info", dependencies=[Depends(require_read_permission)])
@log_performance("get_database_info")
def get_database_info(db_name: str, request: Request):
    """Get information about a specific database."""
    with request_context("get_database_info"):
        try:
            db = get_db(db_name, request)
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


@router.delete("/{db_name}", dependencies=[Depends(require_write_permission)])
@log_performance("delete_database")
def delete_database(db_name: str, request: Request):
    """Delete a database."""
    with request_context("delete_database"):
        db_manager = get_db_manager(request)
        success = db_manager.delete_database(db_name)
        return {
            "message": (
                f"Successfully deleted database '{db_name}'"
                if success
                else f"Database '{db_name}' not found. No action taken."
            ),
            "status": "success",
        }
