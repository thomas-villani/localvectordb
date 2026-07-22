# src/localvectordb_server/routers/schema.py
"""Metadata schema management routes (Pydantic request/response models + dependency injection)."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from localvectordb_server._auth import require_read_permission, require_write_permission
from localvectordb_server._error_handlers import ValidationError
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.routers._deps import get_db
from localvectordb_server.routers._models import StrictModel
from localvectordb_server.utils.schema import parse_metadata_schema

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["schema"])


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class UpdateMetadataSchemaBody(StrictModel):
    metadata_schema: Dict[str, Any]
    drop_columns: bool = False
    column_mapping: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class UpdateMetadataSchemaResponse(StrictModel):
    message: str
    status: str
    changes: Dict[str, Any]
    new_schema: Dict[str, Any]


class SchemaInfoResponse(StrictModel):
    database: str
    schema_info: Dict[str, Any]
    status: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.put(
    "/databases/{db_name}/schema",
    response_model=UpdateMetadataSchemaResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("update_metadata_schema")
async def update_metadata_schema(db_name: str, body: UpdateMetadataSchemaBody, db=Depends(get_db)):
    """Update the metadata schema for a database."""
    with request_context("update_metadata_schema"):
        # Parse metadata schema
        try:
            new_schema = parse_metadata_schema(body.metadata_schema)
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Invalid metadata schema: {str(e)}", field="metadata_schema") from e

        if not new_schema:
            raise ValidationError("Metadata schema cannot be empty", field="metadata_schema")

        drop_columns = body.drop_columns
        column_mapping = body.column_mapping

        try:
            db_logger.log_query(
                "update_metadata_schema",
                database_name=db_name,
                field_count=len(new_schema),
                drop_columns=drop_columns,
                column_mapping=column_mapping,
            )

            # Apply schema update (sync + blocking: offload off the event loop).
            changes = await run_in_threadpool(
                db.update_metadata_schema, new_schema, drop_columns=drop_columns, column_mapping=column_mapping
            )

            db_logger.log_query(
                "update_metadata_schema_success",
                database_name=db_name,
                added_fields=len(changes.get("added_fields", [])),
                removed_fields=len(changes.get("removed_fields", [])),
                modified_fields=len(changes.get("modified_fields", [])),
                populated_defaults=len(changes.get("populated_defaults", [])),
            )

            # Prepare response
            return {
                "message": f"Successfully updated metadata schema for database '{db_name}'",
                "status": "success",
                "changes": changes,
                "new_schema": {
                    field_name: {
                        "type": field.type.value if hasattr(field.type, "value") else str(field.type),
                        "indexed": field.indexed,
                        "required": field.required,
                        "default_value": field.default_value,
                    }
                    for field_name, field in new_schema.items()
                },
            }

        except Exception as e:
            db_logger.log_error("update_metadata_schema", e, database_name=db_name)
            raise


@router.get(
    "/databases/{db_name}/schema",
    response_model=SchemaInfoResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("get_metadata_schema_info")
def get_metadata_schema_info(db_name: str, db=Depends(get_db)):
    """Get detailed information about the current metadata schema."""
    with request_context("get_metadata_schema_info"):
        try:
            schema_info = db.get_metadata_schema_info()

            return {"database": db_name, "schema_info": schema_info, "status": "success"}

        except Exception as e:
            db_logger.log_error("get_metadata_schema_info", e, database_name=db_name)
            raise
