# src/localvectordb_server/routers/schema.py
"""Metadata schema management routes."""

import logging

from fastapi import APIRouter, Depends, Request

from localvectordb_server._auth import require_read_permission, require_write_permission
from localvectordb_server._error_handlers import ValidationError, validate_required_fields
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.routers._deps import get_db
from localvectordb_server.utils.schema import parse_metadata_schema

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["schema"])


@router.put("/{db_name}/schema", dependencies=[Depends(require_write_permission)])
@log_performance("update_metadata_schema")
async def update_metadata_schema(db_name: str, request: Request):
    """Update the metadata schema for a database."""
    with request_context("update_metadata_schema"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate required fields
        validate_required_fields(data, ["metadata_schema"])

        # Parse metadata schema
        try:
            new_schema = parse_metadata_schema(data["metadata_schema"])
        except Exception as e:
            raise ValidationError(f"Invalid metadata schema: {str(e)}", field="metadata_schema") from e

        if not new_schema:
            raise ValidationError("Metadata schema cannot be empty", field="metadata_schema")

        # Get optional parameters
        drop_columns = data.get("drop_columns", False)
        if not isinstance(drop_columns, bool):
            raise ValidationError("drop_columns must be a boolean", field="drop_columns", value=drop_columns)

        column_mapping = data.get("column_mapping", None)
        if column_mapping is not None and not isinstance(column_mapping, dict):
            raise ValidationError("column_mapping must be a dictionary", field="column_mapping", value=column_mapping)

        try:
            db = get_db(db_name, request)

            db_logger.log_query(
                "update_metadata_schema",
                database_name=db_name,
                field_count=len(new_schema),
                drop_columns=drop_columns,
                column_mapping=column_mapping,
            )

            # Apply schema update
            changes = db.update_metadata_schema(new_schema, drop_columns=drop_columns, column_mapping=column_mapping)

            db_logger.log_query(
                "update_metadata_schema_success",
                database_name=db_name,
                added_fields=len(changes.get("added_fields", [])),
                removed_fields=len(changes.get("removed_fields", [])),
                modified_fields=len(changes.get("modified_fields", [])),
                populated_defaults=len(changes.get("populated_defaults", [])),
            )

            # Prepare response
            response_data = {
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

            return response_data

        except Exception as e:
            db_logger.log_error("update_metadata_schema", e, database_name=db_name)
            raise


@router.get("/{db_name}/schema", dependencies=[Depends(require_read_permission)])
@log_performance("get_metadata_schema_info")
def get_metadata_schema_info(db_name: str, request: Request):
    """Get detailed information about the current metadata schema."""
    with request_context("get_metadata_schema_info"):
        try:
            db = get_db(db_name, request)
            schema_info = db.get_metadata_schema_info()

            return {"database": db_name, "schema_info": schema_info, "status": "success"}

        except Exception as e:
            db_logger.log_error("get_metadata_schema_info", e, database_name=db_name)
            raise
