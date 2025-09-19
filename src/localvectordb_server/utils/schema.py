# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/utils/schema.py
"""
Shared utilities for metadata schema parsing and validation.

This module provides canonical implementations for parsing metadata schema
configurations across different parts of the LocalVectorDB server, ensuring
consistent validation and error handling.
"""

from typing import Any, Dict

from localvectordb._schema import validate_sql_identifier
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb_server._error_handlers import ValidationError


def parse_metadata_schema(schema_data: Dict[str, Any]) -> Dict[str, MetadataField]:
    """
    Parse metadata schema from configuration data with validation.
    
    This is the canonical implementation for parsing metadata schema configurations
    across the LocalVectorDB server. It handles both simple string type specifications
    and full field configuration dictionaries.
    
    Parameters
    ----------
    schema_data : Dict[str, Any]
        The schema data to parse. Can be:
        - Empty dict for no schema
        - Dict mapping field names to type strings (e.g., {"title": "text"})
        - Dict mapping field names to configuration dicts (e.g., {"title": {"type": "text", "indexed": true}})
        
    Returns
    -------
    Dict[str, MetadataField]
        Parsed metadata schema with validated fields
        
    Raises
    ------
    ValidationError
        If the schema data is invalid or contains unsafe field names
        
    Examples
    --------
    Simple string types:
    >>> parse_metadata_schema({"title": "text", "created_at": "date"})
    
    Full configuration:
    >>> parse_metadata_schema({
    ...     "title": {"type": "text", "indexed": True, "required": False},
    ...     "score": {"type": "real", "indexed": False, "required": True}
    ... })
    """
    if not schema_data:
        return {}

    if not isinstance(schema_data, dict):
        raise ValidationError("Metadata schema must be an object", field="metadata_schema")

    parsed_schema = {}
    for field_name, field_config in schema_data.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValidationError(
                "Metadata field names must be non-empty strings",
                field=f"metadata_schema.{field_name}"
            )

        # Validate SQL identifier safety
        try:
            validate_sql_identifier(field_name)
        except ValueError as e:
            raise ValidationError(
                f"Invalid field name: {e}",
                field=f"metadata_schema.{field_name}"
            )

        try:
            if isinstance(field_config, str):
                # Simple type string (e.g., "text", "integer")
                try:
                    field_type = MetadataFieldType(field_config)
                except ValueError as e:
                    raise ValidationError(
                        f"Invalid field type '{field_config}': {e}",
                        field=f"metadata_schema.{field_name}.type"
                    )
                parsed_schema[field_name] = MetadataField(type=field_type)

            elif isinstance(field_config, dict):
                # Full field configuration
                if 'type' not in field_config:
                    raise ValidationError(
                        "Field configuration must include 'type'",
                        field=f"metadata_schema.{field_name}"
                    )

                try:
                    field_type = MetadataFieldType(field_config['type'])
                except ValueError as e:
                    raise ValidationError(
                        f"Invalid field type '{field_config['type']}': {e}",
                        field=f"metadata_schema.{field_name}.type"
                    )

                # Validate boolean fields
                indexed = field_config.get('indexed', False)
                required = field_config.get('required', False)

                if not isinstance(indexed, bool):
                    raise ValidationError(
                        "Field 'indexed' must be a boolean",
                        field=f"metadata_schema.{field_name}.indexed"
                    )

                if not isinstance(required, bool):
                    raise ValidationError(
                        "Field 'required' must be a boolean",
                        field=f"metadata_schema.{field_name}.required"
                    )

                # Handle default value if present
                default_value = field_config.get('default_value')

                parsed_schema[field_name] = MetadataField(
                    type=field_type,
                    indexed=indexed,
                    required=required,
                    default_value=default_value
                )
            else:
                raise ValidationError(
                    "Field configuration must be either a type string or configuration object",
                    field=f"metadata_schema.{field_name}"
                )

        except ValidationError:
            # Re-raise ValidationError as-is
            raise
        except Exception as e:
            # Wrap other exceptions in ValidationError
            raise ValidationError(
                f"Error parsing field configuration: {e}",
                field=f"metadata_schema.{field_name}"
            )

    return parsed_schema
