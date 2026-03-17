# src/localvectordb_server/routers/upload.py
"""File upload, extraction preview, and supported formats routes."""

import json
import logging
import mimetypes
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from werkzeug.utils import secure_filename

from localvectordb.extractors import get_extractor_registry, get_supported_formats
from localvectordb_server._auth import require_read_permission, require_write_permission
from localvectordb_server._error_handlers import APIError, ValidationError
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.routers._deps import get_db

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["upload"])


@router.post("/{db_name}/upload", dependencies=[Depends(require_write_permission)])
@log_performance("upload_files")
async def upload_files(
    db_name: str,
    request: Request,
    files: list[UploadFile] = File(...),  # noqa: B008
    metadata: str = Form(None),
    batch_size: int = Form(100),
    ids: str = Form(None),
    mode: str = Form("upsert"),
    errors: str = Form("raise"),
    similarity_threshold: float = Form(None),
    use_filename_as_id: str = Form("false"),
):
    """Upload files to the database with automatic text extraction.

    Supports both single and multiple file uploads. Files are processed to extract
    text content using appropriate extractors based on file type.
    """
    with request_context("upload_files"):
        # Check if server uploads are enabled
        config = request.app.state.config
        file_upload_enabled = getattr(config.server, "file_upload_enabled", True)
        if not file_upload_enabled:
            raise APIError(
                message="File extraction route is not enabled",
                error_code="EXTRACTION_NOT_AVAILABLE",
                status_code=503,
            )

        # Validate files
        if not files or all(f.filename == "" for f in files):
            raise ValidationError("No files selected", field="files")

        # Parse IDs
        file_ids = None
        use_filenames_as_ids = use_filename_as_id.lower() == "true"

        if ids:
            try:
                # Try JSON array first
                file_ids = json.loads(ids)
                if not isinstance(file_ids, list):
                    raise ValueError("IDs must be an array")
            except json.JSONDecodeError:
                # Fallback to comma-separated string
                file_ids = [id_str.strip() for id_str in ids.split(",") if id_str.strip()]

        # Validate IDs if provided
        if file_ids is not None:
            if len(file_ids) != len(files):
                raise ValidationError(
                    f"Number of IDs ({len(file_ids)}) must match number of files ({len(files)})",
                    field="ids",
                )

        # Parse metadata if provided
        base_metadata: list[dict] = [{}] * len(files)
        if metadata:
            try:
                base_metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                raise ValidationError("Invalid JSON in metadata field", field="metadata") from e

            if isinstance(base_metadata, list):
                if len(base_metadata) == 1:
                    base_metadata = base_metadata * len(files)
            elif isinstance(base_metadata, dict):
                base_metadata = [base_metadata] * len(files)

            if len(base_metadata) != len(files):
                raise ValidationError(
                    "Number of items in metadata array must match number of files",
                    field="metadata",
                )

        # Get default batch size from config
        default_batch_size = getattr(config.embedding, "batch_size", 100)
        if batch_size == 100:
            batch_size = default_batch_size

        # Validate batch size
        if batch_size < 1 or batch_size > 1000:
            raise ValidationError(
                "Batch size must be between 1 and 1000",
                field="batch_size",
                value=batch_size,
            )

        try:
            db = get_db(db_name, request)

            documents = []
            metadata_list = []
            extraction_results = []
            document_ids = []

            extractor_registry = get_extractor_registry()

            db_logger.log_query("upload_files", database_name=db_name, file_count=len(files))

            for file_idx, file in enumerate(files):
                if file.filename == "":
                    continue

                if file_ids is not None:
                    file_id = file_ids[file_idx]
                elif use_filenames_as_ids:
                    file_id = file.filename
                else:
                    file_id = None

                # Secure the filename
                filename = secure_filename(file.filename or "")
                if not filename:
                    filename = "uploaded_file"

                # Read file content
                file_content = await file.read()
                if len(file_content) == 0:
                    logger.warning(f"Empty file uploaded: {filename}")
                    continue

                # Get mimetype
                mimetype = file.content_type or mimetypes.guess_type(filename)[0]

                # Prepare file metadata - only include fields that exist in the database schema
                file_metadata = base_metadata[file_idx].copy()
                # Standard file upload metadata (only add if in schema)
                standard_metadata = {
                    "source": "file_upload",
                    "original_filename": file.filename,
                    "secure_filename": filename,
                    "file_size_bytes": len(file_content),
                    "mimetype": mimetype,
                    "upload_timestamp": datetime.now(UTC).isoformat(),
                }

                # Only add metadata fields that exist in the database schema
                for key, value in standard_metadata.items():
                    if key in db.metadata_schema:
                        file_metadata[key] = value

                # Extract text content
                try:
                    extraction_result = extractor_registry.extract_text(file_content, filename, mimetype)

                    if extraction_result.success:
                        documents.append(extraction_result.text)
                        document_ids.append(file_id)
                        # Add extraction metadata - only fields that exist in schema
                        extraction_metadata = {
                            "extraction_method": extraction_result.method,
                            "text_length": len(extraction_result.text),
                            **extraction_result.metadata,
                        }

                        # Filter extraction metadata to only include schema fields
                        for key, value in extraction_metadata.items():
                            if key in db.metadata_schema:
                                file_metadata[key] = value

                        extraction_results.append(
                            {
                                "filename": filename,
                                "extraction_success": True,
                                "extraction_method": extraction_result.method,
                                "text_length": len(extraction_result.text) if extraction_result.text else 0,
                                "error": None,
                                "metadata_fields_used": [
                                    key for key in extraction_result.metadata.keys() if key in db.metadata_schema
                                ],
                                "metadata_fields_ignored": [
                                    key for key in extraction_result.metadata.keys() if key not in db.metadata_schema
                                ],
                            }
                        )
                    else:
                        extraction_results.append(
                            {
                                "filename": filename,
                                "extraction_success": False,
                                "error": extraction_result.error if not extraction_result.success else None,
                                "metadata_fields_used": [],
                                "metadata_fields_ignored": [],
                            }
                        )

                except Exception as e:
                    logger.error(f"Error extracting text from {filename}: {e}")
                    extraction_results.append(
                        {
                            "filename": filename,
                            "extraction_success": False,
                            "error": str(e),
                            "metadata_fields_used": [],
                            "metadata_fields_ignored": [],
                        }
                    )

                metadata_list.append(file_metadata)

            if not documents:
                raise ValidationError("No valid files to process")

            # Insert or upsert documents to database based on mode
            if mode == "insert":
                result_ids = db.insert(
                    documents=documents,
                    metadata=metadata_list,
                    ids=document_ids if document_ids else None,
                    batch_size=batch_size,
                    similarity_threshold=similarity_threshold,
                    errors=errors,
                )
            else:
                # Default to upsert
                result_ids = db.upsert(
                    documents=documents,
                    metadata=metadata_list,
                    ids=document_ids if document_ids else None,
                    batch_size=batch_size,
                    similarity_threshold=similarity_threshold,
                )

            db_logger.log_query(
                "upload_files_success",
                database_name=db_name,
                processed_files=len(documents),
                result_count=len(result_ids),
            )

            # Prepare response
            response_data = {
                "message": f"Successfully processed {len(documents)} file(s)",
                "files_processed": len(documents),
                "document_ids": result_ids,
                "extraction_results": extraction_results,
                "status": "success",
            }

            # Add extraction summary
            successful_extractions = sum(1 for r in extraction_results if r["extraction_success"])
            response_data["extraction_summary"] = {
                "total_files": len(extraction_results),
                "successful_extractions": successful_extractions,
                "failed_extractions": len(extraction_results) - successful_extractions,
                "supported_formats": get_supported_formats(),
            }

            return response_data

        except Exception as e:
            db_logger.log_error("upload_files", e, database_name=db_name)
            raise


@router.get("/upload/supported-formats", dependencies=[Depends(require_read_permission)])
def get_upload_supported_formats(request: Request):
    """Get information about supported file formats for upload."""
    config = request.app.state.config
    file_upload_enabled = getattr(config.server, "file_upload_enabled", True)
    if not file_upload_enabled:
        raise APIError(
            message="File extraction route is not enabled",
            error_code="EXTRACTION_NOT_AVAILABLE",
            status_code=503,
        )

    supported = get_supported_formats()

    # Convert to the expected format for API response
    format_details = {}
    for format_key, format_info in supported.items():
        format_details[format_key] = {
            "extensions": format_info.get("extensions", []),
            "mimetypes": format_info.get("mimetypes", []),
            "description": f"{format_key.upper()} files",
            "extractors": format_info.get("extractors", []),
            "supported": format_info.get("available", False),
        }

    response = {
        "extraction_available": True,
        "supported_formats": format_details,
        "basic_text_support": True,
        "text_file_extensions": [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv"],
    }

    environment = getattr(config, "environment", None) or getattr(config.server, "environment", None)
    if environment == "development":
        response["installation_hints"] = {
            "pdf": "pip install pdfplumber or pip install PyPDF2",
            "docx": "pip install python-docx",
            "pptx": "pip install python-pptx",
            "xlsx": "pip install openpyxl",
            "rtf": "pip install striprtf",
        }

    return response


@router.post("/upload/extract-preview", dependencies=[Depends(require_read_permission)])
@log_performance("extract_preview")
async def extract_preview(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
):
    """Preview text extraction from uploaded files without adding to database.

    This endpoint allows testing text extraction on files before committing
    them to the database. Useful for validating extraction quality.
    """
    with request_context("extract_preview"):
        config = request.app.state.config
        file_upload_enabled = getattr(config.server, "file_upload_enabled", True)
        if not file_upload_enabled:
            raise APIError(
                message="File extraction is not enabled",
                error_code="EXTRACTION_NOT_AVAILABLE",
                status_code=503,
            )

        if file.filename == "":
            raise ValidationError("No file selected", field="file")

        try:
            # Secure the filename
            filename = secure_filename(file.filename or "")
            if not filename:
                filename = "preview_file"

            # Read file content
            file_content = await file.read()
            mimetype = file.content_type or mimetypes.guess_type(filename)[0]

            # Extract text
            extractor_registry = get_extractor_registry()
            extraction_result = extractor_registry.extract_text(file_content, filename, mimetype)

            # Prepare response
            response_data = {
                "filename": filename,
                "original_filename": file.filename,
                "file_size_bytes": len(file_content),
                "mimetype": mimetype,
                "extraction_success": extraction_result.success,
                "extraction_method": extraction_result.method,
                "extraction_metadata": extraction_result.metadata,
                "extracted_text": extraction_result.text,
                "text_length": len(extraction_result.text),
                "text_preview": (
                    extraction_result.text[:500] + "..."
                    if len(extraction_result.text) > 500
                    else extraction_result.text
                ),
            }

            if not extraction_result.success:
                response_data["extraction_error"] = extraction_result.error

            return response_data

        except Exception as e:
            logger.error(f"Error during extraction preview: {e}")
            raise APIError(
                message=f"Preview extraction failed: {str(e)}",
                error_code="EXTRACTION_PREVIEW_FAILED",
                status_code=500,
            ) from e
