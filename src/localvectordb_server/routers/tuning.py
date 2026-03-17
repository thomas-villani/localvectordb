# src/localvectordb_server/routers/tuning.py
"""SQLite tuning and maintenance routes."""

import logging

from fastapi import APIRouter, Depends, Request

from localvectordb_server._auth import require_read_permission, require_write_permission
from localvectordb_server._error_handlers import ValidationError
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.routers._deps import get_db

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["tuning"])


@router.get("/{db_name}/tuning", dependencies=[Depends(require_read_permission)])
@log_performance("get_sqlite_tuning")
def get_sqlite_tuning(db_name: str, request: Request):
    """Get current SQLite tuning configuration."""
    with request_context("get_sqlite_tuning"):
        try:
            db = get_db(db_name, request)

            tuning_db_logger = DatabaseLogger.get_logger(db_name)
            tuning_db_logger.log_query("get_sqlite_tuning", database_name=db_name)

            tuning_config = db.get_sqlite_tuning()

            return {"database": db_name, "tuning": tuning_config, "status": "success"}

        except Exception as e:
            db_logger.log_error("get_sqlite_tuning", e, database_name=db_name)
            raise


@router.put("/{db_name}/tuning", dependencies=[Depends(require_write_permission)])
@log_performance("set_sqlite_tuning")
async def set_sqlite_tuning(db_name: str, request: Request):
    """Apply SQLite tuning profile with optional overrides."""
    with request_context("set_sqlite_tuning"):
        try:
            db = get_db(db_name, request)

            # Parse request data
            data = await request.json()
            if not data:
                raise ValidationError("Request body must contain JSON data")

            profile = data.get("profile")
            if not profile:
                raise ValidationError("Profile name is required")

            overrides = data.get("overrides", {})
            persist = data.get("persist", True)

            tuning_db_logger = DatabaseLogger.get_logger(db_name)
            tuning_db_logger.log_query(
                "set_sqlite_tuning",
                database_name=db_name,
                profile=profile,
                override_count=len(overrides),
                persist=persist,
            )

            # Apply tuning
            db.set_sqlite_tuning(profile, overrides, persist)

            # Get updated configuration
            new_config = db.get_sqlite_tuning()

            tuning_db_logger.log_query("set_sqlite_tuning_success", database_name=db_name, profile=profile)

            return {
                "database": db_name,
                "message": f"Applied SQLite tuning profile '{profile}'",
                "tuning": new_config,
                "status": "success",
            }

        except Exception as e:
            db_logger.log_error("set_sqlite_tuning", e, database_name=db_name)
            raise


@router.post("/{db_name}/maintenance/checkpoint", dependencies=[Depends(require_write_permission)])
@log_performance("sqlite_checkpoint")
async def sqlite_checkpoint(db_name: str, request: Request):
    """Run SQLite WAL checkpoint operation."""
    with request_context("sqlite_checkpoint"):
        try:
            db = get_db(db_name, request)

            # Parse request data
            data = await request.json() or {}
            mode = data.get("mode", "PASSIVE")

            # Validate mode
            valid_modes = ["PASSIVE", "FULL", "RESTART", "TRUNCATE"]
            if mode.upper() not in valid_modes:
                raise ValidationError(f"Invalid checkpoint mode '{mode}'. Valid modes: {valid_modes}")

            tuning_db_logger = DatabaseLogger.get_logger(db_name)
            tuning_db_logger.log_query("sqlite_checkpoint", database_name=db_name, mode=mode)

            # Run checkpoint
            db.sqlite_checkpoint(mode)

            tuning_db_logger.log_query("sqlite_checkpoint_success", database_name=db_name, mode=mode)

            return {
                "database": db_name,
                "message": f"SQLite WAL checkpoint completed with mode '{mode}'",
                "mode": mode,
                "status": "success",
            }

        except Exception as e:
            db_logger.log_error("sqlite_checkpoint", e, database_name=db_name)
            raise


@router.post("/{db_name}/maintenance/optimize", dependencies=[Depends(require_write_permission)])
@log_performance("sqlite_optimize")
async def sqlite_optimize(db_name: str, request: Request):
    """Run SQLite PRAGMA optimize."""
    with request_context("sqlite_optimize"):
        try:
            db = get_db(db_name, request)

            tuning_db_logger = DatabaseLogger.get_logger(db_name)
            tuning_db_logger.log_query("sqlite_optimize", database_name=db_name)

            # Run optimize
            db.sqlite_optimize()

            tuning_db_logger.log_query("sqlite_optimize_success", database_name=db_name)

            return {
                "database": db_name,
                "message": "SQLite PRAGMA optimize completed",
                "status": "success",
            }

        except Exception as e:
            db_logger.log_error("sqlite_optimize", e, database_name=db_name)
            raise


@router.post("/{db_name}/maintenance/vacuum", dependencies=[Depends(require_write_permission)])
@log_performance("sqlite_vacuum")
async def sqlite_vacuum(db_name: str, request: Request):
    """Run SQLite VACUUM operation."""
    with request_context("sqlite_vacuum"):
        try:
            db = get_db(db_name, request)

            tuning_db_logger = DatabaseLogger.get_logger(db_name)
            tuning_db_logger.log_query("sqlite_vacuum", database_name=db_name)

            # Run vacuum
            db.sqlite_vacuum()

            tuning_db_logger.log_query("sqlite_vacuum_success", database_name=db_name)

            return {
                "database": db_name,
                "message": "SQLite VACUUM completed",
                "warning": "This operation requires exclusive database access and may take significant time",
                "status": "success",
            }

        except Exception as e:
            db_logger.log_error("sqlite_vacuum", e, database_name=db_name)
            raise


@router.post(
    "/{db_name}/maintenance/incremental_vacuum",
    dependencies=[Depends(require_write_permission)],
)
@log_performance("sqlite_incremental_vacuum")
async def sqlite_incremental_vacuum(db_name: str, request: Request):
    """Run incremental VACUUM operation."""
    with request_context("sqlite_incremental_vacuum"):
        try:
            db = get_db(db_name, request)

            # Parse request data
            data = await request.json() or {}
            pages = data.get("pages", 2000)

            # Validate pages parameter
            if not isinstance(pages, int) or pages <= 0:
                raise ValidationError("Pages parameter must be a positive integer")

            tuning_db_logger = DatabaseLogger.get_logger(db_name)
            tuning_db_logger.log_query("sqlite_incremental_vacuum", database_name=db_name, pages=pages)

            # Run incremental vacuum
            db.sqlite_incremental_vacuum(pages)

            tuning_db_logger.log_query("sqlite_incremental_vacuum_success", database_name=db_name, pages=pages)

            return {
                "database": db_name,
                "message": f"SQLite incremental vacuum completed for {pages} pages",
                "pages": pages,
                "status": "success",
            }

        except Exception as e:
            db_logger.log_error("sqlite_incremental_vacuum", e, database_name=db_name)
            raise


@router.post("/{db_name}/auto-tune", dependencies=[Depends(require_write_permission)])
@log_performance("auto_tune_database")
async def auto_tune_database(db_name: str, request: Request):
    """Get auto-tuning recommendations for database."""
    with request_context("auto_tune_database"):
        try:
            db = get_db(db_name, request)

            # Parse request data
            data = await request.json() or {}
            workload = data.get("workload")
            apply_settings = data.get("apply", False)

            tuning_db_logger = DatabaseLogger.get_logger(db_name)
            tuning_db_logger.log_query(
                "auto_tune_database",
                database_name=db_name,
                apply_settings=apply_settings,
                has_workload=workload is not None,
            )

            # Get auto-tuning recommendations
            recommendation = db.auto_tune(workload=workload, interactive=False, apply=apply_settings)

            tuning_db_logger.log_query(
                "auto_tune_database_success",
                database_name=db_name,
                recommended_profile=recommendation["profile_name"],
                override_count=len(recommendation["pragma_overrides"]),
                applied=recommendation["applied"],
            )

            return {
                "database": db_name,
                "recommendation": recommendation,
                "status": "success",
            }

        except Exception as e:
            db_logger.log_error("auto_tune_database", e, database_name=db_name)
            raise


@router.post(
    "/{db_name}/maintenance/checkpoint_if_large",
    dependencies=[Depends(require_write_permission)],
)
@log_performance("checkpoint_if_wal_large")
async def checkpoint_if_wal_large(db_name: str, request: Request):
    """Check if WAL is large and checkpoint if needed."""
    with request_context("checkpoint_if_wal_large"):
        try:
            db = get_db(db_name, request)

            # Parse request data
            data = await request.json() or {}
            threshold_mb = data.get("threshold_mb", 128)

            # Validate threshold
            if not isinstance(threshold_mb, (int, float)) or threshold_mb <= 0:
                raise ValidationError("Threshold must be a positive number")

            tuning_db_logger = DatabaseLogger.get_logger(db_name)
            tuning_db_logger.log_query("checkpoint_if_wal_large", database_name=db_name, threshold_mb=threshold_mb)

            # Check and checkpoint if needed
            checkpointed = db.checkpoint_if_wal_large(threshold_mb)

            tuning_db_logger.log_query(
                "checkpoint_if_wal_large_success",
                database_name=db_name,
                checkpointed=checkpointed,
            )

            return {
                "database": db_name,
                "checkpointed": checkpointed,
                "threshold_mb": threshold_mb,
                "message": ("Checkpointed large WAL file" if checkpointed else "WAL file below threshold"),
                "status": "success",
            }

        except Exception as e:
            db_logger.log_error("checkpoint_if_wal_large", e, database_name=db_name)
            raise
