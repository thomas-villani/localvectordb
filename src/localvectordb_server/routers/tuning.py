# src/localvectordb_server/routers/tuning.py
"""SQLite tuning and maintenance routes (Pydantic request/response models + DI)."""

import logging
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends
from pydantic import Field

from localvectordb_server._auth import require_read_permission, require_write_permission
from localvectordb_server._error_handlers import ValidationError
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.routers._deps import get_db
from localvectordb_server.routers._models import StrictModel

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["tuning"])


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class SetTuningBody(StrictModel):
    profile: str
    overrides: Dict[str, Any] = Field(default_factory=dict)
    persist: bool = True


class CheckpointBody(StrictModel):
    mode: str = "PASSIVE"


class IncrementalVacuumBody(StrictModel):
    pages: int = 2000


class AutoTuneBody(StrictModel):
    workload: Optional[Dict[str, Any]] = None
    apply: bool = False


class CheckpointIfLargeBody(StrictModel):
    threshold_mb: Union[int, float] = 128


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class TuningResponse(StrictModel):
    database: str
    tuning: Dict[str, Any]
    status: str


class SetTuningResponse(StrictModel):
    database: str
    message: str
    tuning: Dict[str, Any]
    status: str


class CheckpointResponse(StrictModel):
    database: str
    message: str
    mode: str
    status: str


class OptimizeResponse(StrictModel):
    database: str
    message: str
    status: str


class VacuumResponse(StrictModel):
    database: str
    message: str
    warning: str
    status: str


class IncrementalVacuumResponse(StrictModel):
    database: str
    message: str
    pages: int
    status: str


class AutoTuneResponse(StrictModel):
    database: str
    recommendation: Dict[str, Any]
    status: str


class CheckpointIfLargeResponse(StrictModel):
    database: str
    checkpointed: bool
    threshold_mb: Union[int, float]
    message: str
    status: str


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.get(
    "/databases/{db_name}/tuning",
    response_model=TuningResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("get_sqlite_tuning")
def get_sqlite_tuning(db_name: str, db=Depends(get_db)):
    """Get current SQLite tuning configuration."""
    with request_context("get_sqlite_tuning"):
        try:
            tuning_db_logger = DatabaseLogger.get_logger(db_name)
            tuning_db_logger.log_query("get_sqlite_tuning", database_name=db_name)

            tuning_config = db.get_sqlite_tuning()

            return {"database": db_name, "tuning": tuning_config, "status": "success"}

        except Exception as e:
            db_logger.log_error("get_sqlite_tuning", e, database_name=db_name)
            raise


@router.put(
    "/databases/{db_name}/tuning",
    response_model=SetTuningResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("set_sqlite_tuning")
async def set_sqlite_tuning(db_name: str, body: SetTuningBody, db=Depends(get_db)):
    """Apply SQLite tuning profile with optional overrides."""
    with request_context("set_sqlite_tuning"):
        try:
            profile = body.profile
            if not profile:
                raise ValidationError("Profile name is required")

            overrides = body.overrides
            persist = body.persist

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


@router.post(
    "/databases/{db_name}/maintenance/checkpoint",
    response_model=CheckpointResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("sqlite_checkpoint")
async def sqlite_checkpoint(db_name: str, body: Optional[CheckpointBody] = None, db=Depends(get_db)):
    """Run SQLite WAL checkpoint operation."""
    with request_context("sqlite_checkpoint"):
        try:
            mode = body.mode if body else "PASSIVE"

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


@router.post(
    "/databases/{db_name}/maintenance/optimize",
    response_model=OptimizeResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("sqlite_optimize")
async def sqlite_optimize(db_name: str, db=Depends(get_db)):
    """Run SQLite PRAGMA optimize."""
    with request_context("sqlite_optimize"):
        try:
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


@router.post(
    "/databases/{db_name}/maintenance/vacuum",
    response_model=VacuumResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("sqlite_vacuum")
async def sqlite_vacuum(db_name: str, db=Depends(get_db)):
    """Run SQLite VACUUM operation."""
    with request_context("sqlite_vacuum"):
        try:
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
    "/databases/{db_name}/maintenance/incremental-vacuum",
    response_model=IncrementalVacuumResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("sqlite_incremental_vacuum")
async def sqlite_incremental_vacuum(db_name: str, body: Optional[IncrementalVacuumBody] = None, db=Depends(get_db)):
    """Run incremental VACUUM operation."""
    with request_context("sqlite_incremental_vacuum"):
        try:
            pages = body.pages if body else 2000

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


@router.post(
    "/databases/{db_name}/auto-tune",
    response_model=AutoTuneResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("auto_tune_database")
async def auto_tune_database(db_name: str, body: Optional[AutoTuneBody] = None, db=Depends(get_db)):
    """Get auto-tuning recommendations for database."""
    with request_context("auto_tune_database"):
        try:
            workload = body.workload if body else None
            apply_settings = body.apply if body else False

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
    "/databases/{db_name}/maintenance/checkpoint-if-large",
    response_model=CheckpointIfLargeResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("checkpoint_if_wal_large")
async def checkpoint_if_wal_large(db_name: str, body: Optional[CheckpointIfLargeBody] = None, db=Depends(get_db)):
    """Check if WAL is large and checkpoint if needed."""
    with request_context("checkpoint_if_wal_large"):
        try:
            threshold_mb = body.threshold_mb if body else 128

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
