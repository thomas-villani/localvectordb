# src/localvectordb_server/routers/health.py
"""Health check and system resource routes."""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Response, status

from localvectordb.utils import get_system_version
from localvectordb_server._auth import require_read_permission
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server.utils.checkdeps import check_ollama_service

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["health"])


@router.get("/health")
@log_performance("health_check")
def health_check(response: Response):
    """System health check endpoint."""
    try:
        return {
            "status": "healthy",
            "version": get_system_version(),
            "ollama_available": check_ollama_service(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "error": "Health check failed"}


@router.get("/system/resources", dependencies=[Depends(require_read_permission)])
@log_performance("analyze_system_resources")
def analyze_system_resources():
    """Analyze system resources for tuning recommendations."""
    with request_context("analyze_system_resources"):
        try:
            from localvectordb.sqlite_tuning import AutoTuner

            system_info = AutoTuner.analyze_system()
            resources = {
                "total_ram_mb": system_info.total_ram_mb,
                "available_ram_mb": system_info.available_ram_mb,
                "cpu_cores": system_info.cpu_cores,
                "disk_type": system_info.disk_type,
                "disk_free_gb": system_info.disk_free_gb,
                "os_type": system_info.os_type,
            }
            return {"system_resources": resources, "status": "success"}
        except Exception as e:
            db_logger.log_error("analyze_system_resources", e)
            raise
