"""
Tuning mixin for SQLite performance optimization.

This module provides a shared interface for SQLite tuning operations that can be
used by both LocalVectorDB and RemoteVectorDB classes to ensure API parity.

Classes
-------
TuningMixin
    Shared interface for SQLite tuning operations
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, cast

from localvectordb.database.base import LocalVectorDBBase
from localvectordb.sqlite_tuning import (
    SQLITE_PRAGMA_PROFILES,
    AutoTuner,
    SqliteProfile,
    WorkloadProfile,
    list_profiles,
)

logger = logging.getLogger(__name__)


class TuningMixin(ABC):
    """
    Mixin class providing SQLite tuning interface for vector databases.

    This mixin defines the common interface for SQLite performance tuning
    that is implemented by both LocalVectorDB and RemoteVectorDB classes.
    """

    @abstractmethod
    def get_sqlite_tuning(self) -> Dict[str, Any]:
        """
        Get current SQLite tuning configuration.

        Returns
        -------
        Dict[str, Any]
            Current tuning configuration containing:
            - profile: Current profile name
            - pragmas: Current pragma settings
            - overrides: Profile overrides
        """
        pass

    @abstractmethod
    def set_sqlite_tuning(self, profile: str, overrides: Optional[Dict[str, Any]] = None, persist: bool = True) -> None:
        """
        Apply SQLite tuning profile with optional overrides.

        Parameters
        ----------
        profile : str
            Name of the tuning profile to apply
        overrides : Dict[str, Any], optional
            Pragma overrides for the profile
        persist : bool, optional
            Whether to persist settings to database config, by default True

        Raises
        ------
        ValueError
            If profile name is not recognized
        """
        pass

    @abstractmethod
    def sqlite_checkpoint(self, mode: str = "PASSIVE") -> None:
        """
        Run SQLite WAL checkpoint operation.

        Parameters
        ----------
        mode : str, optional
            Checkpoint mode (PASSIVE, FULL, RESTART, TRUNCATE), by default "PASSIVE"
        """
        pass

    @abstractmethod
    def sqlite_optimize(self) -> None:
        """Run SQLite PRAGMA optimize to update query planner statistics."""
        pass

    @abstractmethod
    def sqlite_vacuum(self) -> None:
        """
        Run SQLite VACUUM operation.

        Warning
        -------
        This operation requires exclusive database access and may take significant time.
        """
        pass

    @abstractmethod
    def sqlite_incremental_vacuum(self, pages: int = 2000) -> None:
        """
        Run incremental VACUUM operation.

        Parameters
        ----------
        pages : int, optional
            Number of pages to reclaim, by default 2000
        """
        pass

    def list_sqlite_profiles(self) -> Dict[str, str]:
        """
        List available SQLite tuning profiles.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping profile names to descriptions
        """
        return {name: desc for name, desc in list_profiles()}

    def analyze_system_resources(self) -> Dict[str, Any]:
        """
        Analyze system resources for tuning recommendations.

        Returns
        -------
        Dict[str, Any]
            System resource information
        """
        system_info = AutoTuner.analyze_system()
        return {
            "total_ram_mb": system_info.total_ram_mb,
            "available_ram_mb": system_info.available_ram_mb,
            "cpu_cores": system_info.cpu_cores,
            "disk_type": system_info.disk_type,
            "disk_free_gb": system_info.disk_free_gb,
            "os_type": system_info.os_type,
        }

    def auto_tune(
        self, workload: Optional[Dict[str, Any]] = None, interactive: bool = False, apply: bool = False
    ) -> Dict[str, Any]:
        """
        Get auto-tuning recommendations based on system and workload.

        Parameters
        ----------
        workload : Dict[str, Any], optional
            Workload characteristics. If None and interactive=True, will prompt user.
        interactive : bool, optional
            Whether to run interactive interview for workload, by default False
        apply : bool, optional
            Whether to apply the recommended settings, by default False

        Returns
        -------
        Dict[str, Any]
            Tuning recommendation containing:
            - profile_name: Recommended profile
            - pragma_overrides: Recommended pragma overrides
            - reasoning: List of reasoning explanations
            - estimated_memory_mb: Estimated memory usage
        """
        system_info = AutoTuner.analyze_system()

        if interactive:
            workload_profile = AutoTuner.interview_user_cli()
        elif workload:
            # Convert dict to WorkloadProfile
            from localvectordb.sqlite_tuning import DurabilityLevel, WorkloadType

            workload_profile = WorkloadProfile(
                workload_type=WorkloadType(workload.get("workload_type", "balanced")),
                document_size=workload.get("document_size", "medium"),
                concurrent_users=workload.get("concurrent_users", 5),
                durability_level=DurabilityLevel(workload.get("durability_level", "normal")),
                memory_constraint=workload.get("memory_constraint", "moderate"),
            )
        else:
            # Use balanced defaults
            from localvectordb.sqlite_tuning import DurabilityLevel, WorkloadType

            workload_profile = WorkloadProfile(
                workload_type=WorkloadType.BALANCED,
                document_size="medium",
                concurrent_users=5,
                durability_level=DurabilityLevel.NORMAL,
                memory_constraint="moderate",
            )

        recommendation = AutoTuner.recommend_profile(system_info, workload_profile)

        result = {
            "profile_name": recommendation.profile_name,
            "pragma_overrides": recommendation.pragma_overrides,
            "reasoning": recommendation.reasoning,
            "estimated_memory_mb": recommendation.estimated_memory_mb,
            "current_settings": self.get_sqlite_tuning(),
        }

        if apply:
            self.set_sqlite_tuning(recommendation.profile_name, recommendation.pragma_overrides, persist=True)
            result["applied"] = True
        else:
            result["applied"] = False

        return result

    def checkpoint_if_wal_large(self, wal_mb_threshold: int = 128) -> bool:
        """
        Check if WAL file is large and checkpoint if needed.

        Parameters
        ----------
        wal_mb_threshold : int, optional
            WAL size threshold in MB, by default 128

        Returns
        -------
        bool
            True if checkpoint was performed, False otherwise
        """
        # This is implemented differently for local vs remote
        # Local implementation checks file size directly
        # Remote implementation calls server endpoint
        # Subclasses should override this method
        return False


class LocalTuningMixin(LocalVectorDBBase, TuningMixin, ABC):
    """
    Local implementation of tuning mixin for LocalVectorDB.

    This class provides the concrete implementation of tuning operations
    for local SQLite databases.
    """

    def get_sqlite_tuning(self) -> Dict[str, Any]:
        """Get current SQLite tuning configuration from local database."""
        # Access stored configuration
        config = {
            "profile": getattr(self, "_sqlite_profile", "balanced"),
            "overrides": getattr(self, "_sqlite_pragma_overrides", {}),
            "pragmas": getattr(self, "_sqlite_pragmas", {}),
        }
        return config

    def set_sqlite_tuning(self, profile: str, overrides: Optional[Dict[str, Any]] = None, persist: bool = True) -> None:
        """Apply SQLite tuning profile to local database."""
        if profile not in SQLITE_PRAGMA_PROFILES:
            raise ValueError(f"Unknown SQLite profile '{profile}'. Available: {list(SQLITE_PRAGMA_PROFILES.keys())}")

        # Get base profile pragmas
        base_pragmas = dict(SQLITE_PRAGMA_PROFILES[profile].pragmas)

        # Apply overrides
        if overrides:
            base_pragmas.update(overrides)

        # Store configuration
        self._sqlite_profile = cast(SqliteProfile, profile)
        self._sqlite_pragma_overrides = overrides or {}
        self._sqlite_pragmas = base_pragmas

        # Apply to existing connection pools
        if hasattr(self, "connection_pool"):
            self.connection_pool._pragmas = self._sqlite_pragmas

        if hasattr(self, "async_connection_pool") and self.async_connection_pool:
            self.async_connection_pool._pragmas = self._sqlite_pragmas

        # Persist to database config if requested
        if persist:
            self._save_sqlite_tuning()

        logger.info(f"Applied SQLite profile '{profile}' with {len(overrides or {})} overrides")

    def sqlite_checkpoint(self, mode: str = "PASSIVE") -> None:
        """Run SQLite WAL checkpoint operation."""
        valid_modes = ["PASSIVE", "FULL", "RESTART", "TRUNCATE"]
        if mode.upper() not in valid_modes:
            raise ValueError(f"Invalid checkpoint mode '{mode}'. Valid modes: {valid_modes}")

        with self.connection_pool.get_connection() as conn:
            conn.execute(f"PRAGMA wal_checkpoint({mode.upper()})")
            conn.commit()

        logger.debug(f"SQLite WAL checkpoint completed with mode '{mode}'")

    def sqlite_optimize(self) -> None:
        """Run SQLite PRAGMA optimize."""
        with self.connection_pool.get_connection() as conn:
            conn.execute("PRAGMA optimize")
            conn.commit()

        logger.debug("SQLite PRAGMA optimize completed")

    def sqlite_vacuum(self) -> None:
        """Run SQLite VACUUM operation."""
        with self.connection_pool.get_connection() as conn:
            conn.execute("VACUUM")
            conn.commit()

        logger.info("SQLite VACUUM completed")

    def sqlite_incremental_vacuum(self, pages: int = 2000) -> None:
        """Run incremental VACUUM operation."""
        with self.connection_pool.get_connection() as conn:
            conn.execute(f"PRAGMA incremental_vacuum({pages})")
            conn.commit()

        logger.debug(f"SQLite incremental vacuum completed for {pages} pages")

    def checkpoint_if_wal_large(self, wal_mb_threshold: int = 128) -> bool:
        """Check if WAL file is large and checkpoint if needed."""
        from pathlib import Path

        if hasattr(self, "db_path") and not self.is_memory_only:
            wal_path = Path(str(self.db_path) + "-wal")
            try:
                if wal_path.exists():
                    wal_size_mb = wal_path.stat().st_size / (1024 * 1024)
                    if wal_size_mb > wal_mb_threshold:
                        self.sqlite_checkpoint("TRUNCATE")
                        logger.info(f"Checkpointed large WAL file ({wal_size_mb:.1f} MB)")
                        return True
            except Exception as e:
                logger.debug(f"Failed to check WAL size: {e}")

        return False

    def _save_sqlite_tuning(self) -> None:
        """Save SQLite tuning configuration to database."""
        if hasattr(self, "connection_pool"):
            with self.connection_pool.get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", ("sqlite_profile", self._sqlite_profile)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                    ("sqlite_pragma_overrides", json.dumps(self._sqlite_pragma_overrides)),
                )
                conn.commit()

    def _load_sqlite_tuning(self, config: Dict[str, str]) -> None:
        """Load SQLite tuning configuration from database config."""
        profile = config.get("sqlite_profile", "balanced")
        overrides_json = config.get("sqlite_pragma_overrides", "{}")

        try:
            overrides = json.loads(overrides_json) if overrides_json else {}
        except (json.JSONDecodeError, TypeError):
            overrides = {}

        if profile in SQLITE_PRAGMA_PROFILES:
            pragmas = dict(SQLITE_PRAGMA_PROFILES[profile].pragmas)
            pragmas.update(overrides)

            self._sqlite_profile = cast(SqliteProfile, profile)
            self._sqlite_pragma_overrides = overrides
            self._sqlite_pragmas = pragmas

            logger.debug(f"Loaded SQLite tuning profile '{profile}' with {len(overrides)} overrides")
        else:
            logger.warning(f"Unknown saved SQLite profile '{profile}', using balanced")
            self._sqlite_profile = "balanced"
            self._sqlite_pragma_overrides = {}
            self._sqlite_pragmas = dict(SQLITE_PRAGMA_PROFILES["balanced"].pragmas)
