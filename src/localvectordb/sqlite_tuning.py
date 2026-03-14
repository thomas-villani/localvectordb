# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/sqlite_tuning.py

"""
SQLite performance tuning and optimization for LocalVectorDB.

This module provides comprehensive SQLite tuning capabilities including:
- Predefined performance profiles for different workloads
- Safe pragma application with validation
- System resource analysis for intelligent tuning
- Maintenance utilities for database optimization

Classes
-------
SQLitePragmaProfile
    Configuration profile containing pragma settings
SystemInfo
    System resource information for tuning decisions
WorkloadProfile
    User workload characteristics for auto-tuning
TuningRecommendation
    Auto-tuner recommendation with profile and overrides
AutoTuner
    Intelligent profile selection based on system and workload
"""

import logging
import os
import platform
import re
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import aiosqlite
import psutil

logger = logging.getLogger(__name__)


class WorkloadType(Enum):
    """Workload type enumeration for auto-tuning."""
    READ_HEAVY = "read_heavy"
    WRITE_HEAVY = "write_heavy"
    BALANCED = "balanced"
    BATCH_INGEST = "batch_ingest"
    REAL_TIME = "real_time"


class DurabilityLevel(Enum):
    """Data durability importance levels."""
    CRITICAL = "critical"  # Banking, medical records
    HIGH = "high"  # Production data
    NORMAL = "normal"  # Standard applications
    LOW = "low"  # Temporary data, caches


@dataclass
class SQLitePragmaProfile:
    """
    SQLite pragma configuration profile.

    Parameters
    ----------
    name : str
        Profile name identifier
    description : str
        Human-readable description of the profile
    pragmas : Dict[str, Any]
        Dictionary of pragma key-value pairs
    """
    name: str
    description: str = ""
    pragmas: Dict[str, Any] = field(default_factory=dict)


# Predefined tuning profiles based on common use cases
SQLITE_PRAGMA_PROFILES: Dict[str, SQLitePragmaProfile] = {
    "balanced": SQLitePragmaProfile(
        name="balanced",
        description="Balanced performance for mixed workloads (default)",
        pragmas={
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "foreign_keys": "ON",
            "busy_timeout": 5000,
            "temp_store": "MEMORY",
            "cache_size": -65536,  # ~64MB
            "mmap_size": 268435456,  # 256MB
            "wal_autocheckpoint": 1000,
            "cache_spill": "ON",
            "automatic_index": "ON",
        }
    ),
    "fast_ingest": SQLitePragmaProfile(
        name="fast_ingest",
        description="Optimized for high-throughput data ingestion",
        pragmas={
            "journal_mode": "WAL",
            "synchronous": "NORMAL",  # Consider OFF only if acceptable risk
            "foreign_keys": "ON",
            "busy_timeout": 10000,
            "temp_store": "MEMORY",
            "cache_size": -262144,  # ~256MB
            "mmap_size": 268435456,  # 256MB
            "wal_autocheckpoint": 4000,  # Larger WAL before checkpoint
            "cache_spill": "ON",
            "automatic_index": "ON",
        }
    ),
    "read_optimized": SQLitePragmaProfile(
        name="read_optimized",
        description="Optimized for low-latency queries and searches",
        pragmas={
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "foreign_keys": "ON",
            "busy_timeout": 3000,
            "temp_store": "MEMORY",
            "cache_size": -131072,  # ~128MB
            "mmap_size": 536870912,  # 512MB
            "wal_autocheckpoint": 1000,
            "cache_spill": "ON",
            "automatic_index": "ON",
        }
    ),
    "durable": SQLitePragmaProfile(
        name="durable",
        description="Maximum data durability (safest but slower)",
        pragmas={
            "journal_mode": "WAL",
            "synchronous": "FULL",
            "foreign_keys": "ON",
            "busy_timeout": 10000,
            "temp_store": "FILE",
            "cache_size": -65536,  # ~64MB
            "mmap_size": 134217728,  # 128MB
            "wal_autocheckpoint": 100,  # Frequent checkpoints
            "cache_spill": "ON",
        }
    ),
    "memory_saver": SQLitePragmaProfile(
        name="memory_saver",
        description="Minimal memory footprint for constrained environments",
        pragmas={
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "foreign_keys": "ON",
            "busy_timeout": 5000,
            "temp_store": "FILE",
            "cache_size": -8192,  # ~8MB
            "mmap_size": 0,  # Disable mmap
            "wal_autocheckpoint": 500,
            "cache_spill": "ON",
        }
    ),
}

SqliteProfile = Literal["balanced", "fast_ingest", "read_optimized", "durable", "memory_saver"]

def is_valid_sqlite_pragma_profile(profile: SqliteProfile) -> bool:
    return profile in SQLITE_PRAGMA_PROFILES

def get_sqlite_pragma_profile(
        profile: SqliteProfile, *, default: Optional[SqliteProfile] = None
) -> SQLitePragmaProfile:
    return SQLITE_PRAGMA_PROFILES.get(profile, SQLITE_PRAGMA_PROFILES.get(default) if default is not None else default)

# Safe pragma values that don't require quotes
SAFE_PRAGMA_VALUES = {
    "ON", "OFF", "WAL", "MEMORY", "FILE", "DELETE", "PERSIST",
    "TRUNCATE", "FULL", "NORMAL", "IMMEDIATE", "EXCLUSIVE",
    "RESTART", "PASSIVE", "INCREMENTAL", "NONE"
}


def validate_pragma_key(key: str) -> bool:
    """
    Validate pragma key for safety.

    Parameters
    ----------
    key : str
        Pragma key to validate

    Returns
    -------
    bool
        True if key is safe, False otherwise
    """
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key))


def format_pragma_value(value: Any) -> str:
    """
    Format pragma value for SQL execution.

    Parameters
    ----------
    value : Any
        Pragma value to format

    Returns
    -------
    str
        Formatted value safe for SQL
    """
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    elif isinstance(value, str):
        upper_val = value.upper()
        if upper_val in SAFE_PRAGMA_VALUES:
            return upper_val
        else:
            # Quote string values
            return f"'{value}'"
    else:
        return str(value)


def apply_pragmas(conn: sqlite3.Connection, pragmas: Dict[str, Any]) -> None:
    """
    Apply pragma settings to a SQLite connection.

    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite database connection
    pragmas : Dict[str, Any]
        Dictionary of pragma key-value pairs to apply
    """
    for key, value in pragmas.items():
        if not validate_pragma_key(key):
            logger.warning(f"Skipping invalid pragma key: {key}")
            continue

        formatted_value = format_pragma_value(value)
        sql = f"PRAGMA {key} = {formatted_value}"

        try:
            conn.execute(sql)
            logger.debug(f"Applied pragma: {sql}")
        except sqlite3.Error as e:
            # Best-effort: some pragmas may not be supported on all platforms
            logger.debug(f"Failed to apply pragma {key}: {e}")


async def apply_pragmas_async(conn: aiosqlite.Connection, pragmas: Dict[str, Any]) -> None:
    """
    Apply pragma settings to an async SQLite connection.

    Parameters
    ----------
    conn : aiosqlite.Connection
        Async SQLite database connection
    pragmas : Dict[str, Any]
        Dictionary of pragma key-value pairs to apply
    """
    for key, value in pragmas.items():
        if not validate_pragma_key(key):
            logger.warning(f"Skipping invalid pragma key: {key}")
            continue

        formatted_value = format_pragma_value(value)
        sql = f"PRAGMA {key} = {formatted_value}"

        try:
            await conn.execute(sql)
            logger.debug(f"Applied pragma: {sql}")
        except Exception as e:
            # Best-effort: some pragmas may not be supported on all platforms
            logger.debug(f"Failed to apply pragma {key}: {e}")


@dataclass
class SystemInfo:
    """
    System resource information for tuning decisions.

    Parameters
    ----------
    total_ram_mb : int
        Total system RAM in megabytes
    available_ram_mb : int
        Available system RAM in megabytes
    cpu_cores : int
        Number of CPU cores
    disk_type : str
        Disk type (SSD/HDD/Unknown)
    disk_free_gb : float
        Free disk space in gigabytes
    os_type : str
        Operating system type
    """
    total_ram_mb: int
    available_ram_mb: int
    cpu_cores: int
    disk_type: str  # SSD, HDD, or Unknown
    disk_free_gb: float
    os_type: str


@dataclass
class WorkloadProfile:
    """
    User workload characteristics for auto-tuning.

    Parameters
    ----------
    workload_type : WorkloadType
        Primary workload pattern
    document_size : str
        Typical document size (small/medium/large)
    concurrent_users : int
        Expected number of concurrent users
    durability_level : DurabilityLevel
        Data persistence importance
    memory_constraint : str
        Memory availability (generous/moderate/limited)
    """
    workload_type: WorkloadType
    document_size: str  # small, medium, large
    concurrent_users: int
    durability_level: DurabilityLevel
    memory_constraint: str  # generous, moderate, limited


@dataclass
class TuningRecommendation:
    """
    Auto-tuner recommendation result.

    Parameters
    ----------
    profile_name : str
        Recommended base profile name
    pragma_overrides : Dict[str, Any]
        Custom pragma overrides for the profile
    reasoning : List[str]
        Explanation of the recommendation
    estimated_memory_mb : int
        Estimated memory usage with these settings
    """
    profile_name: str
    pragma_overrides: Dict[str, Any]
    reasoning: List[str]
    estimated_memory_mb: int


class AutoTuner:
    """
    Intelligent SQLite tuning based on system resources and workload.

    This class analyzes system resources and user requirements to recommend
    optimal SQLite pragma settings for LocalVectorDB.
    """

    @staticmethod
    def analyze_system() -> SystemInfo:
        """
        Analyze system resources for tuning decisions.

        Returns
        -------
        SystemInfo
            System resource information
        """
        # Get memory information
        mem = psutil.virtual_memory()
        total_ram_mb = mem.total // (1024 * 1024)
        available_ram_mb = mem.available // (1024 * 1024)

        # Get CPU information
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1

        # Detect disk type (simplified heuristic)
        disk_type = AutoTuner._detect_disk_type()

        # Get disk space
        disk = shutil.disk_usage(Path.cwd())
        disk_free_gb = disk.free / (1024 ** 3)

        # Get OS type
        os_type = platform.system()

        return SystemInfo(
            total_ram_mb=total_ram_mb,
            available_ram_mb=available_ram_mb,
            cpu_cores=cpu_cores,
            disk_type=disk_type,
            disk_free_gb=disk_free_gb,
            os_type=os_type
        )

    @staticmethod
    def _detect_disk_type() -> str:
        """
        Detect if the current disk is SSD or HDD.

        Returns
        -------
        str
            "SSD", "HDD", or "Unknown"
        """
        try:
            # Simple heuristic: measure sequential write speed
            test_size = 10 * 1024 * 1024  # 10MB test file

            # Use temporary file to avoid CWD permissions and ensure cleanup
            with tempfile.NamedTemporaryFile(delete=True, dir=tempfile.gettempdir()) as temp_file:
                start = time.time()
                temp_file.write(b'0' * test_size)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                elapsed = time.time() - start

            # If write speed > 50MB/s, likely SSD
            speed_mbps = (test_size / (1024 * 1024)) / elapsed
            return "SSD" if speed_mbps > 50 else "HDD"

        except Exception as e:
            logger.debug(f"Could not detect disk type: {e}")
            return "Unknown"

    @staticmethod
    def recommend_profile(
            system: SystemInfo,
            workload: WorkloadProfile
    ) -> TuningRecommendation:
        """
        Recommend optimal tuning profile based on system and workload.

        Parameters
        ----------
        system : SystemInfo
            System resource information
        workload : WorkloadProfile
            User workload characteristics

        Returns
        -------
        TuningRecommendation
            Recommended profile and settings
        """
        reasoning = []
        pragma_overrides = {}

        # Base profile selection
        if workload.workload_type == WorkloadType.WRITE_HEAVY:
            profile_name = "fast_ingest"
            reasoning.append("Selected fast_ingest profile for write-heavy workload")
        elif workload.workload_type == WorkloadType.READ_HEAVY:
            profile_name = "read_optimized"
            reasoning.append("Selected read_optimized profile for read-heavy workload")
        elif workload.durability_level == DurabilityLevel.CRITICAL:
            profile_name = "durable"
            reasoning.append("Selected durable profile for critical data")
        elif workload.memory_constraint == "limited" or system.available_ram_mb < 2048:
            profile_name = "memory_saver"
            reasoning.append("Selected memory_saver profile due to memory constraints")
        else:
            profile_name = "balanced"
            reasoning.append("Selected balanced profile for mixed workload")

        # Adjust cache_size based on available RAM
        if system.available_ram_mb >= 16384:  # 16GB+
            cache_mb = 512
            reasoning.append(f"Set large cache ({cache_mb}MB) for abundant RAM")
        elif system.available_ram_mb >= 8192:  # 8GB+
            cache_mb = 256
            reasoning.append(f"Set medium cache ({cache_mb}MB) for moderate RAM")
        elif system.available_ram_mb >= 4096:  # 4GB+
            cache_mb = 128
            reasoning.append(f"Set standard cache ({cache_mb}MB)")
        else:
            cache_mb = 32
            reasoning.append(f"Set minimal cache ({cache_mb}MB) for limited RAM")

        pragma_overrides["cache_size"] = -(cache_mb * 1024)  # Convert to KB

        # Adjust mmap_size based on disk type and RAM
        if system.disk_type == "SSD" and system.available_ram_mb >= 8192:
            mmap_mb = min(1024, system.available_ram_mb // 8)
            pragma_overrides["mmap_size"] = mmap_mb * 1024 * 1024
            reasoning.append(f"Enabled memory mapping ({mmap_mb}MB) for SSD")
        elif system.disk_type == "HDD":
            pragma_overrides["mmap_size"] = 0
            reasoning.append("Disabled memory mapping for HDD")

        # Adjust WAL settings for workload
        if workload.workload_type == WorkloadType.BATCH_INGEST:
            pragma_overrides["wal_autocheckpoint"] = 10000
            pragma_overrides["synchronous"] = "OFF"
            reasoning.append("Relaxed WAL settings for batch ingestion")
        elif workload.workload_type == WorkloadType.REAL_TIME:
            pragma_overrides["wal_autocheckpoint"] = 100
            pragma_overrides["synchronous"] = "FULL"
            reasoning.append("Strict WAL settings for real-time processing")

        # Platform-specific adjustments
        if (system.os_type == "Darwin"
                and workload.durability_level in [DurabilityLevel.CRITICAL, DurabilityLevel.HIGH]):
            pragma_overrides["fullfsync"] = "ON"
            reasoning.append("Enabled fullfsync for macOS durability")

        # Thread/connection pool adjustments
        if workload.concurrent_users > 10:
            pragma_overrides["busy_timeout"] = 10000
            reasoning.append("Increased busy timeout for high concurrency")

        # Calculate estimated memory usage
        cache_mb = abs(pragma_overrides.get("cache_size", -65536)) // 1024
        mmap_mb = pragma_overrides.get("mmap_size", 268435456) // (1024 * 1024)
        estimated_memory_mb = cache_mb + mmap_mb + 100  # Add overhead

        return TuningRecommendation(
            profile_name=profile_name,
            pragma_overrides=pragma_overrides,
            reasoning=reasoning,
            estimated_memory_mb=estimated_memory_mb
        )

    @staticmethod
    def interview_user_cli() -> WorkloadProfile:
        """
        Interactive CLI interview to gather workload information.

        Returns
        -------
        WorkloadProfile
            User's workload characteristics
        """
        print("\n=== LocalVectorDB Auto-Tuning Interview ===\n")

        # Workload type
        print("1. What is your primary use case?")
        print("   a) Mostly searching and retrieval (read-heavy)")
        print("   b) Mostly adding new documents (write-heavy)")
        print("   c) Balanced mix of both")
        print("   d) Large batch data imports")
        print("   e) Real-time processing")

        choice = input("\nSelect (a-e): ").lower()
        workload_map = {
            'a': WorkloadType.READ_HEAVY,
            'b': WorkloadType.WRITE_HEAVY,
            'c': WorkloadType.BALANCED,
            'd': WorkloadType.BATCH_INGEST,
            'e': WorkloadType.REAL_TIME
        }
        workload_type = workload_map.get(choice, WorkloadType.BALANCED)

        # Document size
        print("\n2. What is your typical document size?")
        print("   a) Small (< 1KB) - tweets, logs")
        print("   b) Medium (1-10KB) - articles, emails")
        print("   c) Large (> 10KB) - books, papers")

        choice = input("\nSelect (a-c): ").lower()
        doc_size_map = {'a': 'small', 'b': 'medium', 'c': 'large'}
        document_size = doc_size_map.get(choice, 'medium')

        # Concurrent users
        print("\n3. How many concurrent users/processes?")
        print("   a) Single user (1)")
        print("   b) Small team (2-5)")
        print("   c) Medium team (6-20)")
        print("   d) Large deployment (20+)")

        choice = input("\nSelect (a-d): ").lower()
        users_map = {'a': 1, 'b': 5, 'c': 15, 'd': 50}
        concurrent_users = users_map.get(choice, 5)

        # Durability
        print("\n4. How important is data durability?")
        print("   a) Critical - Cannot lose any data")
        print("   b) High - Production data")
        print("   c) Normal - Standard application")
        print("   d) Low - Cache/temporary data")

        choice = input("\nSelect (a-d): ").lower()
        durability_map = {
            'a': DurabilityLevel.CRITICAL,
            'b': DurabilityLevel.HIGH,
            'c': DurabilityLevel.NORMAL,
            'd': DurabilityLevel.LOW
        }
        durability_level = durability_map.get(choice, DurabilityLevel.NORMAL)

        # Memory constraints
        print("\n5. Memory availability for the database?")
        print("   a) Generous - Use as much as needed")
        print("   b) Moderate - Balance with other apps")
        print("   c) Limited - Minimize memory usage")

        choice = input("\nSelect (a-c): ").lower()
        memory_map = {'a': 'generous', 'b': 'moderate', 'c': 'limited'}
        memory_constraint = memory_map.get(choice, 'moderate')

        return WorkloadProfile(
            workload_type=workload_type,
            document_size=document_size,
            concurrent_users=concurrent_users,
            durability_level=durability_level,
            memory_constraint=memory_constraint
        )


def get_profile_description(profile_name: str) -> str:
    """
    Get human-readable description of a profile.

    Parameters
    ----------
    profile_name : str
        Name of the profile

    Returns
    -------
    str
        Profile description
    """
    profile = SQLITE_PRAGMA_PROFILES.get(profile_name)
    return profile.description if profile else "Unknown profile"


def list_profiles() -> List[Tuple[str, str]]:
    """
    List all available tuning profiles.

    Returns
    -------
    List[Tuple[str, str]]
        List of (name, description) tuples
    """
    return [(name, profile.description) for name, profile in SQLITE_PRAGMA_PROFILES.items()]
