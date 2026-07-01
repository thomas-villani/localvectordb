"""Timing helpers, memory measurement, and system info collection."""

import gc
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List

import faiss
import numpy as np


@dataclass
class TimingResult:
    """Stores timing measurements for a benchmark operation."""

    name: str
    elapsed_seconds: float
    iterations: int = 1

    @property
    def per_iteration(self) -> float:
        return self.elapsed_seconds / self.iterations if self.iterations > 0 else 0.0

    @property
    def ops_per_second(self) -> float:
        return self.iterations / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


@dataclass
class LatencyStats:
    """Percentile latency statistics from a series of measurements."""

    timings_ms: List[float] = field(default_factory=list)

    @property
    def p50(self) -> float:
        return float(np.percentile(self.timings_ms, 50)) if self.timings_ms else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.timings_ms, 95)) if self.timings_ms else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self.timings_ms, 99)) if self.timings_ms else 0.0

    @property
    def mean(self) -> float:
        return float(np.mean(self.timings_ms)) if self.timings_ms else 0.0

    def to_dict(self) -> dict:
        return {
            "p50_ms": round(self.p50, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
            "mean_ms": round(self.mean, 3),
            "count": len(self.timings_ms),
        }


@contextmanager
def timer(name: str = "") -> Generator[TimingResult, None, None]:
    """Context manager that measures wall-clock time."""
    result = TimingResult(name=name, elapsed_seconds=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start


def measure_memory_mb() -> float:
    """Measure current process RSS in MB using psutil."""
    gc.collect()
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def get_system_info() -> dict:
    """Collect system information for benchmark reports."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
    }
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024**3), 1)
        info["cpu_count"] = psutil.cpu_count(logical=True)
    except ImportError:
        # psutil is optional; omit memory/CPU info when it is unavailable.
        pass
    return info
