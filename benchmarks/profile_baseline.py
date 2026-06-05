"""Function-level profiling baseline for LocalVectorDB.

Complements the wall-clock benchmarks in ``tier2_fullstack`` with cProfile
attribution: *where* time is spent inside our own Python (ingest, search,
chunking, scoring) versus native FAISS / SQLite / numpy.

Uses the ``mock`` embedding provider so the profile isolates library overhead
from network / model latency — i.e. it measures the code we can actually
optimize, not the provider.

Run::

    uv run python -m benchmarks.profile_baseline                 # defaults
    uv run python -m benchmarks.profile_baseline --scale 10000 --queries 200

Outputs:
  * Per-phase top-function tables (by tottime and cumtime) to stdout.
  * Raw ``.prof`` files under benchmarks/results/profiles/ (open with snakeviz).
  * A markdown summary alongside them.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pstats
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from .config import INSERT_BATCH_SIZE, QUERY_K
from .tier2_fullstack import _create_db, _generate_documents, _generate_metadata
from .utils import get_system_info

logger = logging.getLogger("benchmarks.profile")

PROFILE_DIR = Path(__file__).parent / "results" / "profiles"

# Restrict the "our code" tables to frames inside the package so native
# FAISS/SQLite calls don't drown out the Python we can change.
_OURS = "localvectordb"


def _stats_text(profiler: cProfile.Profile, *, sort: str, limit: int, restrict: str | None = None) -> str:
    """Render a pstats table to a string.

    When ``restrict`` is given we must NOT strip_dirs first, because the
    restriction regex is matched against the formatted line which only contains
    the package path when directories are kept.
    """
    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    if restrict:
        stats.sort_stats(sort)
        stats.print_stats(restrict, limit)
    else:
        stats.strip_dirs().sort_stats(sort)
        stats.print_stats(limit)
    return buf.getvalue()


def _total_time(profiler: cProfile.Profile) -> float:
    stats = pstats.Stats(profiler)
    return float(stats.total_tt)  # type: ignore[attr-defined]


def _profile(label: str, fn: Callable[[], Any]) -> Dict[str, Any]:
    """Run ``fn`` under cProfile, persist the .prof, return summary tables."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    fn()
    profiler.disable()
    wall = time.perf_counter() - wall_start

    prof_path = PROFILE_DIR / f"{label}.prof"
    profiler.dump_stats(str(prof_path))

    return {
        "label": label,
        "wall_seconds": wall,
        "profiled_seconds": _total_time(profiler),
        "prof_path": str(prof_path),
        "by_tottime_all": _stats_text(profiler, sort="tottime", limit=20),
        "by_cumtime_ours": _stats_text(profiler, sort="cumtime", limit=25, restrict=_OURS),
        "by_tottime_ours": _stats_text(profiler, sort="tottime", limit=25, restrict=_OURS),
    }


def _run_queries(db: Any, query_texts: List[str], search_type: str) -> None:
    for qt in query_texts:
        db.query(qt, k=QUERY_K, search_type=search_type)


def run(scale: int, n_queries: int) -> Dict[str, Any]:
    logger.info("Building corpus: %d docs", scale)
    docs = _generate_documents(scale)
    meta = _generate_metadata(scale)

    rng = random.Random(99)  # nosec B311
    query_texts = [
        " ".join(random.Random(i).choices(docs[i % scale].split(), k=8)) for i in range(n_queries)  # nosec B311
    ]
    # Mix in some short pseudo-queries too
    query_texts += [_short_query(rng) for _ in range(n_queries // 4)]

    phases: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Ingestion (FTS on, since that's the heavier, more realistic path) ---
        db = _create_db(tmpdir, name="prof_ingest", enable_fts=True)
        phases.append(_profile("ingest", lambda: db.upsert(docs, metadata=meta, batch_size=INSERT_BATCH_SIZE)))

        # Warm up (JIT-y caches, FAISS internal state) before timing queries.
        _run_queries(db, query_texts[:10], "vector")

        # --- Query phases (reuse the populated db) ---
        for st in ("vector", "keyword", "hybrid"):
            phases.append(_profile(f"query_{st}", lambda st=st: _run_queries(db, query_texts, st)))

        db.close()

    return {
        "system": get_system_info(),
        "scale": scale,
        "n_queries": len(query_texts),
        "phases": phases,
    }


def _short_query(rng: random.Random) -> str:
    # consonant/vowel pseudo-words like the corpus generator, short
    pool = "the data system model query vector index search result document chunk score"
    return " ".join(rng.choices(pool.split(), k=4))


def _print_report(result: Dict[str, Any]) -> None:
    sysinfo = result["system"]
    print("\n" + "=" * 78)
    print("LocalVectorDB — Profiling Baseline (mock embeddings)")
    print("=" * 78)
    print(f"Platform : {sysinfo.get('platform')}")
    print(f"Python   : {sysinfo.get('python_version', '').splitlines()[0]}")
    print(f"numpy    : {sysinfo.get('numpy_version')}   faiss: {sysinfo.get('faiss_version')}")
    print(f"Corpus   : {result['scale']} docs   Queries/phase: {result['n_queries']}")
    print("=" * 78)

    for ph in result["phases"]:
        print("\n" + "#" * 78)
        print(f"# PHASE: {ph['label']}")
        print(f"#   wall={ph['wall_seconds']:.3f}s   profiled={ph['profiled_seconds']:.3f}s")
        print(f"#   raw profile: {ph['prof_path']}")
        print("#" * 78)
        print("\n--- Top by tottime (ALL frames incl. native) ---")
        print(ph["by_tottime_all"])
        print("--- Top by cumtime (localvectordb frames only) ---")
        print(ph["by_cumtime_ours"])
        print("--- Top by tottime (localvectordb frames only) ---")
        print(ph["by_tottime_ours"])


def _write_markdown(result: Dict[str, Any]) -> Path:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    out = PROFILE_DIR / "baseline_summary.md"
    sysinfo = result["system"]
    lines = [
        "# LocalVectorDB Profiling Baseline",
        "",
        f"- Platform: `{sysinfo.get('platform')}`",
        f"- Python: `{sysinfo.get('python_version', '').splitlines()[0]}`",
        f"- numpy `{sysinfo.get('numpy_version')}`, faiss `{sysinfo.get('faiss_version')}`",
        f"- Corpus: **{result['scale']} docs**, **{result['n_queries']} queries/phase**",
        "- Embeddings: mock (isolates library overhead from provider latency)",
        "",
        "## Phase wall-clock",
        "",
        "| Phase | wall (s) | profiled (s) |",
        "| --- | --- | --- |",
    ]
    for ph in result["phases"]:
        lines.append(f"| {ph['label']} | {ph['wall_seconds']:.3f} | {ph['profiled_seconds']:.3f} |")
    for ph in result["phases"]:
        lines += [
            "",
            f"## {ph['label']} — top by tottime (all frames)",
            "",
            "```",
            ph["by_tottime_all"].strip(),
            "```",
            "",
            f"## {ph['label']} — top by cumtime (localvectordb only)",
            "",
            "```",
            ph["by_cumtime_ours"].strip(),
            "```",
        ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalVectorDB function-level profiling baseline")
    parser.add_argument("--scale", type=int, default=5_000, help="Number of documents (default 5000)")
    parser.add_argument("--queries", type=int, default=100, help="Queries per search type (default 100)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    result = run(args.scale, args.queries)
    _print_report(result)
    md = _write_markdown(result)
    logger.info("Markdown summary written to %s", md)


if __name__ == "__main__":
    main()
