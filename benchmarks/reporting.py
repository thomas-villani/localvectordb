"""Markdown report generation from benchmark results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .config import RESULTS_DIR
from .utils import get_system_info


def save_results_json(results: Dict[str, Any]) -> Path:
    """Save benchmark results as timestamped JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = RESULTS_DIR / f"results_{ts}.json"
    payload = {
        "timestamp": ts,
        "system_info": get_system_info(),
        **results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Generate a markdown table."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def generate_summary_table(tier1_results: List[Dict[str, Any]]) -> str:
    """Generate a README-friendly summary table from Tier 1 results."""
    headers = ["Scale", "Index", "Path", "Recall@10", "QPS", "Build (s)", "Memory (MB)"]
    rows = []
    for r in sorted(tier1_results, key=lambda x: (x["scale"], x["index_type"], x["path"])):
        rows.append(
            [
                f'{r["scale"]:,}',
                r["index_type"],
                r["path"],
                r.get("recall@10", "N/A"),
                f'{r.get("qps", 0):,.0f}',
                r.get("build_time_s", "N/A"),
                r.get("memory_mb", "N/A"),
            ]
        )
    return _md_table(headers, rows)


def generate_detailed_report(results: Dict[str, Any]) -> str:
    """Generate the full detailed benchmark report in Markdown."""
    lines = []
    lines.append("# LocalVectorDB Benchmark Results")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # System info
    info = get_system_info()
    lines.append("## System Information")
    lines.append("")
    for k, v in info.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Tier 1
    tier1 = results.get("tier1", [])
    if tier1:
        lines.append("## Tier 1: ANN Recall Benchmarks (SIFT-128)")
        lines.append("")

        # Full table with all k values
        headers = ["Scale", "Index", "Path", "Recall@1", "Recall@10", "Recall@100", "QPS", "Build (s)"]
        rows = []
        for r in sorted(tier1, key=lambda x: (x["scale"], x["index_type"], x["path"])):
            rows.append(
                [
                    f'{r["scale"]:,}',
                    r["index_type"],
                    r["path"],
                    r.get("recall@1", "N/A"),
                    r.get("recall@10", "N/A"),
                    r.get("recall@100", "N/A"),
                    f'{r.get("qps", 0):,.0f}',
                    r.get("build_time_s", "N/A"),
                ]
            )
        lines.append(_md_table(headers, rows))
        lines.append("")

        # Overhead comparison
        lines.append("### Raw FAISS vs. LocalVectorDB Overhead")
        lines.append("")
        raw_by_key = {}
        lvdb_by_key = {}
        for r in tier1:
            key = (r["scale"], r["index_type"])
            if r["path"] == "raw_faiss":
                raw_by_key[key] = r
            else:
                lvdb_by_key[key] = r

        headers = ["Scale", "Index", "Raw QPS", "LVDB QPS", "Overhead %"]
        rows = []
        for key in sorted(raw_by_key.keys()):
            raw = raw_by_key[key]
            lvdb = lvdb_by_key.get(key)
            if lvdb:
                raw_qps = raw.get("qps", 0)
                lvdb_qps = lvdb.get("qps", 0)
                if raw_qps > 0:
                    overhead = ((raw_qps - lvdb_qps) / raw_qps) * 100
                else:
                    overhead = 0
                rows.append(
                    [
                        f"{key[0]:,}",
                        key[1],
                        f"{raw_qps:,.0f}",
                        f"{lvdb_qps:,.0f}",
                        f"{overhead:.1f}%",
                    ]
                )
        if rows:
            lines.append(_md_table(headers, rows))
        lines.append("")

    # Tier 2
    tier2 = results.get("tier2", {})
    if tier2:
        lines.append("## Tier 2: Full-Stack Benchmarks")
        lines.append("")

        # Insert throughput
        insert = tier2.get("insert_throughput", [])
        if insert:
            lines.append("### Insert Throughput")
            lines.append("")
            headers = ["Scale", "Time (s)", "Docs/sec", "Memory Delta (MB)"]
            rows = [
                [
                    f'{r["scale"]:,}',
                    r["elapsed_s"],
                    f'{r["docs_per_sec"]:,.0f}',
                    r.get("memory_delta_mb", "N/A"),
                ]
                for r in insert
            ]
            lines.append(_md_table(headers, rows))
            lines.append("")

        # Query latency
        for bench_name, title in [
            ("query_latency", "Query Latency (Vector)"),
            ("hybrid_search", "Hybrid Search Latency"),
            ("filtered_search", "Filtered Search Latency"),
        ]:
            data = tier2.get(bench_name, [])
            if data:
                lines.append(f"### {title}")
                lines.append("")
                headers = ["Scale", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Mean (ms)"]
                rows = [
                    [
                        f'{r["scale"]:,}',
                        r["p50_ms"],
                        r["p95_ms"],
                        r["p99_ms"],
                        r["mean_ms"],
                    ]
                    for r in data
                ]
                lines.append(_md_table(headers, rows))
                lines.append("")

        # Multi-database
        multi = tier2.get("multi_database")
        if multi:
            lines.append("### Multi-Database Scenario")
            lines.append("")
            lines.append(f"- **Databases**: {multi['num_dbs']}")
            lines.append(f"- **Documents per DB**: {multi['docs_per_db']:,}")
            lines.append(f"- **Total documents**: {multi['total_docs']:,}")
            lines.append(f"- **Total insert time**: {multi['total_insert_time_s']}s")
            lines.append(f"- **Memory total**: {multi['memory_delta_mb']}MB ({multi['memory_per_db_mb']}MB/db)")
            ql = multi.get("query_latency", {})
            if ql:
                lines.append(f"- **Query p50**: {ql.get('p50_ms', 'N/A')}ms")
                lines.append(f"- **Query p95**: {ql.get('p95_ms', 'N/A')}ms")
            lines.append("")

        # Memory footprint
        mem = tier2.get("memory_footprint", [])
        if mem:
            lines.append("### Memory Footprint")
            lines.append("")
            headers = ["Scale", "RSS (MB)", "Delta (MB)"]
            rows = [[f'{r["scale"]:,}', r["rss_mb"], r["delta_mb"]] for r in mem]
            lines.append(_md_table(headers, rows))
            lines.append("")

    return "\n".join(lines)
