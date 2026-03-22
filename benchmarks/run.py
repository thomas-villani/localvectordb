"""CLI entry point for the benchmark suite.

Can be invoked as:
    python -m benchmarks.run [args]
    python benchmarks/run.py [args]
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `benchmarks` package is importable
# when run as `python benchmarks/run.py` from the project root.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmarks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LocalVectorDB Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python benchmarks/run.py                           # Run all
  python benchmarks/run.py --tier 1                  # ANN only
  python benchmarks/run.py --tier 2                  # Full-stack only
  python benchmarks/run.py --scales 10000 25000
  python benchmarks/run.py --index-types IndexFlatL2 IndexHNSWFlat
  python benchmarks/run.py --download-only
""",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only tier 1 (ANN) or tier 2 (full-stack). Default: both.",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=None,
        help="Override benchmark scales (e.g., 10000 25000).",
    )
    parser.add_argument(
        "--index-types",
        nargs="+",
        default=None,
        help="Override FAISS index types for tier 1.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the SIFT dataset, don't run benchmarks.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip markdown report generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.download_only:
        from benchmarks.datasets import download_sift

        download_sift()
        logger.info("Dataset download complete.")
        return

    results = {}
    run_tier1 = args.tier is None or args.tier == 1
    run_tier2 = args.tier is None or args.tier == 2

    if run_tier1:
        logger.info("=" * 60)
        logger.info("TIER 1: ANN Recall Benchmarks")
        logger.info("=" * 60)
        from benchmarks.tier1_ann import run_tier1 as _run_tier1

        results["tier1"] = _run_tier1(
            scales=args.scales,
            index_types=args.index_types,
        )

    if run_tier2:
        logger.info("=" * 60)
        logger.info("TIER 2: Full-Stack Benchmarks")
        logger.info("=" * 60)
        from benchmarks.tier2_fullstack import run_tier2 as _run_tier2

        results["tier2"] = _run_tier2(scales=args.scales)

    # Save results
    from benchmarks.reporting import generate_detailed_report, generate_summary_table, save_results_json

    json_path = save_results_json(results)
    logger.info("Results saved to %s", json_path)

    if not args.no_report:
        report = generate_detailed_report(results)
        report_path = json_path.with_suffix(".md")
        with open(report_path, "w") as f:
            f.write(report)
        logger.info("Report saved to %s", report_path)

        # Print summary to stdout
        tier1 = results.get("tier1", [])
        if tier1:
            print("\n" + generate_summary_table(tier1))


if __name__ == "__main__":
    main()
