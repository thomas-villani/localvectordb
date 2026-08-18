"""Extraction gate: fingerprint All2MdExtractor output over committed fixtures.

Neither retrieval gate touches the extraction path — both read their corpora
as text — so a dependency bump that changes what all2md produces moves section
boundaries and chunk content invisibly. Measured 2026-08-13: all2md 1.7.1 vs
1.12.0 changed 30/30 cached arXiv PDFs (0 byte-identical), with headings
monotone −7.2%, and every gated number stayed +0.0000 because the gates never
saw a PDF. This gate is the third leg: it runs the extractor src/ ingestion
actually uses over a small committed corpus and compares *fingerprints* of the
extracted markdown against a committed baseline.

Corpus: ``scripts/e2e/fixtures/`` (well-behaved documents, one per format) and
``benchmarks/extraction_fixtures/`` (adversarial: hyphen-wrapped headings in a
real PDF, numbered contract clauses in a DOCX, nested HTML structure).

A mismatch is not automatically a bug — extraction *fixes* change output too.
The gate's job is to make the change loud so it is reviewed and the corpora
rebuilt: treat any all2md bump as corpus-invalidating for file-derived
databases, and never compare retrieval numbers across the boundary.

Usage:
    # Gate (non-zero exit on any fingerprint drift)
    ./.venv/Scripts/python.exe benchmarks/eval_extraction.py --check

    # Accept the current output as the new baseline (review the diff first!)
    ./.venv/Scripts/python.exe benchmarks/eval_extraction.py --update
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = Path(__file__).resolve().parent / "extraction_baseline.json"
FIXTURE_DIRS = (
    REPO_ROOT / "scripts" / "e2e" / "fixtures",
    REPO_ROOT / "benchmarks" / "extraction_fixtures",
)
_HEADING_RE = re.compile(r"^#{1,6} ", re.MULTILINE)


def _fix_sys_path() -> None:
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def fixture_files() -> Dict[str, Path]:
    """Every committed fixture, keyed by a stable repo-relative name."""
    files: Dict[str, Path] = {}
    for directory in FIXTURE_DIRS:
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            # Generator scripts are not fixtures; task_scheduler.py IS one (a
            # source-format document the e2e suite ingests).
            if path.suffix == ".py" and path.name != "task_scheduler.py":
                continue
            files[path.relative_to(REPO_ROOT).as_posix()] = path
    return files


def fingerprint(path: Path) -> Dict[str, object]:
    """Extract one fixture and reduce the markdown to comparable metrics.

    ``sha256`` catches any change at all; the remaining metrics say what kind
    of change it was (length drift vs heading structure vs section detection),
    because "30 files changed" is unactionable without that.
    """
    from localvectordb.extractors.all2md_extractor import All2MdExtractor
    from localvectordb.section_detection import SectionDetector

    result = All2MdExtractor().extract_text(path.read_bytes(), path.name)
    if not result.success:
        return {"success": False, "error": result.error}
    text = result.text
    headings = _HEADING_RE.findall(text)
    heading_lines = [line for line in text.splitlines() if _HEADING_RE.match(line)]
    sections = SectionDetector().detect_sections(text)
    return {
        "success": True,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "headings": len(headings),
        "heading_digest": hashlib.sha256("\n".join(heading_lines).encode("utf-8")).hexdigest()[:16],
        "sections": len(sections),
    }


def build_report() -> Dict[str, object]:
    from importlib.metadata import version

    return {
        "schema": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "all2md_version": version("all2md"),
        "files": {name: fingerprint(path) for name, path in fixture_files().items()},
    }


def compare(current: Dict[str, object], baseline_path: Path) -> int:
    if not baseline_path.exists():
        print(f"No baseline at {baseline_path}. Run with --update first.", file=sys.stderr)
        return 1
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    print(f"Baseline: {baseline['git_commit']} ({baseline['generated']}), all2md {baseline['all2md_version']}")
    print(f"Current : {current['git_commit']}, all2md {current['all2md_version']}")

    old_files: Dict[str, Dict] = baseline["files"]
    new_files: Dict[str, Dict] = current["files"]  # type: ignore[assignment]
    drifted = []
    for name in sorted(set(old_files) | set(new_files)):
        old, new = old_files.get(name), new_files.get(name)
        if old is None:
            print(f"  NEW      {name}")
            continue
        if new is None:
            drifted.append(name)
            print(f"  MISSING  {name}")
            continue
        if old == new:
            print(f"  ok       {name}")
            continue
        drifted.append(name)
        deltas = [
            f"{key}: {old.get(key)} -> {new.get(key)}"
            for key in ("success", "chars", "lines", "headings", "heading_digest", "sections", "sha256", "error")
            if old.get(key) != new.get(key)
        ]
        print(f"  CHANGED  {name}")
        for delta in deltas:
            print(f"             {delta}")

    if drifted:
        print(
            f"\n{len(drifted)} fixture(s) drifted. If this is an intended extraction change "
            "(e.g. an all2md upgrade), review the diffs, treat file-derived corpora as "
            "invalidated, and re-run with --update.",
            file=sys.stderr,
        )
        return 1
    print("\nExtraction output matches the baseline.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    _fix_sys_path()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Compare against the committed baseline.")
    group.add_argument("--update", action="store_true", help="Write the baseline (tracked in git).")
    args = parser.parse_args(argv)

    report = build_report()
    failures = {name: fp for name, fp in report["files"].items() if not fp.get("success")}  # type: ignore[union-attr]
    if failures:
        for name, fp in failures.items():
            print(f"EXTRACTION FAILED for {name}: {fp.get('error')}", file=sys.stderr)
        return 1

    if args.update:
        BASELINE_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline written to {BASELINE_PATH} ({len(report['files'])} files)")  # type: ignore[arg-type]
        return 0
    return compare(report, BASELINE_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
