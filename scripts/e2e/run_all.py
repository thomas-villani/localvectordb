"""Run the full end-to-end suite: fixtures + local + files + server + CLI.

Usage:
    ./.venv/Scripts/python.exe scripts/e2e/run_all.py [--provider ollama|sentence_transformers]

Exits non-zero if any script fails. Each script is independent and can also
be run on its own.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import make_parser

E2E_DIR = Path(__file__).parent
SCRIPTS = ["e2e_local.py", "e2e_files.py", "e2e_server.py", "e2e_cli.py"]


def main() -> int:
    args = make_parser(__doc__.splitlines()[0]).parse_args()

    print("=== generating fixtures ===")
    fixtures = subprocess.run([sys.executable, str(E2E_DIR / "make_fixtures.py")])
    if fixtures.returncode != 0:
        print("FATAL: fixture generation failed")
        return 1

    results: dict[str, tuple[int, float]] = {}
    for script in SCRIPTS:
        start = time.monotonic()
        cmd = [sys.executable, str(E2E_DIR / script)]
        if args.provider:
            cmd += ["--provider", args.provider]
        proc = subprocess.run(cmd)
        results[script] = (proc.returncode, time.monotonic() - start)

    print()
    print("=" * 60)
    print("E2E SUITE SUMMARY")
    print("=" * 60)
    failed = False
    for script, (code, elapsed) in results.items():
        status = "PASS" if code == 0 else f"FAIL (exit {code})"
        failed = failed or code != 0
        print(f"  {script:<18} {status:<16} {elapsed:.1f}s")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
