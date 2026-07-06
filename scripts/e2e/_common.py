"""Shared helpers for the end-to-end test scripts.

These scripts exercise localvectordb against *real* embedding backends and
real documents. They are release-qualification tools, not unit tests: each
script prints a PASS/FAIL line per check and exits non-zero if any check
failed.

Provider selection (auto-detected, override with --provider):
  1. ollama                — if an Ollama server responds on OLLAMA_URL
                             (default http://localhost:11434) and has an
                             embedding model available.
  2. sentence_transformers — if the sentence-transformers package is
                             installed (fully local, no server needed).
"""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

E2E_DIR = Path(__file__).parent
FIXTURES_DIR = E2E_DIR / "fixtures"

OLLAMA_MODEL = "nomic-embed-text"
SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"


class Checker:
    """Collects named pass/fail checks and reports a summary."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[tuple[str, str]] = []
        self._start = time.monotonic()
        print(f"=== {title} ===")

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed.append(name)
            print(f"  [PASS] {name}")
        else:
            self.failed.append((name, detail))
            print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))
        return condition

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append((name, reason))
        print(f"  [SKIP] {name} -- {reason}")

    @contextmanager
    def step(self, name: str):
        """Run a block; an uncaught exception fails the named check."""
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - report, don't crash the suite
            self.failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  [FAIL] {name} -- {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            # Steps that complete without raising register their own granular
            # checks; the step itself only reports on failure.
            pass

    def section(self, title: str) -> None:
        print(f"--- {title} ---")

    def summary(self) -> int:
        elapsed = time.monotonic() - self._start
        print(
            f"=== {self.title}: {len(self.passed)} passed, {len(self.failed)} failed, "
            f"{len(self.skipped)} skipped in {elapsed:.1f}s ==="
        )
        for name, detail in self.failed:
            print(f"    FAILED: {name} -- {detail}")
        for name, reason in self.skipped:
            print(f"    SKIPPED: {name} -- {reason}")
        return 1 if self.failed else 0


def ollama_available(base_url: str = "http://localhost:11434") -> bool:
    import httpx

    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return any(name.split(":")[0] == OLLAMA_MODEL for name in models)
    except Exception:
        return False


def sentence_transformers_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


def detect_provider(preferred: str | None = None) -> tuple[str, str]:
    """Return (provider, model) for the best available real embedding backend."""
    if preferred == "ollama" or (preferred is None and ollama_available()):
        if not ollama_available():
            sys.exit(
                "Ollama requested but not available: start it with 'ollama serve' and " f"'ollama pull {OLLAMA_MODEL}'"
            )
        return "ollama", OLLAMA_MODEL
    if preferred in (None, "sentence_transformers") and sentence_transformers_available():
        return "sentence_transformers", SENTENCE_TRANSFORMERS_MODEL
    sys.exit(
        "No real embedding backend available. Either start Ollama (ollama serve; "
        f"ollama pull {OLLAMA_MODEL}) or install sentence-transformers "
        "(uv sync --dev / uv add 'localvectordb[sentence-transformers]')."
    )


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--provider",
        choices=["ollama", "sentence_transformers"],
        default=None,
        help="Embedding backend (default: auto-detect, preferring ollama)",
    )
    return parser


def ensure_fixtures() -> Path:
    """Generate the fixture documents if they are not present yet."""
    if not (FIXTURES_DIR / "machine_learning.pdf").exists():
        sys.path.insert(0, str(E2E_DIR))
        import make_fixtures

        make_fixtures.main()
    return FIXTURES_DIR


@contextmanager
def temp_workdir(prefix: str):
    """A temporary directory that survives cleanup failures (Windows locks)."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
