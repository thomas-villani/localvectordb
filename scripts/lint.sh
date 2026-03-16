#!/usr/bin/env bash
set -euo pipefail

VENV=".venv/Scripts"

echo "=== black --check ==="
"$VENV/black.exe" --check src/

echo "=== ruff check ==="
"$VENV/ruff.exe" check .

echo "=== mypy ==="
"$VENV/mypy.exe" src/

echo "=== bandit ==="
"$VENV/bandit.exe" -c pyproject.toml -r src/

echo "All checks passed."
