# Contributing to LocalVectorDB

Thank you for your interest in contributing. This guide covers the practical steps
for setting up a development environment, running tests, and submitting changes.

## Development Setup

```bash
git clone https://github.com/thomas-villani/localvectordb.git
cd localvectordb

# Recommended: uv creates the virtualenv and installs the dev dependency group
uv sync --dev

# Working on the MCP server feature as well? Add its extra:
#   uv sync --dev --extra mcp
```

`uv sync --dev` installs all test, lint, and documentation tooling along with the
server, file-extraction, and visualization extras. The MCP feature is opt-in
(its `fastmcp`/`pywin32` dependency can trip Windows antivirus on install).

<details>
<summary>Without uv (plain venv + pip)</summary>

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (MSYS/Git Bash); use .venv/bin/activate on Linux/macOS
pip install -e ".[server,file-extraction,visualization]"
pip install pytest pytest-asyncio pytest-xdist pytest-cov ruff mypy black "bandit[toml]" pre-commit build twine
```

</details>

Requires Python 3.12 or later.

## Running Tests

```bash
# Run the full test suite
pytest

# Fast tests only (skip slow and network-dependent tests)
pytest -m "not slow and not network"

# Run tests with coverage
pytest --cov=localvectordb --cov-report=term-missing

# Run a specific test file
pytest tests/test_core.py

# Run tests in parallel
pytest -n auto
```

Test markers:
- `unit` -- isolated unit tests
- `integration` -- tests using temporary databases with mock embeddings
- `slow` / `performance` -- benchmarks and long-running tests
- `network` -- tests that would reach external services (currently mocked)

Tests use `MockEmbeddings` for deterministic results without requiring an
embedding server. Temporary directories are cleaned up automatically.

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
The maximum line length is **120 characters**.

```bash
# Check for lint issues
ruff check .

# Auto-fix what can be fixed
ruff check . --fix
```

Enabled rule sets: `E`, `F`, `W`, `B`, `I` (pyflakes, pycodestyle, flake8-bugbear, isort).

Type checking with mypy is also available:

```bash
mypy src/
```

## Pull Request Process

1. Fork the repository and create a branch from `master`.
2. Make your changes. Add or update tests as appropriate.
3. Ensure `pytest` passes and `ruff check .` reports no issues.
4. Write a clear commit message describing what changed and why.
5. Open a pull request against `master`. In the PR description, explain the
   motivation and summarize the changes.

Keep pull requests focused. If you are fixing a bug and adding a feature,
submit them as separate PRs.

## Reporting Issues

Use GitHub Issues: <https://github.com/thomas-villani/localvectordb/issues>

When filing a bug report, include:
- Python version and OS
- Steps to reproduce the issue
- Expected vs. actual behavior
- Relevant logs or tracebacks

For feature requests, describe the use case and proposed behavior.

## Project Structure at a Glance

```
src/localvectordb/       Core library (database, embeddings, chunking)
src/localvectordb_server/ HTTP server, CLI, file extractors
tests/                    Test suite
docs/                     Sphinx documentation sources
```

See `CLAUDE.md` for a detailed architecture overview and common development tasks
such as adding a new embedding provider or file extractor.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License that covers this project.
