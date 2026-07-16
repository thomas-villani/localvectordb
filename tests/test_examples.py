"""Keep examples/ honest.

An example is a promise that this code works. Nothing else in the suite runs
them, so without this module they rot exactly the way the README snippets did --
silently, until a reader hits the error we shipped.

The checks are split by what they need:

* The **fast** checks need no embedding backend and run on every PR. They verify
  that the examples call the real API (real methods, real keyword arguments,
  real ``search_level`` values, real ``QueryResult`` attributes) and that the
  bundled judgments still match the bundled corpus. That is the drift class that
  actually bites: a renamed kwarg or a retitled heading breaks the example while
  it still compiles perfectly.

* The **end-to-end** check actually runs the example and is marked
  ``slow``/``network``, because it needs a real embedding backend. CI's fast lane
  (``-m "not slow and not network"``) skips it. Run it before tagging a release,
  alongside scripts/e2e/.

Why not run the examples with MockEmbeddings so they could be fast and
hermetic: the examples refuse mock on purpose. MockEmbeddings seeds numpy's RNG
from a hash of the text, so it cannot tell whether the right section ranks
first, and an example whose entire subject is ranking quality cannot be
meaningfully exercised by it.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import typing
from pathlib import Path

import pytest

from localvectordb.core import QueryResult
from localvectordb.database import LocalVectorDB

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SAMPLE_CORPUS = EXAMPLES / "sample_corpus"
JUDGMENTS = SAMPLE_CORPUS / "judgments.json"
SECTION_VS_CHUNK = EXAMPLES / "section_vs_chunk_retrieval.py"


def _load_example(path: Path):
    """Import an example by path without requiring examples/ to be a package."""
    spec = importlib.util.spec_from_file_location(f"_example_{path.stem}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _query_params() -> dict:
    return inspect.signature(LocalVectorDB.query).parameters


@pytest.mark.docs
def test_every_example_parses():
    scripts = sorted(EXAMPLES.glob("*.py"))
    assert scripts, "examples/ has no scripts -- did the directory move?"
    for s in scripts:
        ast.parse(s.read_text(encoding="utf-8"), filename=str(s))


@pytest.mark.docs
def test_arms_use_real_query_kwargs():
    """Every arm's kwargs must be real parameters of LocalVectorDB.query.

    The arms are dict literals, so an AST scan of `db.query(**kwargs)` cannot
    see them; check the resolved values instead.
    """
    mod = _load_example(SECTION_VS_CHUNK)
    params = _query_params()
    for label, kwargs in mod.ARMS:
        for kw in kwargs:
            assert kw in params, f"examples arm {label!r}: '{kw}' is not a parameter of LocalVectorDB.query"


@pytest.mark.docs
def test_arms_use_valid_search_levels():
    """A search_level the library no longer accepts must fail here, not at runtime."""
    mod = _load_example(SECTION_VS_CHUNK)
    hints = typing.get_type_hints(LocalVectorDB.query)
    allowed = set(typing.get_args(hints["search_level"]))
    for label, kwargs in mod.ARMS:
        level = kwargs["search_level"]
        assert level in allowed, f"examples arm {label!r}: search_level={level!r} not in {sorted(allowed)}"


@pytest.mark.docs
def test_example_reads_real_queryresult_attributes():
    """The key functions destructure QueryResult; catch a renamed field."""
    fields = set(QueryResult.__dataclass_fields__)
    for name in ("id", "score", "content", "metadata", "document_id"):
        assert name in fields, f"QueryResult no longer has '{name}' -- examples read it"


@pytest.mark.docs
def test_judgments_match_the_sample_corpus():
    """Judgments name documents and headings; both must still exist.

    This is the check that earns its keep. Retitle a heading in the sample
    corpus and the judgments silently stop matching -- the example still runs
    and simply reports worse numbers, which is the most misleading possible
    failure for a script whose whole purpose is reporting numbers.
    """
    data = json.loads(JUDGMENTS.read_text(encoding="utf-8"))
    queries = data["queries"]
    assert queries, "sample judgments are empty"

    # doc id -> set of headings, mirroring how upsert_from_file ids documents
    # (filename stem) and how sections are detected (Markdown headings).
    corpus: dict[str, set[str]] = {}
    for md in SAMPLE_CORPUS.glob("*.md"):
        headings = set(re.findall(r"^#{1,6}\s+(.+?)\s*$", md.read_text(encoding="utf-8"), re.MULTILINE))
        corpus[md.stem] = headings
    assert corpus, "sample corpus has no .md files"

    for q in queries:
        for doc in q["relevant_docs"]:
            assert doc in corpus, f"judgments reference unknown document {doc!r}; have {sorted(corpus)}"
        for ref in q["relevant_sections"]:
            doc, _, heading = ref.partition("::")
            assert doc in corpus, f"judgments reference unknown document {doc!r} in {ref!r}"
            assert heading in corpus[doc], f"judgments reference heading {heading!r} not found in {doc}.md"


@pytest.mark.docs
def test_example_refuses_mock_embeddings():
    """Mock cannot measure relevance; the example must not pretend otherwise."""
    proc = subprocess.run(
        [sys.executable, str(SECTION_VS_CHUNK), "--provider", "mock"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0, "example accepted --provider mock; it must refuse"
    assert "mock" in (proc.stdout + proc.stderr).lower()


def _provider_available() -> bool:
    sys.path.insert(0, str(EXAMPLES))
    try:
        mod = _load_example(SECTION_VS_CHUNK)
        return mod._ollama_available() or mod._sentence_transformers_available()
    finally:
        sys.path.pop(0)


@pytest.mark.docs
@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(
    not _provider_available(),
    reason="needs a real embedding backend (ollama or sentence-transformers)",
)
def test_section_vs_chunk_runs_end_to_end():
    """Actually run it. Everything above only proves it would not crash on import."""
    proc = subprocess.run(
        [sys.executable, str(SECTION_VS_CHUNK)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, f"example failed:\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    out = proc.stdout
    assert "Finding the right DOCUMENT" in out
    assert "Finding the right SECTION" in out
    # The caveats are the honest half of the output; if they stop rendering, the
    # tables start reading as evidence they are not.
    assert "How to read this" in out
