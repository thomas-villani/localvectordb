"""Compare section-level retrieval against chunk-level retrieval on your corpus.

LocalVectorDB's headline claim is that in long, structured documents the answer
is usually spread across a whole *section*, and that embedding each section's
own text and retrieving it alongside chunks measurably beats chunk-only
retrieval. The published study measures that on Qasper:

    https://thomas-villani.github.io/localvectordb/hierarchical-evaluation.html

This script exists so you do not have to take that on faith. Point it at your
own documents and your own queries and it will run the same comparison and
print the same kind of table.

Usage
-----
    # The bundled sample corpus, with judgments, end to end:
    python examples/section_vs_chunk_retrieval.py

    # Your corpus, your judgments -> nDCG@10 and recall@k per retrieval mode:
    python examples/section_vs_chunk_retrieval.py \
        --corpus ./docs --judgments ./my_judgments.json

    # Your corpus, no judgments -> a side-by-side of what each mode returns:
    python examples/section_vs_chunk_retrieval.py --corpus ./docs --query "..."

Requirements
------------
A real embedding backend: either Ollama (``ollama serve`` and
``ollama pull nomic-embed-text``) or the ``sentence-transformers`` package.
Auto-detected, override with ``--provider``.

**Not** the ``mock`` provider, and this script refuses to use it. MockEmbeddings
seeds numpy's RNG from a hash of the input text, so two sentences about the same
subject get unrelated vectors. Mock is perfectly good for exercising plumbing
and useless for measuring relevance -- a mock run would produce a table of noise
that looks exactly like a table of results.

Reading the output honestly
---------------------------
The bundled sample corpus is three documents and twelve queries. That is enough
to show you the mechanism and the workflow; it is not enough to *establish*
anything, and if you see section retrieval win here by some margin, that margin
is not a measurement of anything general. The evidence for the claim is the
study linked above (15 papers, 48 queries, three encoders). The point of this
script is the third column of the table: what happens on *your* data.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

HERE = Path(__file__).resolve().parent
SAMPLE_CORPUS = HERE / "sample_corpus"
SAMPLE_JUDGMENTS = SAMPLE_CORPUS / "judgments.json"

OLLAMA_MODEL = "nomic-embed-text"
SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"

K = 10
RECALL_K_VALUES = (1, 5, 10)


# --------------------------------------------------------------------------
# Metrics.
#
# Deliberately reimplemented here rather than imported from benchmarks/, which
# is not part of the distributed package -- an example that only runs inside a
# git checkout is not much of an example. These follow trec_eval conventions:
# linear gain, ideal DCG computed over all judged-relevant items for the query.
# --------------------------------------------------------------------------


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """nDCG@k for binary relevance."""
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel & set(ranked[:k])) / len(rel)


# --------------------------------------------------------------------------
# Backend detection.
# --------------------------------------------------------------------------


def _ollama_available() -> bool:
    try:
        import os

        import requests

        url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        resp = requests.get(f"{url}/api/tags", timeout=2)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return any(n.split(":")[0] == OLLAMA_MODEL for n in models)
    except Exception:
        return False


def _sentence_transformers_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


def detect_provider(preferred: str | None) -> tuple[str, str]:
    if preferred == "mock":
        sys.exit(
            "Refusing to run on mock embeddings: they cannot measure relevance.\n"
            "MockEmbeddings seeds numpy's RNG from a hash of the text, so related\n"
            "sentences get unrelated vectors. The table would be noise."
        )
    if preferred == "ollama" or (preferred is None and _ollama_available()):
        if not _ollama_available():
            sys.exit(f"Ollama requested but unavailable. Start it: ollama serve && ollama pull {OLLAMA_MODEL}")
        return "ollama", OLLAMA_MODEL
    if preferred in (None, "sentence_transformers") and _sentence_transformers_available():
        return "sentence_transformers", SENTENCE_TRANSFORMERS_MODEL
    sys.exit(
        "No real embedding backend found.\n"
        f"  Either: ollama serve && ollama pull {OLLAMA_MODEL}\n"
        "  Or:     pip install sentence-transformers"
    )


# --------------------------------------------------------------------------
# The comparison.
#
# One database serves every arm. `hierarchical_embeddings=True` adds section
# vectors; it does not change chunk retrieval, so search_level="chunks" on this
# database is the same chunk-only baseline you would get without the flag. That
# makes the arms differ *only* in how they are queried, which is the whole point
# -- rebuilding per arm would let ingestion differences leak into the result.
# --------------------------------------------------------------------------

# (label, query kwargs). return_type is added per target below.
ARMS: List[tuple[str, Dict[str, Any]]] = [
    ("chunk-only (baseline)", {"search_level": "chunks"}),
    ("raw-span sections", {"search_level": "sections"}),
    ("fused (section_weight=0.65)", {"search_level": "fused", "section_weight": 0.65}),
]


def section_key(result: Any) -> str:
    """'<doc id>::<section heading>' -- the id shape used in the judgments file."""
    heading = result.metadata.get("section_heading")
    return f"{result.document_id}::{heading}"


def document_key(result: Any) -> str:
    """The parent document of a result, whatever level the result came from.

    Every arm is asked for return_type="documents" at the document target, so
    each result is already a whole document and `id` is the document id. The
    fallback to `document_id` keeps this honest for a result that arrived at
    some other level: a document is ranked by its best-matching passage, which
    is standard practice in passage retrieval and what the study does.
    """
    return result.document_id or result.id


def dedupe(items: Sequence[str]) -> List[str]:
    """First occurrence wins -- preserves rank order."""
    seen: set = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_db(corpus: Path, workdir: Path, provider: str, model: str, chunk_size: int):
    from localvectordb import VectorDB

    files = sorted(p for p in corpus.rglob("*") if p.suffix.lower() in {".md", ".markdown", ".txt"})
    if not files:
        sys.exit(f"No .md/.markdown/.txt files found under {corpus}")

    print(f"Corpus     : {len(files)} file(s) from {corpus}")
    print(f"Provider   : {provider} / {model}")
    print(f"Chunk size : {chunk_size}")
    print("Building the database (embedding chunks + one vector per section)...")

    db = VectorDB(
        "section_vs_chunk",
        workdir,
        embedding_provider=provider,
        embedding_model=model,
        hierarchical_embeddings=True,
        chunk_size=chunk_size,
    )
    db.upsert_from_file([str(p) for p in files])
    db.save()

    # Sections are detected from Markdown headings. If a corpus has none, every
    # document is one big section and the section arm degenerates into a
    # document arm -- worth saying out loud rather than quietly scoring 0.
    n_sections = sum(
        1
        for f in files
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith("#")
    )
    print(f"Sections   : {n_sections} heading(s) detected across the corpus")
    if n_sections == 0:
        print(
            "\nWARNING: no Markdown headings found. Sections are detected from Markdown\n"
            "         headings by default, so section retrieval has nothing to work with\n"
            "         here and this comparison will not be meaningful. Pass a\n"
            "         section_pattern matching your own heading convention."
        )
    print()
    return db, len(files)


def run_arms(db, queries: List[Dict[str, Any]], target: str) -> Dict[str, Dict[str, float]]:
    """Score every arm at one target ('document' or 'section')."""
    return_type = "documents" if target == "document" else "sections"
    rel_field = "relevant_docs" if target == "document" else "relevant_sections"
    key = document_key if target == "document" else section_key

    scored: Dict[str, Dict[str, float]] = {}
    for label, kwargs in ARMS:
        per_query: List[tuple[List[str], List[str]]] = []
        # Every arm is asked for the same unit, so each is scored on a list of
        # the same length. The section arm's roll-up to documents (including the
        # over-fetch that keeps K documents reachable) is the library's job, not
        # this script's -- passing return_type is all it takes.
        for q in queries:
            relevant = q.get(rel_field) or []
            if not relevant:
                continue
            results = db.query(q["query"], k=K, return_type=return_type, **kwargs)
            ranked = dedupe([key(r) for r in results])[:K]
            per_query.append((ranked, relevant))

        if not per_query:
            continue
        metrics = {f"ndcg@{K}": sum(ndcg_at_k(r, rel, K) for r, rel in per_query) / len(per_query)}
        for kk in RECALL_K_VALUES:
            metrics[f"recall@{kk}"] = sum(recall_at_k(r, rel, kk) for r, rel in per_query) / len(per_query)
        scored[label] = metrics
    return scored


def format_table(scored: Dict[str, Dict[str, float]], title: str, baseline: str) -> str:
    if not scored:
        return f"{title}\n  (no judgments for this target)\n"

    metrics = list(next(iter(scored.values())).keys())
    label_w = max(len(x) for x in scored) + 2
    head = "  ".join(f"{m:>10s}" for m in metrics)
    lines = [title, "-" * len(title), f"{'':{label_w}s}{head}   vs baseline"]

    base = scored.get(baseline, {})
    for label, m in scored.items():
        row = "  ".join(f"{m[k]:10.4f}" for k in metrics)
        if label == baseline or not base:
            delta = ""
        else:
            d = m[f"ndcg@{K}"] - base[f"ndcg@{K}"]
            delta = f"   {d:+.4f} nDCG@{K}"
        lines.append(f"{label:{label_w}s}{row}{delta}")
    return "\n".join(lines) + "\n"


def show_side_by_side(db, query: str) -> None:
    """No judgments: show what each mode actually returns, and let the reader judge."""
    print(f'Query: "{query}"\n')
    for label, kwargs in ARMS:
        # Ask each arm for sections, since the interesting question is *what span
        # of text* each mode considers the answer. Note search_level="chunks"
        # with return_type="sections" is a roll-up: chunks are retrieved, then
        # reported under the section they fell in. That is the honest comparison
        # against genuine section retrieval, which embeds the section itself.
        results = db.query(query, k=3, return_type="sections", **kwargs)
        print(f"--- {label} ---")
        if not results:
            print("    (nothing returned)\n")
            continue
        for i, r in enumerate(results, 1):
            heading = r.metadata.get("section_heading") or "(preamble)"
            snippet = " ".join(r.content.split())[:130]
            print(f"  {i}. [{r.score:.3f}] {r.document_id} > {heading}")
            print(f"     {snippet}...")
        print()


STUDY_URL = "https://thomas-villani.github.io/localvectordb/hierarchical-evaluation.html"
# The published rawspan-section gain over the chunk baseline, for calibration.
STUDY_SECTION_GAIN = (0.07, 0.17)


def report_caveats(
    all_scored: Dict[str, Dict[str, Dict[str, float]]],
    n_docs: int,
    n_queries: int,
    is_sample: bool,
) -> None:
    """Say what the table above does and does not support.

    A table of numbers reads as evidence whether or not it is any. These checks
    are computed from the run rather than hardcoded, so they stay true if the
    corpus changes.
    """
    notes: List[str] = []
    baseline = ARMS[0][0]

    doc = all_scored.get("document", {})
    if doc and len({round(m[f"ndcg@{K}"], 4) for m in doc.values()}) == 1 and n_docs <= K:
        notes.append(
            f"* The DOCUMENT target is saturated. With {n_docs} documents and k={K}, every\n"
            f"  arm retrieves the whole corpus, so all of them score identically. That is a\n"
            f"  fact about the corpus size, not a finding that the modes are equivalent.\n"
            f"  This target only becomes informative when the corpus is much larger than k."
        )

    sec = all_scored.get("section", {})
    if sec and baseline in sec:
        best = max((m[f"ndcg@{K}"] - sec[baseline][f"ndcg@{K}"]) for m in sec.values())
        lo, hi = STUDY_SECTION_GAIN
        if best > hi:
            notes.append(
                f"* The SECTION gain here (+{best:.2f} nDCG@{K}) is much larger than the published\n"
                f"  study's (+{lo:.2f} to +{hi:.2f}). Do not quote it. A margin this size means the\n"
                f"  corpus is close to the best case for section retrieval -- short documents,\n"
                f"  clean headings, and one obviously-correct section per query. Real corpora\n"
                f"  are messier, which is exactly why the study was run on real papers."
            )

    if n_queries < 30:
        notes.append(
            f"* {n_queries} queries is too few for the differences above to be stable. Expect\n"
            f"  any single number to move by a lot if you reword a query or two."
        )

    if not notes:
        return
    print("How to read this")
    print("----------------")
    print("\n".join(notes))
    if is_sample:
        print(
            "\nThe sample corpus exists to show you the mechanism and the workflow. The\n"
            "number worth having comes from --corpus pointed at your own documents."
        )
    print(f"\nThe published study: {STUDY_URL}")


def load_judgments(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data.get("queries") if isinstance(data, dict) else data
    if not queries:
        sys.exit(f"{path} contains no queries. See {SAMPLE_JUDGMENTS} for the shape.")
    return queries


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--corpus", type=Path, default=SAMPLE_CORPUS, help="Directory of .md/.txt documents")
    p.add_argument("--judgments", type=Path, default=None, help="Relevance judgments JSON (see sample_corpus/)")
    p.add_argument("--query", default=None, help="Single query for a qualitative side-by-side (no judgments needed)")
    p.add_argument("--provider", choices=["ollama", "sentence_transformers"], default=None)
    p.add_argument("--chunk-size", type=int, default=500, help="Smaller chunks make sections span more chunks")
    p.add_argument("--keep", action="store_true", help="Keep the built database instead of deleting it")
    args = p.parse_args()

    if args.judgments is None and args.corpus == SAMPLE_CORPUS and args.query is None:
        args.judgments = SAMPLE_JUDGMENTS

    provider, model = detect_provider(args.provider)
    workdir = Path(tempfile.mkdtemp(prefix="lvdb-section-vs-chunk-"))
    try:
        db, n_docs = build_db(args.corpus, workdir, provider, model, args.chunk_size)
        try:
            if args.query:
                show_side_by_side(db, args.query)
                return 0

            if args.judgments is None:
                print(
                    "No --judgments given, so there is nothing to score against.\n"
                    "Either pass --query 'your question' for a side-by-side, or write\n"
                    f"judgments in the shape of {SAMPLE_JUDGMENTS} and pass --judgments."
                )
                return 2

            queries = load_judgments(args.judgments)
            print(f"Judgments  : {len(queries)} queries from {args.judgments}\n")

            baseline = ARMS[0][0]
            all_scored = {}
            for target, blurb in (
                ("document", "Finding the right DOCUMENT"),
                ("section", "Finding the right SECTION"),
            ):
                all_scored[target] = run_arms(db, queries, target)
                print(format_table(all_scored[target], blurb, baseline))

            report_caveats(all_scored, n_docs, len(queries), args.corpus == SAMPLE_CORPUS)
            return 0
        finally:
            db.close()
    finally:
        if args.keep:
            print(f"\nDatabase kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
