"""Retrieval quality evaluation -- does the right document come back?

    ./.venv/Scripts/python.exe benchmarks/eval_retrieval.py

This is the T3-min harness, and it is a prerequisite for any T1 retrieval change.
The rest of ``benchmarks/`` measures *geometric* recall (does FAISS find the true
nearest vector) and latency. Neither says anything about relevance, so before
this existed there was no way to tell whether a change to fusion, reranking, or
document scoring helped or hurt.

Why real embeddings: ``MockEmbeddings`` seeds ``np.random`` on a SHA-256 of the
text, so semantically related strings get *orthogonal* vectors. No test using it
can possibly measure whether the right document ranks first.

Why the real ``sentence_transformers`` provider rather than precomputed vectors:
the provider's own ``normalize`` handling is part of what T1.4 is about. Feeding
vectors in through ``PrecomputedEmbeddings`` would bypass exactly the code under
evaluation.

Design note: every configuration in the sweep varies only *query-time*
parameters, so the database is built once and reused. Rebuilding is keyed on the
things that actually change the index (dataset, model, index type, chunking).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _fix_sys_path() -> None:
    """Put the project root on ``sys.path`` and take ``benchmarks/`` off it.

    Running ``python benchmarks/eval_retrieval.py`` makes Python prepend
    ``benchmarks/`` to ``sys.path``, where ``datasets.py``, ``config.py``,
    ``providers.py`` and ``utils.py`` all shadow importable top-level modules.
    ``sentence_transformers`` imports ``datasets``, gets our SIFT loader instead,
    and dies on its relative import -- which ``validate_model()`` swallows and
    reports as "Embedding model 'all-MiniLM-L6-v2' is not available".

    Must run before anything drags in ``sentence_transformers``.
    """
    here = Path(__file__).resolve().parent
    root = here.parent
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != here]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_fix_sys_path()

from benchmarks.config import (  # noqa: E402
    BASELINE_JSON,
    DATA_DIR,
    EVAL_DATASET,
    EVAL_EMBEDDING_MODEL,
    EVAL_EMBEDDING_PROVIDER,
    EVAL_K,
    EVAL_RECALL_K_VALUES,
    EVAL_REGRESSION_TOLERANCE,
    EVAL_RERANKER_MODEL,
    EVAL_RERANKER_PROVIDER,
    RESULTS_DIR,
)
from benchmarks.metrics import evaluate  # noqa: E402

logger = logging.getLogger("benchmarks.eval")

PRIMARY_METRIC = f"ndcg@{EVAL_K}"

# The library's own defaults, spelled out so a change to them shows up as a diff
# here rather than silently moving the baseline.
DEFAULT_INDEX_TYPE = "IndexFlatL2"
DEFAULT_CHUNKING = {"chunking_method": "sentences", "chunk_size": 500, "chunk_overlap": 1}

# T1.6 pruned document scoring to these three (see benchmarks/RETRIEVAL_BASELINE.md). The
# other eight were measured on NFCorpus and none beat these; they are gone from the library,
# so requesting one now raises ValueError. ALL_SCORING_METHODS == CORE_SCORING_METHODS is kept
# so the historical --all-scoring flag still runs (it is now a no-op alias for the core sweep).
CORE_SCORING_METHODS = ("best", "average", "frequency_boost")
ALL_SCORING_METHODS = CORE_SCORING_METHODS

# (search_type, vector_weight). vector_weight is ignored unless search_type is hybrid.
SEARCH_VARIANTS = (
    ("vector", None),
    ("keyword", None),
    ("hybrid", 0.3),
    ("hybrid", 0.5),
    ("hybrid", 0.7),  # library default
    ("hybrid", 0.9),
)


@dataclass(frozen=True)
class EvalConfig:
    search_type: str
    document_scoring_method: str
    vector_weight: Optional[float] = None
    rerank: bool = False

    @property
    def label(self) -> str:
        head = self.search_type
        if self.vector_weight is not None:
            head += f" vw={self.vector_weight:g}"
        tail = " +rerank" if self.rerank else ""
        return f"{head} · {self.document_scoring_method}{tail}"

    @property
    def is_library_default(self) -> bool:
        return (
            self.search_type == "hybrid"
            and self.vector_weight == 0.5
            and self.document_scoring_method == "frequency_boost"
            and not self.rerank
        )


def build_sweep(*, all_scoring: bool, rerank: bool) -> List[EvalConfig]:
    methods = ALL_SCORING_METHODS if all_scoring else CORE_SCORING_METHODS
    configs = [
        EvalConfig(search_type=st, vector_weight=vw, document_scoring_method=m)
        for st, vw in SEARCH_VARIANTS
        for m in methods
    ]
    if rerank:
        # Reranking is a reordering of the top-k the search already returned
        # (see T1.2), so it only needs one scoring method to be informative.
        configs += [
            EvalConfig(search_type=st, vector_weight=vw, document_scoring_method="frequency_boost", rerank=True)
            for st, vw in SEARCH_VARIANTS
        ]
    return configs


def preflight_embedding_model(provider: str, model: str) -> None:
    """Load the embedding model up front, surfacing the real error if it fails.

    ``LocalVectorDB.__init__`` calls ``provider.validate_model()``, which catches
    *every* exception and turns it into a bare "model is not available". A
    missing dependency, a shadowed import, and an offline HuggingFace hub are
    then indistinguishable. Fail here instead, with the traceback intact.
    """
    from localvectordb.embeddings import EmbeddingRegistry

    instance = EmbeddingRegistry.create_provider(provider, model)
    loader = getattr(instance, "_load_model", None)
    if loader is None:  # not every provider is model-loading; nothing to check
        return
    try:
        loader()
    except Exception as exc:
        raise RuntimeError(
            f"Embedding model {model!r} ({provider}) failed to load. "
            f"LocalVectorDB would have reported only 'not available'."
        ) from exc


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return out.stdout.strip()


def _git_commit() -> str:
    """Identify the tree these numbers were measured on.

    A dirty tree is marked as such. A baseline records what the code scored, and
    `HEAD` alone would attribute uncommitted changes to the commit that predates
    them -- which is precisely how a baseline comes to describe code that never
    produced it.
    """
    try:
        commit = _git("rev-parse", "--short", "HEAD")
        return f"{commit}-dirty" if _git("status", "--porcelain") else commit
    except Exception:  # pragma: no cover - a benchmark should not die over provenance
        return "unknown"


def _database_key(dataset: str, model: str, index_type: str, chunking: Dict[str, Any], max_docs: Optional[int]) -> str:
    parts = [dataset, model.replace("/", "_"), index_type]
    parts += [f"{k}={v}" for k, v in sorted(chunking.items())]
    if max_docs is not None:
        parts.append(f"max{max_docs}")
    return "__".join(parts)


def build_database(
    dataset,
    *,
    provider: str,
    model: str,
    index_type: str,
    chunking: Dict[str, Any],
    max_docs: Optional[int],
    rebuild: bool,
):
    """Build (or reopen) the evaluation database. Returns an open LocalVectorDB.

    The build is the expensive step -- every document is chunked and embedded --
    so it is cached on disk under a key covering everything that affects the
    index. A sentinel file marks completion: an interrupted build leaves a
    partially-populated database that would otherwise be silently reused and
    quietly depress every metric.
    """
    from localvectordb import LocalVectorDB

    key = _database_key(dataset.name, model, index_type, chunking, max_docs)
    base = DATA_DIR / "db"
    base.mkdir(parents=True, exist_ok=True)
    sentinel = base / f"{key}.complete"

    def _discard() -> None:
        # LocalVectorDB stores `<base>/<name>.sqlite` and `<base>/<name>.faiss`,
        # not a `<base>/<name>/` directory. The trailing dot in the glob keeps a
        # key from matching a longer key that merely starts with it (e.g.
        # `...sentences` must not match `...sentences__max400`).
        for path in base.glob(f"{key}.*"):
            path.unlink(missing_ok=True)

    if rebuild:
        _discard()

    if sentinel.exists():
        logger.info("Reusing cached database %s", key)
        return LocalVectorDB(
            key, base, embedding_provider=provider, embedding_model=model, faiss_index_type=index_type, **chunking
        )

    if any(base.glob(f"{key}.*")):
        # A build that died partway leaves a database that looks perfectly
        # usable and would quietly depress every metric. Only the sentinel,
        # written after the final save, certifies a complete ingest.
        logger.warning("Discarding incomplete database %s", key)
        _discard()

    logger.info("Building database %s (%d documents) -- this is the slow part", key, len(dataset.corpus))
    db = LocalVectorDB(
        key, base, embedding_provider=provider, embedding_model=model, faiss_index_type=index_type, **chunking
    )
    doc_ids = list(dataset.corpus)
    # One upsert call rewrites the whole FAISS file once (see T2.4), so ingest in
    # slabs rather than per document: 5k documents one at a time would be 5k
    # full-index rewrites.
    slab = 500
    for start in range(0, len(doc_ids), slab):
        batch = doc_ids[start : start + slab]
        db.upsert([dataset.corpus[d] for d in batch], ids=batch)
        logger.info("  ingested %d/%d", min(start + slab, len(doc_ids)), len(doc_ids))

    db.save()
    sentinel.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return db


def _make_reranker():
    from localvectordb.reranking import RerankerRegistry

    return RerankerRegistry.create_reranker(EVAL_RERANKER_PROVIDER, EVAL_RERANKER_MODEL)


def run_config(db, dataset, config: EvalConfig, *, k: int, reranker=None) -> Dict[str, List[str]]:
    """Execute every query under one configuration; return qid -> ranked doc ids."""
    kwargs: Dict[str, Any] = {
        "search_type": config.search_type,
        "return_type": "documents",
        "k": k,
        "document_scoring_method": config.document_scoring_method,
    }
    if config.vector_weight is not None:
        kwargs["vector_weight"] = config.vector_weight
    if config.rerank:
        kwargs["reranker"] = reranker

    run: Dict[str, List[str]] = {}
    for query_id, text in dataset.queries.items():
        run[query_id] = [hit.id for hit in db.query(text, **kwargs)]
    return run


def format_table(results: Dict[str, Dict[str, float]], *, default_label: Optional[str] = None) -> str:
    metric_names = [f"recall@{k}" for k in EVAL_RECALL_K_VALUES] + [PRIMARY_METRIC]
    width = max(len(label) for label in results) + 2
    header = f"| {'configuration'.ljust(width)} | " + " | ".join(m.rjust(9) for m in metric_names) + " |"
    rule = f"|{'-' * (width + 2)}|" + "|".join("-" * 11 for _ in metric_names) + "|"

    rows = []
    for label, scores in sorted(results.items(), key=lambda kv: -kv[1][PRIMARY_METRIC]):
        marker = " ←default" if label == default_label else ""
        cells = " | ".join(f"{scores[m]:9.4f}" for m in metric_names)
        rows.append(f"| {(label + marker).ljust(width)} | {cells} |")
    return "\n".join([header, rule, *rows])


def build_report(dataset, results: Dict[str, Dict[str, float]], *, k: int, index_type: str, chunking, model, provider):
    default = next((c.label for c in build_sweep(all_scoring=False, rerank=False) if c.is_library_default), None)
    return {
        "schema": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "dataset": {
            "name": dataset.name,
            "documents": len(dataset.corpus),
            "queries": len(dataset.queries),
        },
        "embedding": {"provider": provider, "model": model},
        "index": {"faiss_index_type": index_type},
        "chunking": dict(chunking),
        "k": k,
        "primary_metric": PRIMARY_METRIC,
        "default_config": default,
        "results": results,
    }


def compare_to_baseline(results: Dict[str, Dict[str, float]], baseline_path: Path, tolerance: float) -> int:
    """Diff against the committed baseline. Returns a process exit code."""
    if not baseline_path.exists():
        print(f"No baseline at {baseline_path}. Run with --save-baseline first.", file=sys.stderr)
        return 1

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    old = baseline["results"]
    print(f"\nBaseline: {baseline['git_commit']} ({baseline['generated']})")
    print(f"{'configuration':<44} {'baseline':>9} {'current':>9} {'delta':>9}")
    print("-" * 74)

    regressions, missing = [], []
    for label in sorted(set(old) | set(results)):
        if label not in results:
            missing.append(label)
            continue
        if label not in old:
            print(f"{label:<44} {'--':>9} {results[label][PRIMARY_METRIC]:>9.4f} {'new':>9}")
            continue
        before, after = old[label][PRIMARY_METRIC], results[label][PRIMARY_METRIC]
        delta = after - before
        flag = ""
        if delta < -tolerance:
            regressions.append((label, before, after))
            flag = "  REGRESSION"
        print(f"{label:<44} {before:>9.4f} {after:>9.4f} {delta:>+9.4f}{flag}")

    if missing:
        print(f"\nNot run this time (sweep changed?): {', '.join(missing)}", file=sys.stderr)
    if regressions:
        print(f"\n{len(regressions)} configuration(s) regressed on {PRIMARY_METRIC} by > {tolerance}.", file=sys.stderr)
        return 1
    print(f"\nNo regression on {PRIMARY_METRIC} beyond {tolerance}.")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieval relevance evaluation (nDCG@10 / recall@k) on a BEIR dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python benchmarks/eval_retrieval.py                      # default sweep, print table
  python benchmarks/eval_retrieval.py --save-baseline      # record the T1 baseline
  python benchmarks/eval_retrieval.py --check              # fail if nDCG@10 regressed
  python benchmarks/eval_retrieval.py --all-scoring        # evidence for T1.6
  python benchmarks/eval_retrieval.py --rerank             # evidence for T1.2 (slow)
  python benchmarks/eval_retrieval.py --max-docs 500 --max-queries 30   # smoke test
""",
    )
    p.add_argument("--dataset", default=EVAL_DATASET, choices=("scifact", "nfcorpus"))
    p.add_argument("--embedding-provider", default=EVAL_EMBEDDING_PROVIDER)
    p.add_argument("--embedding-model", default=EVAL_EMBEDDING_MODEL)
    p.add_argument("--index-type", default=DEFAULT_INDEX_TYPE)
    p.add_argument("--k", type=int, default=EVAL_K)
    p.add_argument(
        "--all-scoring",
        action="store_true",
        help="Deprecated no-op since T1.6 pruned scoring to 3 methods; sweeps the same core methods.",
    )
    p.add_argument("--rerank", action="store_true", help="Add cross-encoder reranked variants (T1.2). Slow on CPU.")
    p.add_argument("--rebuild", action="store_true", help="Discard the cached database and re-ingest.")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--save-baseline", action="store_true", help=f"Write {BASELINE_JSON.name} (tracked in git).")
    p.add_argument("--check", action="store_true", help="Compare against the committed baseline; non-zero on drop.")
    p.add_argument("--baseline", type=Path, default=BASELINE_JSON, help="Baseline file for --check/--save-baseline.")
    p.add_argument("--tolerance", type=float, default=EVAL_REGRESSION_TOLERANCE)
    p.add_argument("--max-docs", type=int, default=None, help="Smoke test only. Inflates every metric.")
    p.add_argument("--max-queries", type=int, default=None, help="Smoke test only.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    from benchmarks import beir_data

    if args.download_only:
        beir_data.download(args.dataset)
        return 0

    dataset = beir_data.load(args.dataset, max_docs=args.max_docs)
    if args.max_queries is not None:
        keep = list(dataset.queries)[: args.max_queries]
        dataset = beir_data.BeirDataset(
            name=dataset.name,
            corpus=dataset.corpus,
            queries={q: dataset.queries[q] for q in keep},
            qrels={q: dataset.qrels[q] for q in keep},
        )
    smoke = args.max_docs is not None or args.max_queries is not None
    logger.info("%r", dataset)

    preflight_embedding_model(args.embedding_provider, args.embedding_model)

    chunking = dict(DEFAULT_CHUNKING)
    db = build_database(
        dataset,
        provider=args.embedding_provider,
        model=args.embedding_model,
        index_type=args.index_type,
        chunking=chunking,
        max_docs=args.max_docs,
        rebuild=args.rebuild,
    )

    reranker = _make_reranker() if args.rerank else None
    configs = build_sweep(all_scoring=args.all_scoring, rerank=args.rerank)
    results: Dict[str, Dict[str, float]] = {}
    try:
        for i, config in enumerate(configs, 1):
            run = run_config(db, dataset, config, k=args.k, reranker=reranker)
            scores = evaluate(run, dataset.qrels, k_values=EVAL_RECALL_K_VALUES)
            results[config.label] = scores
            logger.info("[%2d/%d] %-46s %s=%.4f", i, len(configs), config.label, PRIMARY_METRIC, scores[PRIMARY_METRIC])
    finally:
        db.close()

    report = build_report(
        dataset,
        results,
        k=args.k,
        index_type=args.index_type,
        chunking=chunking,
        model=args.embedding_model,
        provider=args.embedding_provider,
    )

    print()
    print(format_table(results, default_label=report["default_config"]))
    best = max(results.items(), key=lambda kv: kv[1][PRIMARY_METRIC])
    default_scores = results.get(report["default_config"] or "")
    print()
    if default_scores:
        print(f"Library default : {report['default_config']}  {PRIMARY_METRIC}={default_scores[PRIMARY_METRIC]:.4f}")
    print(f"Best in sweep   : {best[0]}  {PRIMARY_METRIC}={best[1][PRIMARY_METRIC]:.4f}")

    if smoke:
        print("\nSMOKE RUN -- corpus and/or queries truncated. Not a valid baseline.", file=sys.stderr)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS_DIR / f"retrieval_{args.dataset}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)

    if args.save_baseline:
        if smoke:
            print("Refusing to save a baseline from a truncated run.", file=sys.stderr)
            return 1
        args.baseline.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nBaseline written to {args.baseline}")

    if args.check:
        return compare_to_baseline(results, args.baseline, args.tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
