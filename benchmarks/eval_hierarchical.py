"""Hierarchical retrieval experiment engine (Phase 1 / E1-E2).

    ./.venv/Scripts/python.exe benchmarks/eval_hierarchical.py --dataset nfcorpus

See ``hierarchical-test-plan.md``. This measures, on the synthetic super-document
set from ``benchmarks/superdocs.py``, whether representing a section or a whole
document by the **embedding of its actual text** (raw-span) beats the current
**centroid** (mean of chunk embeddings), and whether **fusing** all levels beats
any single level and the chunk-only baseline. It also computes the **oracle**
ceiling of a perfect per-query level selector -- the H4 gate that decides whether
a selector is worth building at all.

Everything here is harness-side on purpose (plan decision, 2026-07-11): no change
to ``src/`` until the numbers justify one. Level vectors are built with the real
embedding provider and ranked by exact cosine in NumPy -- no FAISS, so no
mock-index hazards and no approximation between us and the geometry.

Why real embeddings: ``MockEmbeddings`` gives semantically related texts
orthogonal vectors, so it cannot tell whether the right document ranks first.
The decided reference encoder is OpenAI ``text-embedding-3-small`` (needs
``OPENAI_API_KEY``); ``sentence_transformers`` works for a local, short-context
plumbing check but its 512-token cap truncates section/doc spans, so its numbers
validate the *engine*, not the hypothesis.

Arms
----
``chunk``                today's effective baseline: rank chunks, map to doc.
``centroid-section``     section = mean of its chunk vectors (unit-norm, T1.5).
``centroid-doc``         document = mean of all its chunk vectors.
``rawspan-section``      section = embedding of the section's text (H1).
``rawspan-doc``          document = embedding of the whole document's text (H1).
``fusion-centroid``      relative-score fusion of chunk+centroid levels (H3).
``fusion-rawspan``       relative-score fusion of chunk+rawspan levels (H3).
``oracle-*``             per-query best single level -- the ceiling (H4).

Every arm is scored on the **same** document-level qrels, so a section/chunk hit
is mapped up to its parent document before scoring. That makes all arms
comparable on one number (nDCG@10) and is exactly what the fusion and oracle
analyses need.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


def _fix_sys_path() -> None:
    """Root on ``sys.path``, ``benchmarks/`` off it -- see ``eval_retrieval.py``.

    ``sentence_transformers`` imports ``datasets``; with ``benchmarks/`` on the
    path it gets our shadowing module instead and dies, which the provider
    reports only as "model not available".
    """
    here = Path(__file__).resolve().parent
    root = here.parent
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != here]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_fix_sys_path()

import numpy as np  # noqa: E402

from benchmarks import beir_data  # noqa: E402
from benchmarks.config import CACHE_DIR, RESULTS_DIR  # noqa: E402
from benchmarks.metrics import evaluate, ndcg_at_k  # noqa: E402
from benchmarks.superdocs import (  # noqa: E402
    SyntheticBenchmark,
    build_synthetic_benchmark,
    section_qrel_id,
)

logger = logging.getLogger("benchmarks.eval_hier")

PRIMARY_K = 10
RECALL_K_VALUES = (1, 5, 10)
PRIMARY_METRIC = f"ndcg@{PRIMARY_K}"

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "text-embedding-3-small"
# The DB's default chunking (see eval_retrieval.DEFAULT_CHUNKING); reproduced here
# so harness chunk boundaries match what LocalVectorDB would ingest.
CHUNKING = {"method": "sentences", "max_tokens": 500, "overlap": 1}


def _unit(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalise; leave all-zero rows at zero (empty-section centroid)."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# OpenAI caps one embeddings request at 300k tokens and 2048 inputs, and
# ``embed_sync`` batches by input *count* only -- so a single batch of long
# section/doc spans (e.g. ~970 sections x ~700 tokens) sails past the token cap.
# Batch here by an estimated-token budget (~3.5 chars/token) with margin.
_MAX_TOKENS_PER_REQUEST = 200_000
_MAX_INPUTS_PER_REQUEST = 2000
_CHARS_PER_TOKEN = 3.5
# A single input over the model's ~8191-token window is a hard 400 from OpenAI.
# Real long docs (Qasper papers, long sections) exceed it, so cap each input to a
# conservative char bound (worst-case ~3 chars/token). Truncation is honest for a
# doc-level raw-span vector -- a single embedding cannot see an entire long paper,
# which is exactly why the doc arm is the weak one. No-op for the short BEIR spans.
_MAX_EMBED_CHARS = 24_000


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _batch_by_budget(texts: Sequence[str]) -> Iterator[List[str]]:
    """Split ``texts`` into request-sized batches under the token and input caps.

    Order is preserved and every text appears exactly once, so concatenating the
    per-batch embeddings reproduces the single-call result. A lone text over the
    token budget is still emitted alone (our spans are <= the encoder window, so
    this cannot exceed the hard cap in practice).
    """
    batch: List[str] = []
    tokens = 0
    for text in texts:
        cost = _estimate_tokens(text)
        if batch and (len(batch) >= _MAX_INPUTS_PER_REQUEST or tokens + cost > _MAX_TOKENS_PER_REQUEST):
            yield batch
            batch, tokens = [], 0
        batch.append(text)
        tokens += cost
    if batch:
        yield batch


class CachedEncoder:
    """Batch-embed with a persistent per-text disk cache keyed on (model, text).

    Embeddings cost money and every sweep re-touches the same spans, so a run
    after the first is free. Cached vectors are the provider's **raw** output;
    normalisation is applied at use, so a change of metric never invalidates the
    cache. Cache lives under ``benchmarks/.cache/`` (gitignored).
    """

    def __init__(self, provider_name: str, model: str) -> None:
        from localvectordb.embeddings import EmbeddingRegistry

        self.provider = EmbeddingRegistry.create_provider(provider_name, model)
        self.model = model
        self.cache_dir = CACHE_DIR / "hier_embed" / f"{provider_name}__{model.replace('/', '_')}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_embedded = 0
        self.n_cached = 0
        self.dim: Optional[int] = None

    def _path(self, text: str) -> Path:
        h = hashlib.sha256(f"{self.model}\x00{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.npy"

    def encode(self, texts: Sequence[str], *, normalize: bool = True) -> np.ndarray:
        """Return an ``(n, dim)`` array for ``texts`` (unit-normalised by default)."""
        if not texts:
            raise ValueError("encode() called with no texts")
        # Truncate over-long inputs to stay under the encoder's context window.
        # Done before hashing so the cache key matches what is actually embedded.
        texts = [t if len(t) <= _MAX_EMBED_CHARS else t[:_MAX_EMBED_CHARS] for t in texts]
        vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        miss_texts: List[str] = []
        miss_idx: List[int] = []
        for i, text in enumerate(texts):
            path = self._path(text)
            if path.exists():
                vectors[i] = np.load(path)
                self.n_cached += 1
            else:
                miss_texts.append(text)
                miss_idx.append(i)

        if miss_texts:
            parts = [
                np.asarray(self.provider.embed_sync(batch), dtype=np.float32) for batch in _batch_by_budget(miss_texts)
            ]
            fresh = np.vstack(parts)
            for j, i in enumerate(miss_idx):
                v = fresh[j].astype(np.float32)
                np.save(self._path(texts[i]), v)
                vectors[i] = v
                self.n_embedded += 1

        out = np.vstack(vectors).astype(np.float32)
        self.dim = out.shape[1]
        return _unit(out) if normalize else out


@dataclass
class LevelIndex:
    """A set of retrieval units at one level, each mapping up to a target unit.

    ``vectors`` are unit-normalised. ``unit_docs[i]`` is the document a unit is
    scored against at the *doc* target (chunk/section -> its parent, doc ->
    itself). ``unit_sections[i]`` is the section id at the *section* target
    (chunk -> its containing section, section -> itself); ``None`` for a level
    with no section (the doc arms), which is then skipped at the section target.
    """

    name: str
    vectors: np.ndarray
    unit_docs: List[str]
    unit_sections: Optional[List[str]] = None


@dataclass
class BuiltVectors:
    """All level indices for one arm-family plus the shared query matrix."""

    query_ids: List[str]
    query_vecs: np.ndarray
    chunk: LevelIndex
    centroid_section: LevelIndex
    centroid_doc: LevelIndex
    rawspan_section: LevelIndex
    rawspan_doc: LevelIndex
    # Present only when --summary was requested (directed-summary arm, H2).
    summary_section: Optional[LevelIndex] = None
    summary_doc: Optional[LevelIndex] = None


def _chunker():
    from localvectordb.chunking import ChunkerFactory

    return ChunkerFactory.create_chunker(CHUNKING["method"], CHUNKING["max_tokens"], CHUNKING["overlap"])


def _detect_sections(text: str):
    from localvectordb.section_detection import SectionDetector

    return SectionDetector().detect_sections(text)


def build_vectors(
    bench: SyntheticBenchmark, encoder: CachedEncoder, summarizer: "Optional[object]" = None
) -> BuiltVectors:
    """Chunk every super-doc, then build all five level indices for the sweep.

    Chunk vectors come from the real chunker (matching the DB). Centroids are the
    mean of a span's raw chunk vectors, then unit-normalised (T1.5 semantics).
    Raw-span vectors are the embedding of the span's own text. Queries embedded
    once and shared by every arm.

    If ``summarizer`` is given (a ``benchmarks.summarize.CachedSummarizer``), also
    build the directed-summary section/doc levels (H2): each span is summarised,
    then the summary is embedded with the same encoder.
    """
    chunker = _chunker()

    # Flat registries, filled per document, embedded in bulk afterwards.
    chunk_texts: List[str] = []
    chunk_doc: List[str] = []
    chunk_section: List[str] = []  # section id containing each chunk, for section-target scoring
    chunk_span: List[Tuple[int, int]] = []  # char span of each chunk, for section grouping

    section_texts: List[str] = []
    section_units: List[str] = []
    section_doc: List[str] = []
    section_span: List[Tuple[int, int]] = []

    doc_texts: List[str] = []
    doc_ids: List[str] = []

    # Remember, per doc, the (start, end) of each chunk so centroids can group by
    # section without a second pass over the text.
    per_doc_chunk_rows: Dict[str, List[int]] = {}

    max_chars = 8191 * 4  # rough token->char; warn if a span likely exceeds the encoder window
    for doc_id, text in bench.corpus.items():
        doc_ids.append(doc_id)
        doc_texts.append(text)
        if len(text) > max_chars:
            logger.warning("Doc %s is ~%d chars; may exceed the encoder context window", doc_id, len(text))

        sections = _detect_sections(text)
        for sec in sections:
            if sec.heading is None:
                continue  # no preamble in our construction, but guard anyway
            section_texts.append(text[sec.start_pos : sec.end_pos])
            section_units.append(section_qrel_id(doc_id, sec.index))
            section_doc.append(doc_id)
            section_span.append((sec.start_pos, sec.end_pos))

        rows: List[int] = []
        for ch in chunker.chunk(text):
            rows.append(len(chunk_texts))
            chunk_texts.append(ch.content)
            chunk_doc.append(doc_id)
            mid = (ch.position.start + ch.position.end) // 2
            owner = next((s for s in sections if s.start_pos <= mid < s.end_pos), sections[-1])
            chunk_section.append(section_qrel_id(doc_id, owner.index))
            chunk_span.append((ch.position.start, ch.position.end))
        per_doc_chunk_rows[doc_id] = rows

    # Bulk embed (raw; normalise at use). One pass each.
    raw_chunk = encoder.encode(chunk_texts, normalize=False)
    chunk_unit = _unit(raw_chunk)
    raw_section = encoder.encode(section_texts, normalize=False)
    raw_doc = encoder.encode(doc_texts, normalize=False)
    query_ids = list(bench.queries)
    query_vecs = encoder.encode([bench.queries[q] for q in query_ids], normalize=True)

    # Centroid indices: mean the raw chunk vectors of each span, then unit-norm.
    doc_row = {d: i for i, d in enumerate(doc_ids)}

    centroid_doc_vecs = np.zeros((len(doc_ids), raw_chunk.shape[1]), dtype=np.float32)
    for d, rows in per_doc_chunk_rows.items():
        if rows:
            centroid_doc_vecs[doc_row[d]] = raw_chunk[rows].mean(axis=0)
    centroid_doc_vecs = _unit(centroid_doc_vecs)

    centroid_section_vecs = np.zeros((len(section_units), raw_chunk.shape[1]), dtype=np.float32)
    for si in range(len(section_units)):
        d = section_doc[si]
        s0, s1 = section_span[si]
        member_rows = [r for r in per_doc_chunk_rows[d] if s0 <= (chunk_span[r][0] + chunk_span[r][1]) // 2 < s1]
        if member_rows:
            centroid_section_vecs[si] = raw_chunk[member_rows].mean(axis=0)
    centroid_section_vecs = _unit(centroid_section_vecs)

    summary_section = summary_doc = None
    if summarizer is not None:
        # Summarise the same section/doc texts, then embed the summaries in the
        # shared vector space. Order is preserved, so the unit maps are reused.
        sec_summaries = summarizer.summarize_all(section_texts)
        doc_summaries = summarizer.summarize_all(doc_texts)
        summary_section = LevelIndex("summary-section", encoder.encode(sec_summaries), section_doc, section_units)
        summary_doc = LevelIndex("summary-doc", encoder.encode(doc_summaries), doc_ids, None)

    return BuiltVectors(
        query_ids=query_ids,
        query_vecs=query_vecs,
        chunk=LevelIndex("chunk", chunk_unit, chunk_doc, chunk_section),
        centroid_section=LevelIndex("centroid-section", centroid_section_vecs, section_doc, section_units),
        centroid_doc=LevelIndex("centroid-doc", centroid_doc_vecs, doc_ids, None),
        rawspan_section=LevelIndex("rawspan-section", _unit(raw_section), section_doc, section_units),
        rawspan_doc=LevelIndex("rawspan-doc", _unit(raw_doc), doc_ids, None),
        summary_section=summary_section,
        summary_doc=summary_doc,
    )


def rank_units(
    query_vecs: np.ndarray, query_ids: List[str], level: LevelIndex, target: str
) -> Optional[Dict[str, List[Tuple[str, float]]]]:
    """Per query, score each *target unit* by its best-matching vector at this level.

    ``target`` is ``"doc"`` or ``"section"``. A target's score is the max cosine
    over the vectors that map to it, so a chunk/section hit is credited to its
    parent doc (or containing section). Returns ``qid -> [(unit, score)]`` ranked
    descending, or ``None`` if this level has no mapping for ``target`` (the doc
    arms have no section, so they are absent from the section target).
    """
    unit_ids = level.unit_docs if target == "doc" else level.unit_sections
    if unit_ids is None:
        return None
    units_unique = sorted(set(unit_ids))
    unit_index = {u: i for i, u in enumerate(units_unique)}
    col = np.fromiter((unit_index[u] for u in unit_ids), count=len(unit_ids), dtype=np.int64)
    sims = query_vecs @ level.vectors.T  # (nq, N)

    run: Dict[str, List[Tuple[str, float]]] = {}
    for qi, qid in enumerate(query_ids):
        acc = np.full(len(units_unique), -np.inf, dtype=np.float64)
        np.maximum.at(acc, col, sims[qi])
        order = np.argsort(-acc)
        run[qid] = [(units_unique[k], float(acc[k])) for k in order if acc[k] > -np.inf]
    return run


def fuse(runs: List[Dict[str, List[Tuple[str, float]]]], query_ids: List[str]) -> Dict[str, List[str]]:
    """Relative-score fusion (mirrors T1.1): min-max normalise each source's
    scores into ``[0, 1]`` over its own pool, then sum per document."""
    fused: Dict[str, List[str]] = {}
    for qid in query_ids:
        agg: Dict[str, float] = {}
        for run in runs:
            pairs = run[qid]
            if not pairs:
                continue
            scores = [s for _, s in pairs]
            lo, hi = min(scores), max(scores)
            span = (hi - lo) or 1.0
            for doc, s in pairs:
                agg[doc] = agg.get(doc, 0.0) + (s - lo) / span
        fused[qid] = [d for d, _ in sorted(agg.items(), key=lambda kv: -kv[1])]
    return fused


def ids_only(run: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[str]]:
    return {qid: [d for d, _ in pairs] for qid, pairs in run.items()}


def oracle_over(
    level_runs: Dict[str, Dict[str, List[str]]], query_ids: List[str], qrels: Dict[str, Dict[str, int]], k: int
) -> Tuple[float, Counter]:
    """Ceiling of a perfect per-query level selector over ``level_runs``.

    For each query, take the best per-query nDCG@k across the given single-level
    arms. Also returns how often each level would be chosen -- the level-usage
    distribution behind the H4 headroom.
    """
    total = 0.0
    chosen: Counter = Counter()
    for qid in query_ids:
        best_score, best_arm = -1.0, None
        for arm, run in level_runs.items():
            score = ndcg_at_k(run[qid], qrels[qid], k)
            if score > best_score:
                best_score, best_arm = score, arm
        total += best_score
        chosen[best_arm] += 1
    return total / len(query_ids), chosen


def _level_map(built: BuiltVectors) -> Dict[str, LevelIndex]:
    levels = {
        "chunk": built.chunk,
        "centroid-section": built.centroid_section,
        "centroid-doc": built.centroid_doc,
        "rawspan-section": built.rawspan_section,
        "rawspan-doc": built.rawspan_doc,
    }
    if built.summary_section is not None and built.summary_doc is not None:
        levels["summary-section"] = built.summary_section
        levels["summary-doc"] = built.summary_doc
    return levels


def _score_target(
    built: BuiltVectors, qids: List[str], qrels: Dict[str, Dict[str, int]], target: str
) -> Dict[str, object]:
    """Score every arm supporting ``target`` (doc/section): singles, fusion, oracle.

    Arms without a mapping for ``target`` (the doc arms at the section target) are
    absent, and each fusion/oracle family uses whichever of its levels are
    present. Every arm is scored against the same ``qrels`` for this target, so a
    hit is credited to its target unit before scoring.
    """
    single = {}
    for name, level in _level_map(built).items():
        run = rank_units(built.query_vecs, qids, level, target)
        if run is not None:
            single[name] = run
    single_ids = {name: ids_only(run) for name, run in single.items()}
    all_runs: Dict[str, Dict[str, List[str]]] = dict(single_ids)

    families = {
        "centroid": ["chunk", "centroid-section", "centroid-doc"],
        "rawspan": ["chunk", "rawspan-section", "rawspan-doc"],
        "summary": ["chunk", "summary-section", "summary-doc"],
    }
    oracle: Dict[str, object] = {}
    for fam, arms in families.items():
        present = [a for a in arms if a in single]
        if len(present) < 2:  # need chunk + at least one coarse level to be meaningful
            continue
        all_runs[f"fusion-{fam}"] = fuse([single[a] for a in present], qids)
        score, usage = oracle_over({a: single_ids[a] for a in present}, qids, qrels, PRIMARY_K)
        oracle[fam] = {"ndcg@10": score, "level_usage": dict(usage)}

    results = {name: evaluate(run, qrels, k_values=RECALL_K_VALUES) for name, run in all_runs.items()}
    return {"results": results, "oracle": oracle}


def run_experiment(bench: SyntheticBenchmark, built: BuiltVectors) -> Dict[str, object]:
    """Score every arm at both the document and section targets.

    Doc target: does the right super-document come back. Section target: does the
    right section come back -- the metric that isolates the hierarchy premise,
    especially under ``mode="section"`` where the answer *is* a section.
    """
    qids = built.query_ids
    return {
        "doc": _score_target(built, qids, bench.doc_qrels, "doc"),
        "section": _score_target(built, qids, bench.section_qrels, "section"),
    }


def format_table(results: Dict[str, Dict[str, float]]) -> str:
    metric_names = [f"recall@{k}" for k in RECALL_K_VALUES] + [PRIMARY_METRIC]
    width = max(len(label) for label in results) + 2
    header = f"| {'arm'.ljust(width)} | " + " | ".join(m.rjust(9) for m in metric_names) + " |"
    rule = f"|{'-' * (width + 2)}|" + "|".join("-" * 11 for _ in metric_names) + "|"
    rows = []
    for label, scores in sorted(results.items(), key=lambda kv: -kv[1][PRIMARY_METRIC]):
        cells = " | ".join(f"{scores[m]:9.4f}" for m in metric_names)
        rows.append(f"| {label.ljust(width)} | {cells} |")
    return "\n".join([header, rule, *rows])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hierarchical retrieval experiment (raw-span vs centroid, fusion, oracle).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default="nfcorpus", choices=("fiqa", "nfcorpus", "scifact", "qasper"))
    p.add_argument("--split", default="dev", help="qasper only: dev (~280 papers) or train (~880).")
    p.add_argument("--max-papers", type=int, default=None, help="qasper only: cap the number of papers.")
    p.add_argument("--provider", default=DEFAULT_PROVIDER, help="Embedding provider (default: openai).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model (default: text-embedding-3-small).")
    p.add_argument("--sections", type=int, default=3, help="Sections per super-document (S).")
    p.add_argument("--passages", type=int, default=3, help="Passages per section (P).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-queries", type=int, default=None, help="Cap placed queries (smaller run).")
    p.add_argument(
        "--mode",
        default="point",
        choices=("point", "section"),
        help="point: one gold passage per doc (favours chunk). section: cluster a query's golds "
        "into one gold-dense section (the fair test of the hierarchy premise).",
    )
    p.add_argument("--min-section-gold", type=int, default=2, help="section mode: min in-corpus golds per query.")
    p.add_argument(
        "--summary",
        action="store_true",
        help="Add the directed-summary arm (H2): LLM-summarise each span, then embed. Costs LLM tokens.",
    )
    p.add_argument("--summary-model", default="gpt-5.4-nano", help="OpenAI chat model for --summary.")
    p.add_argument("--directive", default="retrieval", help="Summary directive key (benchmarks/summarize.py).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    if args.provider == "mock":
        print("Refusing to run on mock embeddings: they cannot measure relevance.", file=sys.stderr)
        return 1

    if args.dataset == "qasper":
        from benchmarks.qasper_data import load_qasper

        bench = load_qasper(split=args.split, max_papers=args.max_papers, seed=args.seed)
    else:
        source = beir_data.load(args.dataset)
        logger.info("Source: %r", source)
        bench = build_synthetic_benchmark(
            source,
            sections_per_doc=args.sections,
            passages_per_section=args.passages,
            seed=args.seed,
            max_queries=args.max_queries,
            mode=args.mode,
            min_section_gold=args.min_section_gold,
        )
    logger.info("%r", bench)

    summarizer = None
    if args.summary:
        from benchmarks.summarize import CachedSummarizer

        summarizer = CachedSummarizer(args.summary_model, directive=args.directive)
        logger.info("Directed-summary arm on: model=%s directive=%s", args.summary_model, args.directive)

    encoder = CachedEncoder(args.provider, args.model)
    built = build_vectors(bench, encoder, summarizer)
    logger.info("Encoded: %d new, %d cached (dim=%s)", encoder.n_embedded, encoder.n_cached, encoder.dim)
    if summarizer is not None:
        logger.info("Summarised: %d new, %d cached", summarizer.n_called, summarizer.n_cached)

    report = run_experiment(bench, built)

    for target in ("doc", "section"):
        block = report[target]  # type: ignore[index]
        results = block["results"]
        oracle = block["oracle"]
        chunk_ndcg = results["chunk"][PRIMARY_METRIC]
        print(f"\n===== target: {target.upper()} (qrels = {target}) =====")
        print(format_table(results))
        print()
        for family in ("rawspan", "centroid", "summary"):
            if family in oracle:
                entry = oracle[family]
                gap = entry["ndcg@10"] - chunk_ndcg
                print(
                    f"Oracle ({family:<8} levels) {PRIMARY_METRIC}={entry['ndcg@10']:.4f}  "
                    f"(vs chunk {gap:+.4f})  usage={entry['level_usage']}"
                )

    cost: Dict[str, object] = {"embedded": encoder.n_embedded, "cached": encoder.n_cached, "dim": encoder.dim}
    if summarizer is not None:
        cost["summaries_called"] = summarizer.n_called
        cost["summaries_cached"] = summarizer.n_cached
        cost["summary_model"] = args.summary_model
        cost["directive"] = args.directive
    full = {
        "schema": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "embedding": {"provider": args.provider, "model": args.model},
        "synthetic": bench.params | {"super_docs": len(bench.corpus), "queries": len(bench.queries)},
        "primary_metric": PRIMARY_METRIC,
        "cost": cost,
        "results_by_target": report,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    if args.dataset == "qasper":
        tag = f"qasper_{args.split}"
    else:
        tag = f"{args.dataset}_s{args.sections}p{args.passages}_{args.mode}"
    out = RESULTS_DIR / f"hierarchical_{tag}_{stamp}.json"
    out.write_text(json.dumps(full, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
