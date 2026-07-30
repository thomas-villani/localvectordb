"""Phase 0 harness for the dual-embedding (concatenated two-model) experiment.

    ./.venv/Scripts/python.exe benchmarks/eval_dual.py embed --model-key nomic
    ./.venv/Scripts/python.exe benchmarks/eval_dual.py analyze --models openai,nomic,egemma,arctic,qwen3
    ./.venv/Scripts/python.exe benchmarks/eval_dual.py p2
    ./.venv/Scripts/python.exe benchmarks/eval_dual.py p2 --dataset maud --max-papers 50
    ./.venv/Scripts/python.exe benchmarks/eval_dual.py align

Plan: ``experiments/dual-embedding-plan.md`` (gitignored) §5. Phase 0 asks three
questions, all answerable from cached full-dim vectors without touching ``src/``:

P0-A  Complementarity / oracle headroom. Score every query with each model's
      chunks-alone and rawspan-sections-alone arms; the oracle per-query pick
      over any two arms is the ceiling of ANY fusion of that pair. If the best
      cross-model (chunk_A + section_B) oracle is no better than the best
      same-model one, dual-model concat has no headroom.
P0-B  Fusion-rule ablation (same model, full dim). Concatenation implements
      RAW-LINEAR score addition at the chunk level; we ship MIN-MAX
      pool-relative fusion. Measure the rule change in isolation before
      attributing anything to the two-model part. A global affine (z-score)
      calibration only rescales the effective alpha -- it shifts every fused
      score by a per-query constant -- so the raw-linear alpha sweep already
      upper-bounds any index-absorbable calibration; the z-linear curve is
      reported to show where the natural (alpha=0.5) operating point lands.
P0-C  MRL truncation ladder. Slice cached full-dim vectors (prefix + renorm)
      at each model's Matryoshka dims. If sections at d/2 lose their edge, the
      equal-budget concat story collapses.

Gate 1 (2026-07-24) killed concat and promoted the dual-model thread; the
``p2`` subcommand is the re-scoped Phase 2' Qasper leg (findings doc §6):
cross-model UNIT-LEVEL fusion of a chunk leg from one model and a section leg
from another (two indexes, two providers -- the shape LocalVectorDB already
ships), with rule refinements (min-max / z / RRF / max) attacking the
realized-vs-oracle gap and @budget-dim MRL arms for an honest matched-budget
comparison. Zero new embedding: every vector is already in the disk cache.

``align`` is the Tier 0 model-selection study that follows P2': pairwise
representational geometry (linear/RBF CKA, SVCCA, directional ridge residual
energy) over document vectors ALONE, bench-marked against realized min-max
fusion gain for every ordered model pair. It asks one question -- can a
LABEL-FREE statistic pick the pair, so choosing a second model on a new corpus
stops requiring an eval set? -- with a pre-registered kill criterion (§ the
"Tier 0" block below). Also zero new embedding.

Embedding goes through the same per-(model, text) disk cache as
``eval_hierarchical`` -- embed once, slice forever. Query/document prefixes are
applied PER WINDOW at embed time and land inside the cache key via the prefixed
text, so a prefixed and an unprefixed run can never share vectors (prefix
hygiene is a silent confound otherwise); an empty prefix reproduces the legacy
keys, which is what lets the openai arm reuse the original hierarchical-study
vectors. ``experiments/.env`` is loaded for OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from benchmarks.config import RESULTS_DIR  # noqa: E402
from benchmarks.eval_hierarchical import (  # noqa: E402
    CachedEncoder,
    _chunker,
    _detect_sections,
    _estimate_tokens,
    _unit,
)
from benchmarks.metrics import ndcg_at_k, recall_at_k  # noqa: E402

logger = logging.getLogger("benchmarks.eval_dual")

PRIMARY_K = 10
TARGETS = ("doc", "section")
# Weight grid shared by every fusion rule (0.05 steps; includes the shipped 0.65).
ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 2)
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0
RRF_K = 60  # matches the shipped T1.1 RRF constant
# P0-A top cross-model pairs + a swapped-legs control (chunkmodel:sectionmodel).
P2_DEFAULT_PAIRS = "egemma:openai,egemma:qwen3,egemma:arctic,qwen3:egemma"


def _load_experiments_env() -> None:
    """Load KEY=VALUE lines from experiments/.env (never overriding real env)."""
    env_file = _ROOT / "experiments" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


# ---------------------------------------------------------------------------
# Model pool (plan §8). Constraint: genuinely MRL-trained (bge-m3 is NOT, which
# excludes the local quality leader -- recorded as a known gap). Prefixes follow
# each model's official retrieval usage; getting these wrong silently costs
# quality and would masquerade as a "model quality" difference.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    model: str
    dim: int
    # Truncation ladder, full dim first. Official MRL training dims where known;
    # arctic v2.0 officially targets 256 (512/128 are off-ladder but informative).
    mrl_dims: Tuple[int, ...]
    num_ctx: Optional[int] = None
    timeout: Optional[int] = None
    query_prefix: str = ""
    doc_prefix: str = ""
    note: str = ""


MODEL_POOL: Dict[str, ModelSpec] = {
    "openai": ModelSpec(
        key="openai",
        provider="openai",
        model="text-embedding-3-small",
        dim=1536,
        mrl_dims=(1536, 768, 512, 256, 128),
        note="8k ctx; MRL via `dimensions`; continuity reference (cached study vectors)",
    ),
    "nomic": ModelSpec(
        key="nomic",
        provider="ollama",
        model="nomic-embed-text",
        dim=768,
        mrl_dims=(768, 512, 256, 128),
        num_ctx=2048,
        timeout=600,
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
        note="v1.5; 2k arch cap; short/chunk leg",
    ),
    "egemma": ModelSpec(
        key="egemma",
        provider="ollama",
        model="embeddinggemma:300m",
        dim=768,
        mrl_dims=(768, 512, 256, 128),
        num_ctx=2048,
        timeout=600,
        query_prefix="task: search result | query: ",
        doc_prefix="title: none | text: ",
        note="2k arch cap; short/chunk leg",
    ),
    "arctic": ModelSpec(
        key="arctic",
        provider="ollama",
        model="snowflake-arctic-embed2",
        dim=768,
        mrl_dims=(768, 512, 256, 128),
        num_ctx=8192,
        timeout=1800,
        query_prefix="query: ",
        doc_prefix="",
        note="m-v2.0; 8k ctx; long/section leg (official MRL dim: 256)",
    ),
    "qwen3": ModelSpec(
        key="qwen3",
        provider="ollama",
        model="qwen3-embedding:0.6b",
        dim=1024,
        mrl_dims=(1024, 512, 256, 128),
        num_ctx=8192,
        timeout=1800,
        query_prefix=("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "),
        doc_prefix="",
        note="32k-capable, run @8k (memory); long/section leg; instruction-aware queries",
    ),
}


class PrefixedEncoder(CachedEncoder):
    """CachedEncoder with an embed-time prefix and crash-safe incremental caching.

    The prefix is applied to EVERY window (not just the first), so a
    window-pooled long section is prefixed consistently. The parent embeds all
    missing texts before writing any cache file; a crash in a multi-hour local
    arm would lose everything, so this override saves per batch and logs
    progress/ETA.
    """

    def __init__(self, spec: ModelSpec, prefix: str) -> None:
        super().__init__(spec.provider, spec.model, num_ctx=spec.num_ctx, timeout=spec.timeout)
        self.prefix = prefix
        # Small request batches bound cache loss and keep progress observable on
        # CPU-bound Ollama encoders; OpenAI tolerates far bigger requests.
        if spec.provider == "openai":
            self.max_inputs, self.max_tokens = 512, 150_000
        else:
            self.max_inputs, self.max_tokens = 32, 25_000

    def _windows(self, text: str) -> List[str]:
        return [self.prefix + w for w in super()._windows(text)]

    def count_misses(self, texts: Sequence[str]) -> Tuple[int, int]:
        """(cache hits, misses) at window granularity, without embedding anything."""
        hits = misses = 0
        for t in texts:
            for w in self._windows(t):
                if self._path(w).exists():
                    hits += 1
                else:
                    misses += 1
        return hits, misses

    def _batches(self, texts: Sequence[str]) -> Iterator[List[int]]:
        batch: List[int] = []
        tokens = 0
        for i, t in enumerate(texts):
            cost = _estimate_tokens(t)
            if batch and (len(batch) >= self.max_inputs or tokens + cost > self.max_tokens):
                yield batch
                batch, tokens = [], 0
            batch.append(i)
            tokens += cost
        if batch:
            yield batch

    def _embed_raw(self, texts: List[str]) -> np.ndarray:
        vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        miss_idx: List[int] = []
        for i, text in enumerate(texts):
            path = self._path(text)
            if path.exists():
                vectors[i] = np.load(path)
                self.n_cached += 1
            else:
                miss_idx.append(i)

        if miss_idx:
            miss_texts = [texts[i] for i in miss_idx]
            total, done, t0 = len(miss_texts), 0, time.monotonic()
            for batch in self._batches(miss_texts):
                fresh = np.asarray(self.provider.embed_sync([miss_texts[j] for j in batch]), dtype=np.float32)
                for row, j in enumerate(batch):
                    i = miss_idx[j]
                    np.save(self._path(texts[i]), fresh[row])
                    vectors[i] = fresh[row]
                    self.n_embedded += 1
                done += len(batch)
                rate = done / max(time.monotonic() - t0, 1e-9)
                eta_min = (total - done) / max(rate, 1e-9) / 60
                logger.info(
                    "  %s: %d/%d windows embedded (%.2f/s, ETA %.0f min)", self.model, done, total, rate, eta_min
                )
        return np.vstack(vectors).astype(np.float32)


# ---------------------------------------------------------------------------
# Corpus units: chunks (500-token default, matching the DB), rawspan sections,
# queries. No doc-span or centroid arms in Phase 0 -- they are Phase 1 material
# and doc spans are the expensive ones to embed.
# ---------------------------------------------------------------------------


@dataclass
class CorpusUnits:
    chunk_texts: List[str]
    chunk_doc: List[str]
    chunk_section: List[str]
    section_texts: List[str]
    section_ids: List[str]
    section_doc: List[str]
    query_ids: List[str]
    query_texts: List[str]
    # True where a chunk's span crosses a section boundary, i.e. midpoint
    # attribution credited it to one section while part of it lives in another.
    # This is the §6.1 attribution artifact, made countable.
    chunk_crosses: List[bool] = field(default_factory=list)
    # (start, end) char offsets into the source doc, parallel to chunk_texts.
    # Needed to ask whether a cut lands inside an annotated gold span; keeping
    # them here means an analysis never has to re-chunk the corpus to find out.
    chunk_spans: List[Tuple[int, int]] = field(default_factory=list)
    # EVERY section a chunk's span overlaps, parallel to chunk_texts. Midpoint
    # attribution (`chunk_section`) picks exactly one of these, which is a
    # measurement convention, not a fact about the corpus: at 1000 tokens 87% of
    # chunks straddle, so the choice is close to arbitrary, and a section owning
    # no chunk becomes unreachable through chunk rows entirely (~800/3,974 on
    # Qasper, 1,494/4,709 on MAUD). Keeping the full list lets `overlap_units`
    # build the many-to-many alternative without re-chunking.
    chunk_sections_all: List[List[str]] = field(default_factory=list)


# Section detection is independent of chunk size, but a size sweep calls
# load_units once per size and pays it again every time (~4 min per pass on
# MAUD@50, single-threaded). Memoise on the doc text itself so the key cannot
# outlive the content it describes.
_SECTION_MEMO: Dict[Tuple[str, str, int], List[Any]] = {}


def _sections_for(detect, doc_id: str, text: str) -> List[Any]:
    key = (getattr(detect, "__qualname__", repr(detect)), doc_id, hash(text))
    hit = _SECTION_MEMO.get(key)
    if hit is None:
        hit = list(detect(text))
        _SECTION_MEMO[key] = hit
    return hit


def _build_chunker(chunk_tokens: Optional[int], method: Optional[str], overlap: Optional[int], **kwargs):
    """``_chunker`` plus method/overlap overrides, for the chunker-comparison arms."""
    if method is None and overlap is None and not kwargs:
        return _chunker(chunk_tokens)
    from benchmarks.eval_hierarchical import CHUNKING
    from localvectordb.chunking import ChunkerFactory

    return ChunkerFactory.create_chunker(
        method or CHUNKING["method"],
        chunk_tokens or CHUNKING["max_tokens"],
        CHUNKING["overlap"] if overlap is None else overlap,
        **kwargs,
    )


def load_units(
    bench,
    chunk_tokens: Optional[int],
    section_detector=None,
    chunk_method: Optional[str] = None,
    chunk_overlap: Optional[int] = None,
    chunk_kwargs: Optional[Dict[str, Any]] = None,
) -> CorpusUnits:
    """Chunk + section every doc exactly as ``eval_hierarchical.build_vectors`` does.

    ``section_detector`` overrides the markdown detector (MAUD contracts need
    ``maud_data.detect_contract_sections``); it must be the SAME detector the
    dataset's loader used to build its qrels, or section ids drift.

    ``chunk_method`` / ``chunk_overlap`` override the shipped ``CHUNKING`` config
    so one run can compare chunkers at matched size; leave both ``None`` for the
    published-baseline behaviour.
    """
    from benchmarks.superdocs import section_qrel_id

    chunker = _build_chunker(chunk_tokens, chunk_method, chunk_overlap, **(chunk_kwargs or {}))
    detect = section_detector or _detect_sections
    u = CorpusUnits([], [], [], [], [], [], [], [])
    for doc_id, text in bench.corpus.items():
        sections = _sections_for(detect, doc_id, text)
        for sec in sections:
            if sec.heading is None:
                continue
            u.section_texts.append(text[sec.start_pos : sec.end_pos])
            u.section_ids.append(section_qrel_id(doc_id, sec.index))
            u.section_doc.append(doc_id)
        for ch in chunker.chunk(text):
            u.chunk_texts.append(ch.content)
            u.chunk_doc.append(doc_id)
            mid = (ch.position.start + ch.position.end) // 2
            owner = next((s for s in sections if s.start_pos <= mid < s.end_pos), sections[-1])
            u.chunk_section.append(section_qrel_id(doc_id, owner.index))
            # end is exclusive, so a chunk ending exactly on a boundary is clean.
            u.chunk_crosses.append(ch.position.start < owner.start_pos or ch.position.end > owner.end_pos)
            u.chunk_spans.append((ch.position.start, ch.position.end))
            # Same section pool as midpoint above -- deliberately NOT filtered to
            # `heading is not None`, so the two assignments differ in exactly one
            # respect (one owner vs all of them) and nothing else. Headingless
            # sections are not real units and can never be gold, so including
            # them only ever adds distractors to the overlap arm: conservative.
            u.chunk_sections_all.append(
                [
                    section_qrel_id(doc_id, s.index)
                    for s in sections
                    if s.start_pos < ch.position.end and s.end_pos > ch.position.start
                ]
                or [section_qrel_id(doc_id, owner.index)]
            )
    u.query_ids.extend(bench.queries)
    u.query_texts.extend(bench.queries[q] for q in u.query_ids)
    return u


@dataclass
class ModelVectors:
    """Raw (un-normalised) full-dim vectors; normalise after any MRL slice."""

    chunks: np.ndarray
    sections: np.ndarray
    queries: np.ndarray


def embed_model(
    spec: ModelSpec, units: CorpusUnits, dry_run: bool = False
) -> Tuple[Optional[ModelVectors], Dict[str, object]]:
    doc_enc = PrefixedEncoder(spec, spec.doc_prefix)
    qry_enc = PrefixedEncoder(spec, spec.query_prefix)

    if dry_run:
        stats: Dict[str, object] = {}
        for name, enc, texts in (
            ("chunks", doc_enc, units.chunk_texts),
            ("sections", doc_enc, units.section_texts),
            ("queries", qry_enc, units.query_texts),
        ):
            hits, misses = enc.count_misses(texts)
            stats[name] = {"hits": hits, "misses": misses}
        return None, stats

    logger.info("[%s] chunks: %d texts ...", spec.key, len(units.chunk_texts))
    chunks = doc_enc.encode(units.chunk_texts, normalize=False)
    logger.info("[%s] sections: %d texts ...", spec.key, len(units.section_texts))
    sections = doc_enc.encode(units.section_texts, normalize=False)
    logger.info("[%s] queries: %d texts ...", spec.key, len(units.query_texts))
    queries = qry_enc.encode(units.query_texts, normalize=False)
    if chunks.shape[1] != spec.dim:
        logger.warning("[%s] dim %d != expected %d", spec.key, chunks.shape[1], spec.dim)
    embed_stats = {
        "embedded": doc_enc.n_embedded + qry_enc.n_embedded,
        "cached": doc_enc.n_cached + qry_enc.n_cached,
        "pooled": doc_enc.n_pooled + qry_enc.n_pooled,
        "dim": int(chunks.shape[1]),
    }
    logger.info("[%s] done: %s", spec.key, embed_stats)
    return ModelVectors(chunks, sections, queries), embed_stats


# ---------------------------------------------------------------------------
# Scoring machinery: max-pool vector scores to target units, per-query metrics,
# paired bootstrap, rank correlation.
# ---------------------------------------------------------------------------

# Per-contract retrieval scoping (MAUD): qid -> scope key, set once in main().
# None (Qasper) = global search. When set, score_arm ranks each query only
# against units whose id prefix matches its scope, and the fusion rules
# normalize within-scope (out-of-scope entries become NaN before min-max/z/RRF,
# so per-query calibration is over the contract's pool, mirroring how a
# metadata-filtered search would behave in the shipped system).
SCOPE_QID: Optional[Dict[str, str]] = None


def _unit_scope(u: str) -> str:
    return u.rsplit("#s", 1)[0]


def _scope_mask(qids: Sequence[str], units: Sequence[str]) -> np.ndarray:
    """Bool (nq, U): True where the unit is in the query's scope."""
    assert SCOPE_QID is not None
    codes: Dict[str, int] = {}
    uc = np.fromiter((codes.setdefault(_unit_scope(u), len(codes)) for u in units), np.int64, count=len(units))
    qc = np.fromiter((codes.get(SCOPE_QID.get(q, ""), -1) for q in qids), np.int64, count=len(qids))
    return qc[:, None] == uc[None, :]


def _scope_nan(mats: Sequence[np.ndarray], qids: Sequence[str], units: Sequence[str]) -> Tuple[np.ndarray, ...]:
    """Out-of-scope entries -> NaN so row-wise normalization is within-scope."""
    if SCOPE_QID is None:
        return tuple(mats)
    keep = _scope_mask(qids, units)
    return tuple(np.where(keep, m, np.nan) for m in mats)


class Pooler:
    """Pool per-vector similarity rows to per-unit scores via reduceat.

    ``mode="mean"`` exists to test claims that are specifically ABOUT max-pooling:
    "two sub-chunks give two shots at a high score" only buys anything under max,
    so a mechanism resting on it must weaken under mean. Default stays ``max`` --
    every published number in this work is max-pooled.
    """

    def __init__(
        self,
        unit_ids: Sequence[str],
        mode: str = "max",
        cols: Optional[Sequence[int]] = None,
    ) -> None:
        """``unit_ids[i]`` is a unit that source column ``cols[i]`` contributes to.

        ``cols=None`` is the one-to-one case: ``unit_ids`` runs parallel to the
        similarity matrix's columns, i.e. midpoint attribution, and every
        published number in this work uses it.

        Passing an explicit ``(cols, unit_ids)`` pair lets ONE column contribute
        to SEVERAL units -- overlap assignment, where a straddling chunk counts
        toward every section it touches instead of only the one holding its
        midpoint. The pair is generated by :func:`overlap_units`.
        """
        if mode not in ("max", "mean"):
            raise ValueError(f"unknown pooling mode {mode!r}")
        if cols is not None and len(cols) != len(unit_ids):
            raise ValueError(f"cols/unit_ids length mismatch: {len(cols)} vs {len(unit_ids)}")
        self.mode = mode
        self.units = sorted(set(unit_ids))
        idx = {unit: i for i, unit in enumerate(self.units)}
        col = np.fromiter((idx[unit] for unit in unit_ids), dtype=np.int64, count=len(unit_ids))
        self.order = np.argsort(col, kind="stable")
        src = np.arange(len(unit_ids), dtype=np.int64) if cols is None else np.asarray(cols, dtype=np.int64)
        # Which similarity column each sorted entry reads from. Under one-to-one
        # this is just `order`; under overlap a column repeats once per unit.
        self.src = src[self.order]
        sorted_col = col[self.order]
        self.starts = np.flatnonzero(np.r_[True, np.diff(sorted_col) > 0])
        self.counts = np.diff(np.r_[self.starts, len(unit_ids)]).astype(np.float32)

    def pool(self, sims: np.ndarray) -> np.ndarray:
        ordered = sims[:, self.src]
        if self.mode == "mean":
            return np.add.reduceat(ordered, self.starts, axis=1) / self.counts
        return np.maximum.reduceat(ordered, self.starts, axis=1)


def overlap_units(units: CorpusUnits) -> Tuple[np.ndarray, List[str]]:
    """Expand ``chunk_sections_all`` into the ``(cols, unit_ids)`` pair Pooler wants.

    Returns one entry per (chunk, overlapped section) pair, so a chunk crossing
    three sections appears three times. Feeding these to ``Pooler`` scores every
    section by the best (or mean) chunk that *touches* it rather than the best
    chunk that happens to be centred in it.
    """
    cols: List[int] = []
    ids: List[str] = []
    for i, sections in enumerate(units.chunk_sections_all):
        for sid in sections:
            cols.append(i)
            ids.append(sid)
    return np.asarray(cols, dtype=np.int64), ids


def make_pooler(units: CorpusUnits, mode: str = "max", assign: str = "midpoint") -> Pooler:
    """Pooler over ``assign`` in {midpoint, overlap}."""
    if assign == "midpoint":
        return Pooler(units.chunk_section, mode=mode)
    if assign == "overlap":
        cols, ids = overlap_units(units)
        return Pooler(ids, mode=mode, cols=cols)
    raise ValueError(f"unknown assignment {assign!r}")


def score_arm(
    units_sorted: Sequence[str],
    pooled: np.ndarray,
    qids: Sequence[str],
    qrels: Dict[str, Dict[str, int]],
    k: int = PRIMARY_K,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-query (nDCG@k, recall@k) for a (nq, U) unit-score matrix."""
    if SCOPE_QID is not None:
        pooled = np.where(_scope_mask(qids, units_sorted), pooled, -np.inf)
    order = np.argsort(-pooled, axis=1)[:, :k]
    pq = np.zeros(len(qids), dtype=np.float64)
    rec = np.zeros(len(qids), dtype=np.float64)
    for qi, qid in enumerate(qids):
        ranked = [units_sorted[j] for j in order[qi]]
        pq[qi] = ndcg_at_k(ranked, qrels[qid], k)
        rec[qi] = recall_at_k(ranked, qrels[qid], k)
    return pq, rec


@dataclass
class ArmScores:
    units: List[str]
    mat: np.ndarray  # (nq, U) float32 pooled unit scores
    pq_ndcg: np.ndarray  # (nq,)
    mean_ndcg: float
    recall10: float


def paired_bootstrap(pq_a: np.ndarray, pq_b: np.ndarray, n: int = BOOTSTRAP_N) -> Dict[str, float]:
    """Paired bootstrap over queries for mean(pq_a) - mean(pq_b)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_q = len(pq_a)
    idx = rng.integers(0, n_q, size=(n, n_q))
    deltas = pq_a[idx].mean(axis=1) - pq_b[idx].mean(axis=1)
    return {
        "delta": float(pq_a.mean() - pq_b.mean()),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "p_win": float((deltas > 0).mean()),
    }


def _rankdata_rows(m: np.ndarray) -> np.ndarray:
    order = np.argsort(m, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(m.shape[0])[:, None]
    ranks[rows, order] = np.arange(m.shape[1])[None, :]
    return ranks.astype(np.float64)


def mean_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over rows of the Spearman correlation between aligned score rows."""
    ra, rb = _rankdata_rows(a), _rankdata_rows(b)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    den[den == 0] = 1.0
    return float(np.mean(num / den))


def _align(
    units_a: List[str], mat_a: np.ndarray, units_b: List[str], mat_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    common = sorted(set(units_a) & set(units_b))
    ia = {u: i for i, u in enumerate(units_a)}
    ib = {u: i for i, u in enumerate(units_b)}
    sel_a = np.fromiter((ia[u] for u in common), dtype=np.int64, count=len(common))
    sel_b = np.fromiter((ib[u] for u in common), dtype=np.int64, count=len(common))
    return mat_a[:, sel_a], mat_b[:, sel_b]


def _union_fill(
    units_a: List[str], mat_a: np.ndarray, units_b: List[str], mat_b: np.ndarray
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Expand both matrices onto the union unit universe, NaN where absent."""
    union = sorted(set(units_a) | set(units_b))

    def expand(units: List[str], mat: np.ndarray) -> np.ndarray:
        idx = {u: i for i, u in enumerate(units)}
        out = np.full((mat.shape[0], len(union)), np.nan, dtype=np.float32)
        cols = [j for j, u in enumerate(union) if u in idx]
        out[:, cols] = mat[:, [idx[union[j]] for j in cols]]
        return out

    return union, expand(units_a, mat_a), expand(units_b, mat_b)


def _fill_rowmin(m: np.ndarray) -> np.ndarray:
    """NaN -> that row's minimum (a missing leg contributes its floor score)."""
    return np.where(np.isnan(m), np.nanmin(m, axis=1, keepdims=True), m)


def _minmax_rows(m: np.ndarray) -> np.ndarray:
    """Per-row min-max to [0, 1] (mirrors the shipped pool-relative fusion); NaN -> 0."""
    lo = np.nanmin(m, axis=1, keepdims=True)
    hi = np.nanmax(m, axis=1, keepdims=True)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return np.nan_to_num((m - lo) / span, nan=0.0)


# ---------------------------------------------------------------------------
# P0 analyses
# ---------------------------------------------------------------------------


def build_singles(
    vec: ModelVectors,
    poolers: Dict[Tuple[str, str], Pooler],
    qrels_by_target: Dict[str, Dict[str, Dict[str, int]]],
    qids: List[str],
) -> Dict[Tuple[str, str], ArmScores]:
    """Full-dim chunks-alone and sections-alone arms at both targets."""
    q = _unit(vec.queries)
    out: Dict[Tuple[str, str], ArmScores] = {}
    for level, raw in (("chunk", vec.chunks), ("section", vec.sections)):
        sims = (q @ _unit(raw).T).astype(np.float32)
        for target in TARGETS:
            pooler = poolers[(level, target)]
            pooled = pooler.pool(sims)
            pq, rec = score_arm(pooler.units, pooled, qids, qrels_by_target[target])
            out[(level, target)] = ArmScores(pooler.units, pooled, pq, float(pq.mean()), float(rec.mean()))
    return out


def run_p0b(
    vec: ModelVectors,
    units: CorpusUnits,
    singles_m: Dict[Tuple[str, str], ArmScores],
    qrels_by_target: Dict[str, Dict[str, Dict[str, int]]],
    qids: List[str],
) -> Dict[str, object]:
    """Fusion-rule ablation on the same model's chunk+section legs.

    raw_linear / z_linear emulate concatenation exactly: every CHUNK is scored
    ``alpha * cos(q, chunk) + (1 - alpha) * cos(q, its_section)`` and then
    max-pooled to the target unit -- one vector per chunk, one query.
    minmax_unit / raw_unit fuse the two UNIT-level runs (the shipped shape);
    their weight is the SECTION weight (shipped default 0.65).
    """
    sec_row = {u: i for i, u in enumerate(units.section_ids)}
    known = [i for i, s in enumerate(units.chunk_section) if s in sec_row]
    if len(known) != len(units.chunk_section):
        logger.warning(
            "P0-B: %d chunks have no section row; dropped from concat arms", len(units.chunk_section) - len(known)
        )
    owner = np.fromiter((sec_row[units.chunk_section[i]] for i in known), dtype=np.int64, count=len(known))

    q = _unit(vec.queries)
    chunk_sims = (q @ _unit(vec.chunks).T).astype(np.float32)[:, known]
    section_sims = (q @ _unit(vec.sections).T).astype(np.float32)
    owner_sims = section_sims[:, owner]
    z = lambda m: (m - float(m.mean())) / (float(m.std()) + 1e-9)  # noqa: E731
    chunk_poolers = {
        "doc": Pooler([units.chunk_doc[i] for i in known]),
        "section": Pooler([units.chunk_section[i] for i in known]),
    }

    out: Dict[str, object] = {}
    for target in TARGETS:
        qrels = qrels_by_target[target]
        entry: Dict[str, object] = {}
        pq_store: Dict[Tuple[str, int], np.ndarray] = {}

        chunk_level_rules = (
            ("raw_linear", chunk_sims, owner_sims),
            ("z_linear", z(chunk_sims), z(owner_sims)),
        )
        pooler = chunk_poolers[target]
        for rule, ma, mb in chunk_level_rules:
            means = []
            for ai, alpha in enumerate(ALPHAS):
                fused = alpha * ma + (1.0 - alpha) * mb  # alpha = CHUNK-block weight
                pq, _ = score_arm(pooler.units, pooler.pool(fused), qids, qrels)
                pq_store[(rule, ai)] = pq
                means.append(float(pq.mean()))
            best = int(np.argmax(means))
            entry[rule] = {
                "weight_is": "chunk",
                "weights": ALPHAS.tolist(),
                "ndcg": means,
                "best_weight": float(ALPHAS[best]),
                "best_ndcg": means[best],
                "_best_idx": best,
            }

        cs, ss = singles_m[("chunk", target)], singles_m[("section", target)]
        union, mc, ms = _union_fill(cs.units, cs.mat, ss.units, ss.mat)
        unit_level_rules = (
            ("minmax_unit", _minmax_rows(mc), _minmax_rows(ms)),
            ("raw_unit", _fill_rowmin(mc), _fill_rowmin(ms)),
        )
        for rule, uc, us in unit_level_rules:
            means = []
            for wi, w in enumerate(ALPHAS):
                fused = (1.0 - w) * uc + w * us  # w = SECTION weight (shipped semantics)
                pq, _ = score_arm(union, fused, qids, qrels)
                pq_store[(rule, wi)] = pq
                means.append(float(pq.mean()))
            best = int(np.argmax(means))
            entry[rule] = {
                "weight_is": "section",
                "weights": ALPHAS.tolist(),
                "ndcg": means,
                "best_weight": float(ALPHAS[best]),
                "best_ndcg": means[best],
                "_best_idx": best,
            }

        raw_best = pq_store[("raw_linear", entry["raw_linear"]["_best_idx"])]  # type: ignore[index]
        mm_best = pq_store[("minmax_unit", entry["minmax_unit"]["_best_idx"])]  # type: ignore[index]
        alpha_half = int(np.argmin(np.abs(ALPHAS - 0.5)))
        w_shipped = int(np.argmin(np.abs(ALPHAS - 0.65)))
        entry["references"] = {
            "chunk_alone": cs.mean_ndcg,
            "section_alone": ss.mean_ndcg,
            "raw_linear_at_0.5": float(pq_store[("raw_linear", alpha_half)].mean()),
            "minmax_at_0.65": float(pq_store[("minmax_unit", w_shipped)].mean()),
        }
        entry["bootstrap"] = {
            "raw_best_vs_minmax_best": paired_bootstrap(raw_best, mm_best),
            "raw_best_vs_section_alone": paired_bootstrap(raw_best, ss.pq_ndcg),
            "raw_at_0.5_vs_section_alone": paired_bootstrap(pq_store[("raw_linear", alpha_half)], ss.pq_ndcg),
        }
        for rule in ("raw_linear", "z_linear", "minmax_unit", "raw_unit"):
            entry[rule].pop("_best_idx")  # type: ignore[union-attr]
        out[target] = entry
    return out


def run_p0c(
    spec: ModelSpec,
    vec: ModelVectors,
    poolers: Dict[Tuple[str, str], Pooler],
    qrels_by_target: Dict[str, Dict[str, Dict[str, int]]],
    qids: List[str],
) -> Dict[str, object]:
    """MRL truncation ladder: slice cached full-dim vectors, renormalise, rescore."""
    dims = [d for d in spec.mrl_dims if d <= vec.chunks.shape[1]]
    out: Dict[str, object] = {"dims": dims}
    full_pq: Dict[Tuple[str, str], np.ndarray] = {}
    boots: Dict[str, Dict[str, object]] = {t: {} for t in TARGETS}
    for dim in dims:
        q = _unit(vec.queries[:, :dim])
        for level, raw in (("chunk", vec.chunks), ("section", vec.sections)):
            sims = (q @ _unit(raw[:, :dim]).T).astype(np.float32)
            for target in TARGETS:
                pooler = poolers[(level, target)]
                pq, _ = score_arm(pooler.units, pooler.pool(sims), qids, qrels_by_target[target])
                out.setdefault(target, {}).setdefault(level, []).append(float(pq.mean()))  # type: ignore[union-attr]
                if dim == dims[0]:
                    full_pq[(level, target)] = pq
                elif level == "section":
                    boots[target][str(dim)] = paired_bootstrap(pq, full_pq[(level, target)])
    out["boot_section_vs_full"] = boots
    return out


def run_p0a(
    all_singles: Dict[str, Dict[Tuple[str, str], ArmScores]],
    qids: List[str],
) -> Dict[str, object]:
    """Cross-arm complementarity: correlations + pair/pool oracle ceilings."""
    out: Dict[str, object] = {}
    for target in TARGETS:
        arms = {
            f"{mkey}/{level}": singles[(level, target)]
            for mkey, singles in all_singles.items()
            for level in ("chunk", "section")
        }
        names = sorted(arms)
        pq_mat = np.vstack([arms[n].pq_ndcg for n in names])

        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(pq_mat)
        ndcg_corr = {}
        rank_corr = {}
        pairs = {}
        for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
            ndcg_corr[f"{a}|{b}"] = float(corr[i, j])
            ma, mb = _align(arms[a].units, arms[a].mat, arms[b].units, arms[b].mat)
            rank_corr[f"{a}|{b}"] = mean_spearman(ma, mb)
            pa, pb = arms[a].pq_ndcg, arms[b].pq_ndcg
            oracle_pq = np.maximum(pa, pb)
            best_arm = a if pa.mean() >= pb.mean() else b
            pairs[f"{a}|{b}"] = {
                "oracle": float(oracle_pq.mean()),
                "best_single": float(arms[best_arm].pq_ndcg.mean()),
                "best_arm": best_arm,
                "gain": float(oracle_pq.mean() - arms[best_arm].pq_ndcg.mean()),
                "a_better": int((pa > pb).sum()),
                "b_better": int((pb > pa).sum()),
                "bootstrap": paired_bootstrap(oracle_pq, arms[best_arm].pq_ndcg),
            }

        pool_pq = pq_mat.max(axis=0)
        usage = Counter(names[i] for i in pq_mat.argmax(axis=0))
        out[target] = {
            "arms": {n: {"ndcg@10": arms[n].mean_ndcg, "recall@10": arms[n].recall10} for n in names},
            "ndcg_corr": ndcg_corr,
            "rank_corr": rank_corr,
            "pairs": pairs,
            "pool_oracle": {"ndcg@10": float(pool_pq.mean()), "usage": dict(usage)},
        }
    return out


# ---------------------------------------------------------------------------
# Phase 2' (Qasper leg): cross-model unit-level fusion from cached vectors.
# ---------------------------------------------------------------------------


def _z_rows(m: np.ndarray) -> np.ndarray:
    """Per-row z-score over present (non-NaN) entries; NaN -> the row's floor z.

    NaN entries are missing legs (union fill) or out-of-scope units (per-contract
    scoping) -- excluding them from the stats keeps the calibration over the
    query's actual pool. (Before 2026-07-24 this z-scored a row-min-filled
    matrix; ranking-equivalent per leg, slightly different blend calibration.)
    """
    mu = np.nanmean(m, axis=1, keepdims=True)
    sd = np.nanstd(m, axis=1, keepdims=True)
    z = (m - mu) / np.where(sd == 0, 1.0, sd)
    return np.where(np.isnan(z), np.nanmin(z, axis=1, keepdims=True), z).astype(np.float32)


def _rrf_rows(m: np.ndarray, k: int = RRF_K) -> np.ndarray:
    """Per-row reciprocal-rank transform 1/(k + rank), rank 1 = best; NaN ranks worst."""
    filled = np.where(np.isnan(m), -np.inf, m)
    order = np.argsort(-filled, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(m.shape[0])[:, None]
    ranks[rows, order] = np.arange(m.shape[1])[None, :]
    return (1.0 / (k + 1 + ranks)).astype(np.float32)


def _sweep(
    uc: np.ndarray,
    us: np.ndarray,
    union: List[str],
    qids: Sequence[str],
    qrels: Dict[str, Dict[str, int]],
) -> Tuple[List[float], int, List[np.ndarray]]:
    """Weighted linear fusion sweep; w = SECTION-leg weight (shipped semantics)."""
    means: List[float] = []
    pqs: List[np.ndarray] = []
    for w in ALPHAS:
        fused = (1.0 - w) * uc + w * us
        pq, _ = score_arm(union, fused, qids, qrels)
        means.append(float(pq.mean()))
        pqs.append(pq)
    return means, int(np.argmax(means)), pqs


def run_p2(
    pairs: List[Tuple[str, str]],
    singles_by_budget: Dict[str, Dict[str, Dict[Tuple[str, str], ArmScores]]],
    qids: List[str],
    qrels_by_target: Dict[str, Dict[str, Dict[str, int]]],
) -> Dict[str, object]:
    """Cross-model unit-level fusion (findings §6): realized capture of the P0-A oracle.

    For each (chunk-model, section-model) pair and each dim budget, fuse the two
    UNIT-level runs under four rules -- minmax (shipped), z-scored, RRF, and
    per-unit max -- and bootstrap the headline deltas against (a) the pair's
    best arm, (b) the best single arm at the same budget, and (c) the best
    same-model fused baseline (the shipped shape). ``capture_minmax`` is the
    fraction of the pair's oracle gain the tuned minmax fusion realizes.
    """
    w_ship = int(np.argmin(np.abs(ALPHAS - 0.65)))
    w_half = int(np.argmin(np.abs(ALPHAS - 0.5)))

    out: Dict[str, object] = {}
    for target in TARGETS:
        qrels = qrels_by_target[target]

        best_single: Dict[str, Tuple[str, ArmScores]] = {}
        for budget, singles in singles_by_budget.items():
            arms = {f"{m}/{lvl}": s[(lvl, target)] for m, s in singles.items() for lvl in ("chunk", "section")}
            name = max(arms, key=lambda n: arms[n].mean_ndcg)
            best_single[budget] = (name, arms[name])

        # Same-model minmax fusion (shipped shape) at full dim: the bar gate 2
        # says dual-model must clear, tuned per model so the bar is honest.
        same_fused: Dict[str, Dict[str, float]] = {}
        best_same: Optional[Tuple[str, float, np.ndarray]] = None
        for m, singles in singles_by_budget["full"].items():
            cs, ss = singles[("chunk", target)], singles[("section", target)]
            union, mc, ms = _union_fill(cs.units, cs.mat, ss.units, ss.mat)
            mc, ms = _scope_nan((mc, ms), qids, union)
            means, bi, pqs = _sweep(_minmax_rows(mc), _minmax_rows(ms), union, qids, qrels)
            same_fused[m] = {"best_weight": float(ALPHAS[bi]), "best_ndcg": means[bi], "at_0.65": means[w_ship]}
            if best_same is None or means[bi] > best_same[1]:
                best_same = (m, means[bi], pqs[bi])
        assert best_same is not None

        entry: Dict[str, object] = {
            "baselines": {
                "best_single": {b: {"arm": n, "ndcg": a.mean_ndcg} for b, (n, a) in best_single.items()},
                "same_model_fused_full": same_fused,
                "best_same_model_fused": {"model": best_same[0], "ndcg": best_same[1]},
            },
            "pairs": {},
        }

        for cm, sm in pairs:
            pname = f"{cm}/chunk+{sm}/section"
            per_budget: Dict[str, object] = {}
            for budget, singles in singles_by_budget.items():
                cs, ss = singles[cm][("chunk", target)], singles[sm][("section", target)]
                union, mc, ms = _union_fill(cs.units, cs.mat, ss.units, ss.mat)
                mc, ms = _scope_nan((mc, ms), qids, union)
                mmc, mms = _minmax_rows(mc), _minmax_rows(ms)

                rules: Dict[str, Dict[str, object]] = {}
                best_pq: Dict[str, np.ndarray] = {}
                mm_ship_pq: Optional[np.ndarray] = None
                weighted_rules = (
                    ("minmax", mmc, mms),
                    ("z", _z_rows(mc), _z_rows(ms)),
                    ("rrf", _rrf_rows(mc), _rrf_rows(ms)),
                )
                for rule, ua, ub in weighted_rules:
                    means, bi, pqs = _sweep(ua, ub, union, qids, qrels)
                    rules[rule] = {
                        "weights": ALPHAS.tolist(),
                        "ndcg": means,
                        "best_weight": float(ALPHAS[bi]),
                        "best_ndcg": means[bi],
                        "at_0.5": means[w_half],
                        "at_0.65": means[w_ship],
                    }
                    best_pq[rule] = pqs[bi]
                    if rule == "minmax":
                        mm_ship_pq = pqs[w_ship]
                assert mm_ship_pq is not None
                pq_max, _ = score_arm(union, np.maximum(mmc, mms), qids, qrels)
                rules["max"] = {"ndcg": float(pq_max.mean())}
                best_pq["max"] = pq_max

                oracle_pq = np.maximum(cs.pq_ndcg, ss.pq_ndcg)
                arm_best = cs if cs.mean_ndcg >= ss.mean_ndcg else ss
                best_rule = max(best_pq, key=lambda r: float(best_pq[r].mean()))
                mm_best_ndcg = float(best_pq["minmax"].mean())
                gain_oracle = float(oracle_pq.mean()) - arm_best.mean_ndcg

                boots = {
                    "minmax_best_vs_best_arm": paired_bootstrap(best_pq["minmax"], arm_best.pq_ndcg),
                    "minmax_best_vs_best_single_same_budget": paired_bootstrap(
                        best_pq["minmax"], best_single[budget][1].pq_ndcg
                    ),
                    "minmax_best_vs_best_same_model_fused": paired_bootstrap(best_pq["minmax"], best_same[2]),
                    "minmax_at_0.65_vs_best_single_same_budget": paired_bootstrap(
                        mm_ship_pq, best_single[budget][1].pq_ndcg
                    ),
                    "oracle_vs_minmax_best": paired_bootstrap(oracle_pq, best_pq["minmax"]),
                }
                if budget != "full":
                    boots["minmax_best_vs_best_single_fulldim"] = paired_bootstrap(
                        best_pq["minmax"], best_single["full"][1].pq_ndcg
                    )
                if best_rule != "minmax":
                    boots["best_rule_vs_minmax_best"] = paired_bootstrap(best_pq[best_rule], best_pq["minmax"])

                per_budget[budget] = {
                    "arms": {
                        "chunk": {"name": f"{cm}/chunk", "ndcg": cs.mean_ndcg},
                        "section": {"name": f"{sm}/section", "ndcg": ss.mean_ndcg},
                    },
                    "oracle": float(oracle_pq.mean()),
                    "rules": rules,
                    "best_rule": best_rule,
                    "capture_minmax": (
                        (mm_best_ndcg - arm_best.mean_ndcg) / gain_oracle if gain_oracle > 1e-9 else None
                    ),
                    "bootstrap": boots,
                }
            entry["pairs"][pname] = per_budget  # type: ignore[index]
        out[target] = entry
    return out


# ---------------------------------------------------------------------------
# Tier 0 -- representational geometry ("align"): can a LABEL-FREE statistic over
# document vectors alone predict which model pair fuses well?
#
# Motivation, and the reason to be sceptical up front: findings §8.2/1 showed
# that ORACLE HEADROOM -- a query-aware AND label-aware complementarity measure
# -- MISPREDICTED realized fusion gain (the pair with the largest oracle,
# egemma/chunk+openai/section at +0.0722, realized the least). A purely
# representational statistic sits strictly upstream of that, so it starts
# behind. The only reason to want one is that it needs no queries, no qrels and
# no eval run: "pick the second model for this corpus" becomes an unsupervised
# operation on the user's own documents.
#
# The bake-off is therefore pre-registered rather than exploratory: rank every
# candidate predictor against realized min-max fusion gain over the SAME pair
# set, with mean per-query Spearman -- label-requiring, already computed in
# P0-A -- as the bar to beat. A label-free predictor that does not come close
# to it has no selection value and the thread dies here.
#
# Predictors:
#   linear CKA   Symmetric, Gram-normalised, inverts nothing -- the only one
#                that is statistically safe at Qasper's N (3.1k chunks / 4.0k
#                sections against 768-1536 dims). Handicap: SYMMETRIC, so the
#                two orderings of a pair get an identical score while their
#                realized gains differ (qwen3:egemma vs egemma:qwen3 differed by
#                +0.008 nDCG). Reported over both scopes for that reason.
#   RBF CKA      Kernel variant on a subsample (median-heuristic bandwidth),
#                sensitive to non-linear structure linear CKA misses.
#   SVCCA        PCA-to-99%-variance, then subspace canonical correlations.
#                Reported WITH a degeneracy flag: CCA at d=1536 on N=3155 is
#                near rank-deficient and returns spuriously high correlations.
#                Cross-check only until a bigger corpus (CUAD) lands.
#   residual     DIRECTIONAL: held-out fraction of B's variance NOT linearly
#   energy       predictable from A (ridge, K-fold, PCA-truncated at several
#                ranks). Matches the asymmetric backbone+adjunct shape that
#                actually won, and its rank curve doubles as the sizing curve
#                for a Tier-1 residual-concat leg ("how many dims does B's
#                unique part need?"). Also evaluated on QUERIES using the
#                doc-fit map: models with asymmetric prefixes (qwen3, nomic,
#                egemma prefix queries differently from documents) need not
#                share one projection across both sides, and a residual leg is
#                only implementable if they do.
# ---------------------------------------------------------------------------

GEOM_LEVELS = ("chunk", "section")
RESID_DIMS = (64, 128, 256, 512)
# Ridge penalties as multiples of the mean Gram eigenvalue, so one grid is valid
# across models of different dimension and across corpora of different size.
RIDGE_LAM_MULTS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
SVCCA_VAR = 0.99


def _rank_avg(x: np.ndarray) -> np.ndarray:
    """Average ranks with ties shared (Spearman over a handful of model pairs)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = float(np.arange(i, j + 1).mean())
        i = j + 1
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return float("nan")
    rx, ry = _rank_avg(x[ok]), _rank_avg(y[ok])
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _spearman_boot(a: Sequence[float], b: Sequence[float], n: int = 1000) -> Dict[str, float]:
    """Bootstrap CI for a predictor's rank correlation, resampling PAIRS.

    With ~20 pairs this interval is wide by construction -- that width IS the
    finding (five models cannot settle a selection rule), so it is reported
    rather than hidden.
    """
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), size=len(x))
        v = _spearman(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"rho": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    arr = np.asarray(vals)
    return {
        "rho": _spearman(x, y),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
    }


def _partial_spearman(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> float:
    """First-order partial rank correlation of x with y, controlling for z.

    Load-bearing for the bake-off: on Qasper the most decorrelated model (nomic)
    is also the WORST one, so raw diversity-vs-gain correlations are confounded
    with arm quality. Partialling out best-arm nDCG asks whether diversity says
    anything ONCE quality is known -- which is the question a selection rule
    actually faces, since quality is the thing you would condition on first.
    """
    rxy, rxz, ryz = _spearman(x, y), _spearman(x, z), _spearman(y, z)
    den = np.sqrt(max(1.0 - rxz**2, 0.0) * max(1.0 - ryz**2, 0.0))
    return float((rxy - rxz * ryz) / den) if den > 1e-12 else float("nan")


def _partial_boot(x: Sequence[float], y: Sequence[float], z: Sequence[float], n: int = 1000) -> Dict[str, float]:
    ax, ay, az = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(ax), size=len(ax))
        v = _partial_spearman(ax[i], ay[i], az[i])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"rho": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    arr = np.asarray(vals)
    return {
        "rho": _partial_spearman(ax, ay, az),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
    }


def _centre(m: np.ndarray) -> np.ndarray:
    """Column-centre. Load-bearing for CKA: embedding spaces are strongly
    anisotropic, and uncentred CKA mostly measures the shared mean direction --
    it scores every pair ~0.9 and predicts nothing."""
    return (m - m.mean(axis=0, keepdims=True)).astype(np.float32)


def _gram_norm(xc: np.ndarray) -> float:
    return float(np.linalg.norm((xc.T @ xc).astype(np.float64)))


def linear_cka(xc: np.ndarray, yc: np.ndarray, xnorm: float, ynorm: float) -> float:
    """||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F) on column-centred features.

    Feature-space form, O(N p q): at Qasper's N this is both cheaper and better
    conditioned than the N x N Gram form. Self-norms are passed in because they
    are model-level constants reused across every pair.
    """
    xy = (xc.T @ yc).astype(np.float64)
    den = xnorm * ynorm
    return float((xy**2).sum() / den) if den > 0 else float("nan")


def _rbf_gram_centred(x: np.ndarray) -> np.ndarray:
    """Double-centred RBF kernel with the median-distance bandwidth heuristic.

    Distance-based, hence translation-invariant, so this takes UNIT (uncentred)
    vectors -- the double-centring below plays the role _centre plays for the
    linear form.
    """
    sq = np.maximum(2.0 - 2.0 * (x @ x.T), 0.0).astype(np.float32)
    med = float(np.median(sq[np.triu_indices_from(sq, k=1)]))
    k = np.exp(-sq / (2.0 * (med if med > 0 else 1.0)))
    k -= k.mean(axis=0, keepdims=True)
    k -= k.mean(axis=1, keepdims=True)
    return k


def kernel_cka(kc: np.ndarray, lc: np.ndarray) -> float:
    """CKA from pre-double-centred kernels: tr(Kc Lc)/sqrt(tr(Kc Kc) tr(Lc Lc))."""
    a, b = kc.astype(np.float64), lc.astype(np.float64)
    den = float(np.sqrt((a**2).sum() * (b**2).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def svcca_basis(xc: np.ndarray, var_frac: float = SVCCA_VAR) -> np.ndarray:
    """Orthonormal basis of the 99%-variance PCA subspace (left singular vectors).

    Computed once per (model, level) and reused across every pair -- the SVD is
    the most expensive and most memory-hungry step in the geometry pass.
    """
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    ev = (s.astype(np.float64)) ** 2
    total = float(ev.sum())
    keep = int(np.searchsorted(np.cumsum(ev) / total, var_frac) + 1) if total > 0 else 1
    return np.ascontiguousarray(u[:, :keep])


def svcca(qx: np.ndarray, qy: np.ndarray, n: int) -> Dict[str, object]:
    """Mean canonical correlation between two PCA subspaces given their bases.

    Using the left singular vectors as the basis whitens implicitly, so the
    canonical correlations are exactly the singular values of Qx^T Qy. The
    ``degenerate`` flag fires when the retained components approach N/10, where
    CCA reports correlations near 1.0 even for unrelated data.
    """
    kx, ky = qx.shape[1], qy.shape[1]
    sv = np.linalg.svd((qx.T @ qy).astype(np.float64), compute_uv=False)
    return {
        "svcca": float(sv.mean()),
        "mean_top20": float(sv[: min(20, len(sv))].mean()),
        "kx": kx,
        "ky": ky,
        "degenerate": bool(kx + ky > n / 10.0),
    }


def _fold_ids(n: int, folds: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.permutation(np.arange(n) % folds)


def residual_energy(
    docs: Dict[str, np.ndarray],
    queries: Dict[str, np.ndarray],
    dims: Sequence[int],
    folds: int,
) -> Dict[str, Dict[str, object]]:
    """Held-out residual energy 1 - R^2 for every ordered (source -> target) pair.

    Regress target model B's vectors on source model A's, ridge-penalised, in
    A's PCA basis truncated at each rank in ``dims``. Because the PCA basis
    diagonalises A's Gram, one eigendecomposition per (source, fold) yields
    every rank and every penalty for free.

    Two things make this the better-matched statistic for the residual-concat
    design: it is DIRECTIONAL (A->B is not B->A, matching backbone+adjunct), and
    the rank curve answers the sizing question directly -- if 80% of B is
    recoverable from A's top 128 components, B's unique contribution needs far
    fewer than 768 dims.

    ``residual_queries`` applies the DOC-fit map to query vectors. A residual leg
    is only implementable as a fixed projection if the same map works on both
    sides; models that prefix queries differently from documents (qwen3, nomic,
    egemma) are exactly where that can fail, and it fails silently.
    """
    models = sorted(docs)
    n = next(iter(docs.values())).shape[0]
    fid = _fold_ids(n, folds)
    lam_mults = np.asarray(RIDGE_LAM_MULTS, dtype=np.float64)

    ranks: Dict[str, List[int]] = {
        m: sorted({min(d, docs[m].shape[1]) for d in list(dims) + [docs[m].shape[1]]}) for m in models
    }
    # Ranks beyond the training fold's row count address null directions of the
    # Gram; ridge suppresses them safely, so the curve plateaus rather than
    # blowing up -- but that plateau is an artifact of N, not a property of the
    # models, and reads as "the residual saturates" if unflagged.
    n_train = n - n // folds
    if max(max(r) for r in ranks.values()) > n_train:
        logger.warning(
            "residual: requested ranks exceed the %d training rows per fold -- curve plateaus at rank(X)", n_train
        )
    sse: Dict[Tuple[str, str], np.ndarray] = {}
    denom: Dict[Tuple[str, str], float] = {}

    for src in models:
        ks = ranks[src]
        for f in range(folds):
            tr, te = fid != f, fid == f
            xtr = docs[src][tr].astype(np.float64)
            mx = xtr.mean(axis=0)
            xtr -= mx
            xte = docs[src][te].astype(np.float64) - mx
            w, v = np.linalg.eigh(xtr.T @ xtr)
            w = np.maximum(w[::-1].copy(), 0.0)
            v = v[:, ::-1].copy()
            lams = lam_mults * max(float(w.mean()), 1e-12)
            ztr, zte = xtr @ v, xte @ v
            for tgt in models:
                if tgt == src:
                    continue
                ytr = docs[tgt][tr].astype(np.float64)
                my = ytr.mean(axis=0)
                ytr -= my
                yte = docs[tgt][te].astype(np.float64) - my
                zty = ztr.T @ ytr
                key = (src, tgt)
                acc = sse.setdefault(key, np.zeros((len(ks), len(lams)), dtype=np.float64))
                denom[key] = denom.get(key, 0.0) + float((yte**2).sum())
                for ki, k in enumerate(ks):
                    for li, lam in enumerate(lams):
                        pred = zte[:, :k] @ (zty[:k] / (w[:k] + lam)[:, None])
                        acc[ki, li] += float(((yte - pred) ** 2).sum())
            del ztr, zte, xtr, xte, v
        logger.info("  residual: source %s done (%d folds)", src, folds)

    out: Dict[str, Dict[str, object]] = {}
    for (src, tgt), acc in sse.items():
        r2 = 1.0 - acc / denom[(src, tgt)]
        best = r2.argmax(axis=1)
        out[f"{src}->{tgt}"] = {
            "ranks": ranks[src],
            "residual": [float(1.0 - r2[i, best[i]]) for i in range(len(ranks[src]))],
            "r2": [float(r2[i, best[i]]) for i in range(len(ranks[src]))],
            "lam_mult": [float(lam_mults[best[i]]) for i in range(len(ranks[src]))],
        }

    # Query transfer: refit on ALL documents, apply the same fixed projection to
    # the query side. Queries are inherently held out (different texts).
    for src in models:
        xc = docs[src].astype(np.float64)
        mx = xc.mean(axis=0)
        xc = xc - mx
        w, v = np.linalg.eigh(xc.T @ xc)
        w = np.maximum(w[::-1].copy(), 0.0)
        v = v[:, ::-1].copy()
        z = xc @ v
        qx = (queries[src].astype(np.float64) - mx) @ v
        wmean = max(float(w.mean()), 1e-12)
        for tgt in models:
            if tgt == src:
                continue
            yc = docs[tgt].astype(np.float64)
            my = yc.mean(axis=0)
            yc = yc - my
            qy = queries[tgt].astype(np.float64) - my
            zty = z.T @ yc
            qden = float((qy**2).sum())
            entry = out[f"{src}->{tgt}"]
            qres = []
            for ki, k in enumerate(ranks[src]):
                lam = entry["lam_mult"][ki] * wmean
                pred = qx[:, :k] @ (zty[:k] / (w[:k] + lam)[:, None])
                qres.append(float(((qy - pred) ** 2).sum() / qden))
            entry["residual_queries"] = qres
        del xc, z, qx, v
        logger.info("  residual: query transfer for %s done", src)
    return out


def run_geometry(
    unit_docs: Dict[str, Dict[str, np.ndarray]],
    unit_queries: Dict[str, np.ndarray],
    sample: int,
    dims: Sequence[int],
    folds: int,
    skip_residual: bool = False,
) -> Dict[str, object]:
    """All label-free pairwise statistics, per unit level.

    Row order is identical across models by construction (every model embeds the
    same ``CorpusUnits`` lists), so no alignment step is needed -- but that
    invariant is asserted, since a silent chunking drift would make every number
    here meaningless while still producing plausible output.
    """
    models = sorted(unit_docs)
    out: Dict[str, object] = {"models": models, "levels": {}, "residual": {}}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for level in GEOM_LEVELS:
        mats = {m: unit_docs[m][level] for m in models}
        n = mats[models[0]].shape[0]
        assert all(v.shape[0] == n for v in mats.values()), f"row misalignment at level {level}"
        logger.info("[geometry/%s] N=%d, %d models", level, n, len(models))

        # CKA inverts nothing, but it is still estimated from a sample covariance:
        # below N ~ d the centred data is rank-deficient and every pair scores
        # ~0.9 regardless of relationship. Record the regime rather than trusting
        # the number blindly (Qasper full dev is only ~2x the largest dim).
        max_dim = max(v.shape[1] for v in mats.values())
        n_over_d = n / max_dim
        if n_over_d < 2.0:
            logger.warning("[geometry/%s] N/d = %.1f -- CKA is in the rank-deficient regime", level, n_over_d)

        centred = {m: _centre(mats[m]) for m in models}
        norms = {m: _gram_norm(centred[m]) for m in models}
        idx = rng.choice(n, size=min(sample, n), replace=False)
        grams = {m: _rbf_gram_centred(mats[m][idx]) for m in models}
        bases = {m: svcca_basis(centred[m]) for m in models}

        cka_lin: Dict[str, float] = {}
        cka_rbf: Dict[str, float] = {}
        sv: Dict[str, object] = {}
        for a, b in itertools.combinations(models, 2):
            key = f"{a}|{b}"
            cka_lin[key] = linear_cka(centred[a], centred[b], norms[a], norms[b])
            cka_rbf[key] = kernel_cka(grams[a], grams[b])
            sv[key] = svcca(bases[a], bases[b], n)
            logger.info(
                "  %-16s cka_lin=%.4f cka_rbf=%.4f svcca=%.4f (k=%d/%d%s)",
                key,
                cka_lin[key],
                cka_rbf[key],
                sv[key]["svcca"],  # type: ignore[index]
                sv[key]["kx"],  # type: ignore[index]
                sv[key]["ky"],  # type: ignore[index]
                " DEGENERATE" if sv[key]["degenerate"] else "",  # type: ignore[index]
            )
        del grams, bases

        out["levels"][level] = {  # type: ignore[index]
            "n": int(n),
            "max_dim": int(max_dim),
            "n_over_d": float(n_over_d),
            "rbf_sample": int(len(idx)),
            "cka_linear": cka_lin,
            "cka_rbf": cka_rbf,
            "svcca": sv,
        }
        del centred

        if not skip_residual:
            logger.info("[residual/%s] ridge %d-fold over %d models ...", level, folds, len(models))
            out["residual"][level] = residual_energy(mats, unit_queries, dims, folds)  # type: ignore[index]
    return out


def run_align_fusion(
    all_singles: Dict[str, Dict[Tuple[str, str], ArmScores]],
    qids: List[str],
    qrels_by_target: Dict[str, Dict[str, Dict[str, int]]],
) -> Dict[str, object]:
    """Realized min-max fusion gain for EVERY ordered (chunk-model, section-model) pair.

    This is the bake-off's target column. Full dim, min-max only, w swept -- P2'
    §8.2/4 already settled that the rule choice is immaterial (z ~= minmax, RRF
    and max lose), so spending the sweep budget on pair COVERAGE instead of rule
    coverage is what buys statistical power for the predictor correlation.
    """
    w_ship = int(np.argmin(np.abs(ALPHAS - 0.65)))
    out: Dict[str, object] = {}
    for target in TARGETS:
        qrels = qrels_by_target[target]
        arms = {f"{m}/{lvl}": s[(lvl, target)] for m, s in all_singles.items() for lvl in ("chunk", "section")}
        bs_name = max(arms, key=lambda n: arms[n].mean_ndcg)
        bs = arms[bs_name]

        rows: Dict[str, object] = {}
        for cm in sorted(all_singles):
            for sm in sorted(all_singles):
                cs, ss = all_singles[cm][("chunk", target)], all_singles[sm][("section", target)]
                union, mc, ms = _union_fill(cs.units, cs.mat, ss.units, ss.mat)
                means, bi, pqs = _sweep(_minmax_rows(mc), _minmax_rows(ms), union, qids, qrels)
                arm_best = cs if cs.mean_ndcg >= ss.mean_ndcg else ss
                oracle_pq = np.maximum(cs.pq_ndcg, ss.pq_ndcg)
                gain_oracle = float(oracle_pq.mean()) - arm_best.mean_ndcg
                ma, mb = _align(cs.units, cs.mat, ss.units, ss.mat)
                rows[f"{cm}:{sm}"] = {
                    "same_model": cm == sm,
                    "chunk_ndcg": cs.mean_ndcg,
                    "section_ndcg": ss.mean_ndcg,
                    "best_arm_ndcg": arm_best.mean_ndcg,
                    "fused_best": means[bi],
                    "best_weight": float(ALPHAS[bi]),
                    "fused_at_0.65": means[w_ship],
                    "gain_vs_best_arm": means[bi] - arm_best.mean_ndcg,
                    "gain_vs_best_single": means[bi] - bs.mean_ndcg,
                    "oracle": float(oracle_pq.mean()),
                    "oracle_gain": gain_oracle,
                    "capture": (means[bi] - arm_best.mean_ndcg) / gain_oracle if gain_oracle > 1e-9 else None,
                    "mean_spearman": mean_spearman(ma, mb),
                    "bootstrap_vs_best_single": paired_bootstrap(pqs[bi], bs.pq_ndcg),
                }
            logger.info("  fusion targets [%s]: chunk-model %s done", target, cm)
        out[target] = {"best_single": {"arm": bs_name, "ndcg": bs.mean_ndcg}, "pairs": rows}
    return out


def _predictors(cm: str, sm: str, geom: Dict[str, object], row: Dict[str, object]) -> Dict[str, float]:
    """Every candidate predictor for one ordered pair, label-free ones first."""
    lv = geom["levels"]  # type: ignore[index]
    res = geom.get("residual", {})
    key = f"{cm}|{sm}" if f"{cm}|{sm}" in lv["chunk"]["cka_linear"] else f"{sm}|{cm}"

    def r(level: str, a: str, b: str, which: str, ki: int) -> float:
        block = res.get(level, {}).get(f"{a}->{b}")  # type: ignore[union-attr]
        if not block:
            return float("nan")
        vals = block[which]
        return float(vals[min(ki, len(vals) - 1)])

    p: Dict[str, float] = {}
    for level in GEOM_LEVELS:
        p[f"cka_lin_{level}"] = float(lv[level]["cka_linear"][key])
        p[f"cka_rbf_{level}"] = float(lv[level]["cka_rbf"][key])
        p[f"svcca_{level}"] = float(lv[level]["svcca"][key]["svcca"])
    p["cka_lin_mean"] = 0.5 * (p["cka_lin_chunk"] + p["cka_lin_section"])
    # Directional: how much of the SECTION model is unpredictable from the CHUNK
    # model (and the reverse), at full rank and at a 128-dim residual budget.
    p["resid_c2s_section"] = r("section", cm, sm, "residual", -1)
    p["resid_s2c_chunk"] = r("chunk", sm, cm, "residual", -1)
    p["resid_c2s_section@128"] = r("section", cm, sm, "residual", 1)
    p["resid_mean"] = float(np.nanmean([p["resid_c2s_section"], p["resid_s2c_chunk"]]))
    # Label-requiring references (the bar, not candidates).
    p["ref_neg_mean_spearman"] = -float(row["mean_spearman"])  # type: ignore[arg-type]
    p["ref_oracle_gain"] = float(row["oracle_gain"])  # type: ignore[arg-type]
    p["ref_best_arm_ndcg"] = float(row["best_arm_ndcg"])  # type: ignore[arg-type]
    # Composite: §8.2/1's backbone principle says arm quality should dominate
    # diversity. If this beats both parts, "best model + any decorrelated
    # second" is the whole selection rule.
    p["ref_quality_x_resid"] = p["ref_best_arm_ndcg"] * p["resid_c2s_section"]
    return p


LABEL_FREE = ("cka_", "svcca_", "resid_")
# Hypothesised direction, so "higher = predicts more fusion gain" holds for every
# row and rho is comparable down the table. CKA and SVCCA measure SIMILARITY, so
# the diversity hypothesis says they run the other way; without this the ranking
# rewards the wrong tail and the argmax "pick" chooses the most redundant pair --
# the exact opposite of the rule under test.
PREDICTOR_ORIENT = {"cka_": -1.0, "svcca_": -1.0}


def _orient(name: str) -> float:
    for prefix, sign in PREDICTOR_ORIENT.items():
        if name.startswith(prefix):
            return sign
    return 1.0


def run_bakeoff(geom: Dict[str, object], fusion: Dict[str, object]) -> Dict[str, object]:
    """Rank every predictor by how well it orders pairs by REALIZED fusion gain.

    Two scopes, because symmetric and directional predictors are not comparable
    on the same one:
      ordered         all cross-model (chunk-model, section-model) pairs. CKA and
                      SVCCA are constant within an unordered pair here, so this
                      scope penalises them for a real limitation.
      unordered_best  one row per unordered pair, target = the better of the two
                      orderings. Fair to symmetric predictors; the price is half
                      the sample size and the loss of the leg-assignment
                      question, which is the one a deployment must answer.
    """
    out: Dict[str, object] = {}
    for target in TARGETS:
        pairs = fusion[target]["pairs"]  # type: ignore[index]
        cross = {k: v for k, v in pairs.items() if not v["same_model"]}

        scopes: Dict[str, Dict[str, object]] = {}
        for scope in ("ordered", "unordered_best"):
            if scope == "ordered":
                items = [(k, v) for k, v in sorted(cross.items())]
            else:
                by_unordered: Dict[str, Tuple[str, Dict[str, object]]] = {}
                for k, v in sorted(cross.items()):
                    a, b = k.split(":")
                    uk = "|".join(sorted((a, b)))
                    cur = by_unordered.get(uk)
                    if cur is None or v["gain_vs_best_single"] > cur[1]["gain_vs_best_single"]:
                        by_unordered[uk] = (k, v)
                items = [by_unordered[u] for u in sorted(by_unordered)]

            names = [k for k, _ in items]
            preds = [_predictors(k.split(":")[0], k.split(":")[1], geom, v) for k, v in items]
            targets = {
                "gain_vs_best_single": [float(v["gain_vs_best_single"]) for _, v in items],
                "gain_vs_best_arm": [float(v["gain_vs_best_arm"]) for _, v in items],
            }
            keys = sorted(preds[0]) if preds else []
            quality = [p["ref_best_arm_ndcg"] for p in preds] if preds else []
            table: Dict[str, object] = {}
            for pk in keys:
                raw = [p[pk] for p in preds]
                sign = _orient(pk)
                col = [sign * v for v in raw]
                entry: Dict[str, object] = {
                    "label_free": pk.startswith(LABEL_FREE),
                    "orient": sign,
                    "values": raw,
                }
                for tname, tcol in targets.items():
                    boot = _spearman_boot(col, tcol)
                    # What this predictor would have picked under its own
                    # hypothesis, and what that costs against the best pair
                    # actually available (regret). Regret is the number a
                    # deployment cares about: rho can be mediocre while the
                    # top-1 pick is still right, and vice versa.
                    if np.all(np.isfinite(col)):
                        pick = int(np.argmax(col))
                        best = int(np.argmax(tcol))
                        boot["pick"] = names[pick]  # type: ignore[assignment]
                        boot["pick_gain"] = tcol[pick]
                        boot["regret"] = tcol[best] - tcol[pick]
                    if pk != "ref_best_arm_ndcg":
                        boot["partial"] = _partial_boot(col, tcol, quality)  # type: ignore[assignment]
                    entry[tname] = boot
                table[pk] = entry
            scopes[scope] = {"pairs": names, "targets": targets, "predictors": table}

        # Leave-one-model-out. THIS IS NOT OPTIONAL COLOUR: on the Qasper pool the
        # raw diversity-vs-gain correlations flip sign when the weakest model is
        # dropped (it is also the most decorrelated one, so it manufactures a
        # negative correlation single-handedly). A rho whose sign depends on one
        # model's presence is not a selection rule, and only this pass shows it.
        models = sorted({m for k in cross for m in k.split(":")})
        loo: Dict[str, object] = {}
        for drop in models:
            items = [(k, v) for k, v in sorted(cross.items()) if drop not in k.split(":")]
            if len(items) < 6:
                continue
            preds = [_predictors(k.split(":")[0], k.split(":")[1], geom, v) for k, v in items]
            tcol = [float(v["gain_vs_best_single"]) for _, v in items]
            qcol = [p["ref_best_arm_ndcg"] for p in preds]
            loo[drop] = {
                "n": len(items),
                "predictors": {
                    pk: {
                        "rho": _spearman([_orient(pk) * p[pk] for p in preds], tcol),
                        "partial": (
                            float("nan")
                            if pk == "ref_best_arm_ndcg"
                            else _partial_spearman([_orient(pk) * p[pk] for p in preds], tcol, qcol)
                        ),
                    }
                    for pk in sorted(preds[0])
                },
            }
        out[target] = {"scopes": scopes, "leave_one_out": loo}
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_boot(b: Dict[str, float]) -> str:
    return f"delta={b['delta']:+.4f} [{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}] p(win)={b['p_win']:.2f}"


def _pair(pairs: Dict[str, Dict[str, object]], a: str, b: str) -> Optional[Dict[str, object]]:
    return pairs.get(f"{a}|{b}") or pairs.get(f"{b}|{a}")


def print_report(models: List[str], p0a: Dict[str, object], p0b: Dict[str, object], p0c: Dict[str, object]) -> None:
    for target in TARGETS:
        block = p0a[target]  # type: ignore[index]
        print(f"\n===== singles + P0-A (target: {target.upper()}) =====")
        print(f"| {'arm':<18} | {'ndcg@10':>8} | {'recall@10':>9} |")
        for name, scores in sorted(block["arms"].items(), key=lambda kv: -kv[1]["ndcg@10"]):
            print(f"| {name:<18} | {scores['ndcg@10']:8.4f} | {scores['recall@10']:9.4f} |")
        pool = block["pool_oracle"]
        print(f"\npool oracle (all arms): ndcg@10={pool['ndcg@10']:.4f} usage={pool['usage']}")

        print(
            "\nP0-A pair oracle: rows = chunk-leg model, cols = section-leg model (cell: oracle, +gain vs best single)"
        )
        header = "| chunk\\section | " + " | ".join(f"{m:>16}" for m in models) + " |"
        print(header)
        for a in models:
            cells = []
            for b in models:
                p = _pair(block["pairs"], f"{a}/chunk", f"{b}/section")
                cells.append(f"{p['oracle']:.4f} +{p['gain']:.4f}" if p else "--")
            print(f"| {a:<13} | " + " | ".join(f"{c:>16}" for c in cells) + " |")

        top = sorted(block["pairs"].items(), key=lambda kv: -kv[1]["gain"])[:5]
        print("\ntop-5 oracle pairs by gain over best single:")
        for name, p in top:
            print(
                f"  {name:<40} oracle={p['oracle']:.4f} best={p['best_single']:.4f} ({p['best_arm']}) "
                f"gain=+{p['gain']:.4f}  boot: {_fmt_boot(p['bootstrap'])}"
            )

    print(
        "\n===== P0-B fusion-rule ablation (per model; weights: raw/z = chunk-block, minmax/raw_unit = section) ====="
    )
    for mkey, per_target in p0b.items():
        for target, entry in per_target.items():
            refs = entry["references"]
            rl, zl = entry["raw_linear"], entry["z_linear"]
            mm, ru = entry["minmax_unit"], entry["raw_unit"]
            print(f"\n[{mkey} @ {target}] chunk-alone={refs['chunk_alone']:.4f}", end="")
            print(f"  section-alone={refs['section_alone']:.4f}")
            print(
                f"  raw-linear:  best {rl['best_ndcg']:.4f} @ alpha={rl['best_weight']:.2f}"
                f"   at 0.5: {refs['raw_linear_at_0.5']:.4f}"
            )
            print(f"  z-linear:    best {zl['best_ndcg']:.4f} @ alpha={zl['best_weight']:.2f}")
            print(
                f"  minmax-unit: best {mm['best_ndcg']:.4f} @ w={mm['best_weight']:.2f}"
                f"   at 0.65: {refs['minmax_at_0.65']:.4f}"
            )
            print(f"  raw-unit:    best {ru['best_ndcg']:.4f} @ w={ru['best_weight']:.2f}")
            for name, b in entry["bootstrap"].items():
                print(f"    {name:<28} {_fmt_boot(b)}")

    print("\n===== P0-C MRL truncation ladder (ndcg@10) =====")
    for mkey, entry in p0c.items():
        dims = entry["dims"]
        for target in TARGETS:
            row_c = entry[target]["chunk"]
            row_s = entry[target]["section"]
            print(f"\n[{mkey} @ {target}]  dims: {dims}")
            print("  chunk:   " + "  ".join(f"{d}={v:.4f}" for d, v in zip(dims, row_c, strict=True)))
            print("  section: " + "  ".join(f"{d}={v:.4f}" for d, v in zip(dims, row_s, strict=True)))
            for dim, b in entry["boot_section_vs_full"][target].items():
                print(f"    section {dim} vs full: {_fmt_boot(b)}")


def print_p2_report(p2: Dict[str, object]) -> None:
    for target in TARGETS:
        entry = p2[target]  # type: ignore[index]
        base = entry["baselines"]
        print(f"\n===== P2' cross-model unit fusion (target: {target.upper()}) =====")
        for budget, info in base["best_single"].items():
            print(f"  best single [{budget}]: {info['arm']} = {info['ndcg']:.4f}")
        bsf = base["best_same_model_fused"]
        print(f"  best same-model fused [full]: {bsf['model']} = {bsf['ndcg']:.4f}")
        for m, sf in base["same_model_fused_full"].items():
            print(
                f"    {m:<8} best {sf['best_ndcg']:.4f} @ w={sf['best_weight']:.2f}" f"   at 0.65: {sf['at_0.65']:.4f}"
            )
        for pname, per_budget in entry["pairs"].items():
            for budget, r in per_budget.items():
                arms, mm = r["arms"], r["rules"]["minmax"]
                cap = r["capture_minmax"]
                print(
                    f"\n[{pname} @ {budget}] chunk={arms['chunk']['ndcg']:.4f}"
                    f" section={arms['section']['ndcg']:.4f} oracle={r['oracle']:.4f}"
                )
                print(
                    f"  minmax: best {mm['best_ndcg']:.4f} @ w={mm['best_weight']:.2f}"
                    f"   at 0.65: {mm['at_0.65']:.4f}   at 0.5: {mm['at_0.5']:.4f}"
                    + (f"   capture={cap:.0%}" if cap is not None else "")
                )
                for rule in ("z", "rrf"):
                    rr = r["rules"][rule]
                    print(
                        f"  {rule:<6}: best {rr['best_ndcg']:.4f} @ w={rr['best_weight']:.2f}"
                        f"   at 0.5: {rr['at_0.5']:.4f}"
                    )
                print(f"  max   : {r['rules']['max']['ndcg']:.4f}   (best rule: {r['best_rule']})")
                for name, b in r["bootstrap"].items():
                    print(f"    {name:<44} {_fmt_boot(b)}")


def print_align_report(geom: Dict[str, Any], fusion: Dict[str, Any], bake: Dict[str, Any]) -> None:
    print("\n===== Tier 0: representational geometry (label-free) =====")
    for level, block in geom["levels"].items():
        warn = "  <-- rank-deficient, CKA inflated" if block["n_over_d"] < 2.0 else ""
        print(
            f"\n-- {level} (N={block['n']}, max dim={block['max_dim']}, "
            f"N/d={block['n_over_d']:.1f}, RBF subsample={block['rbf_sample']}){warn} --"
        )
        print(f"  {'pair':<18} {'cka_lin':>8} {'cka_rbf':>8} {'svcca':>8}  components")
        for key, val in block["cka_linear"].items():
            sv = block["svcca"][key]
            flag = "  <-- DEGENERATE (N too small for CCA)" if sv["degenerate"] else ""
            print(
                f"  {key:<18} {val:>8.4f} {block['cka_rbf'][key]:>8.4f} "
                f"{sv['svcca']:>8.4f}  {sv['kx']}+{sv['ky']}{flag}"
            )
        res = geom.get("residual", {}).get(level)
        if res:
            ranks = next(iter(res.values()))["ranks"]
            print(f"\n  residual energy 1-R^2 (source -> target), ranks {ranks}; [q] = doc-fit map on QUERIES")
            for key in sorted(res):
                curve = " ".join(f"{v:.3f}" for v in res[key]["residual"])
                qcurve = " ".join(f"{v:.3f}" for v in res[key].get("residual_queries", []))
                print(f"    {key:<20} {curve}   [q] {qcurve}")

    for target in TARGETS:
        entry = fusion[target]
        bs = entry["best_single"]
        print(f"\n===== realized min-max fusion, all pairs (target: {target.upper()}) =====")
        print(f"  best single arm: {bs['arm']} = {bs['ndcg']:.4f}")
        print(f"  {'pair':<18} {'fused':>7} {'w':>5} {'vs_arm':>8} {'vs_best':>8} {'oracle+':>8} {'cap':>5} {'rho':>6}")
        ranked = sorted(entry["pairs"].items(), key=lambda kv: -kv[1]["gain_vs_best_single"])
        for name, r in ranked:
            cap = f"{r['capture']:.0%}" if r["capture"] is not None else "  -"
            mark = " *" if r["same_model"] else ""
            print(
                f"  {name:<18} {r['fused_best']:>7.4f} {r['best_weight']:>5.2f} "
                f"{r['gain_vs_best_arm']:>+8.4f} {r['gain_vs_best_single']:>+8.4f} "
                f"{r['oracle_gain']:>+8.4f} {cap:>5} {r['mean_spearman']:>6.3f}{mark}"
            )
        print("  (* = same-model pair, excluded from the bake-off correlations)")

    print("\n===== BAKE-OFF: predictor -> realized gain (Spearman over pairs) =====")
    print("rho is ORIENTED (dir=-1 rows are similarity measures, negated so that")
    print("higher always means 'predicts more gain'); a large NEGATIVE rho means the")
    print("predictor is reliably backwards, which is a finding, not a null result.")
    print("Kill criterion: a label-free predictor must land near ref_neg_mean_spearman,")
    print("which needs queries + qrels. If none does, corpus-level selection has no legs.")
    print("'partial' controls for ref_best_arm_ndcg: on this pool the most decorrelated")
    print("model is also the worst, so raw rho confounds diversity with arm quality.")
    for target in TARGETS:
        for scope, block in bake[target]["scopes"].items():
            names = block["pairs"]
            print(f"\n-- {target.upper()} / {scope} (n={len(names)} pairs) --")
            table = block["predictors"]
            order = sorted(table, key=lambda k: -(table[k]["gain_vs_best_single"]["rho"] or 0.0))
            print(
                f"  {'predictor':<26} {'free':>5} {'dir':>4} {'rho':>7} {'95% CI':>16} "
                f"{'partial':>8} {'95% CI':>16} {'regret':>8}  pick"
            )
            for pk in order:
                e = table[pk]
                b = e["gain_vs_best_single"]
                ci = f"[{b['ci_lo']:+.2f},{b['ci_hi']:+.2f}]"
                reg = f"{b['regret']:+.4f}" if "regret" in b else "       -"
                pt = b.get("partial")
                pcol = f"{pt['rho']:>8.3f}" if pt else "       -"
                pci = f"[{pt['ci_lo']:+.2f},{pt['ci_hi']:+.2f}]" if pt else "-"
                print(
                    f"  {pk:<26} {'yes' if e['label_free'] else 'no':>5} {int(e['orient']):>+4} "
                    f"{b['rho']:>7.3f} {ci:>16} {pcol} {pci:>16} {reg:>8}  {b.get('pick', '-')}"
                )

        loo = bake[target]["leave_one_out"]
        if loo:
            keys = [k for k in sorted(next(iter(loo.values()))["predictors"]) if not k.startswith("cka_rbf")]
            print(f"\n-- {target.upper()} / leave-one-model-out, ordered scope (rho / partial) --")
            print("  A predictor whose SIGN depends on one model's presence is not a rule.")
            print(f"  {'drop':<10} {'n':>3}  " + "  ".join(f"{k[:14]:>14}" for k in keys))
            for drop in sorted(loo):
                cells = []
                for pk in keys:
                    e = loo[drop]["predictors"][pk]
                    p = "  -  " if not np.isfinite(e["partial"]) else f"{e['partial']:+.2f}"
                    cells.append(f"{e['rho']:+.2f}/{p}".rjust(14))
                print(f"  {drop:<10} {loo[drop]['n']:>3}  " + "  ".join(cells))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=("qasper", "maud"), default="qasper", help="Corpus (default qasper).")
    common.add_argument("--split", default="dev", help="Qasper split (dev ~280 papers / train ~880); ignored for maud.")
    common.add_argument(
        "--max-papers", type=int, default=None, help="Cap the number of papers/contracts (default: all)."
    )
    common.add_argument("--chunk-tokens", type=int, default=None, help="Chunk size (default 500, the shipped default).")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", parents=[common], help="Embed one model's chunks/sections/queries into the cache.")
    e.add_argument("--model-key", required=True, choices=sorted(MODEL_POOL))
    e.add_argument("--dry-run", action="store_true", help="Report cache hit/miss counts; embed nothing.")
    a = sub.add_parser("analyze", parents=[common], help="Run P0-A/B/C from cached vectors.")
    a.add_argument("--models", default=",".join(MODEL_POOL), help="Comma-separated model keys.")
    a.add_argument("--tag", default="phase0", help="Result filename tag.")
    p2 = sub.add_parser("p2", parents=[common], help="Phase 2' cross-model unit-level fusion from cached vectors.")
    p2.add_argument("--pairs", default=P2_DEFAULT_PAIRS, help="chunkmodel:sectionmodel pairs, comma-separated.")
    p2.add_argument("--budget-dim", type=int, default=512, help="MRL slice for the matched-budget arms.")
    p2.add_argument("--allow-embed", action="store_true", help="Proceed even if vectors are missing from the cache.")
    p2.add_argument("--tag", default="phase2", help="Result filename tag.")
    al = sub.add_parser("align", parents=[common], help="Tier 0: representational geometry + predictor bake-off.")
    al.add_argument("--models", default=",".join(MODEL_POOL), help="Comma-separated model keys.")
    al.add_argument("--sample", type=int, default=2000, help="Row subsample for the RBF-kernel CKA.")
    al.add_argument("--folds", type=int, default=3, help="K for the ridge cross-validation.")
    al.add_argument(
        "--resid-dims",
        default=",".join(str(d) for d in RESID_DIMS),
        help="PCA ranks for the residual-energy curve (full dim is always appended).",
    )
    al.add_argument("--skip-residual", action="store_true", help="Geometry + fusion only (lower peak memory).")
    al.add_argument("--skip-fusion", action="store_true", help="Geometry only; no bake-off (no target column).")
    al.add_argument("--allow-embed", action="store_true", help="Proceed even if vectors are missing from the cache.")
    al.add_argument("--tag", default="tier0", help="Result filename tag.")
    return p.parse_args(argv)


def _guard_cache(model_keys: Sequence[str], units: CorpusUnits, allow_embed: bool) -> bool:
    """Refuse to start hours of embedding (or paid API calls) on a chunking/prefix drift."""
    missing: Dict[str, int] = {}
    for mkey in model_keys:
        _, stats = embed_model(MODEL_POOL[mkey], units, dry_run=True)
        n_miss = sum(s["misses"] for s in stats.values())  # type: ignore[index]
        if n_miss:
            missing[mkey] = n_miss
    if missing and not allow_embed:
        print(f"Cache misses (expected zero): {missing}. Re-run with --allow-embed to embed.", file=sys.stderr)
        return False
    return True


def _cmd_align(
    args: argparse.Namespace,
    bench,
    units: CorpusUnits,
    poolers: Dict[Tuple[str, str], Pooler],
    qrels_by_target: Dict[str, Dict[str, Dict[str, int]]],
    qids: List[str],
) -> int:
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODEL_POOL]
    if unknown:
        print(f"Unknown model keys: {unknown}", file=sys.stderr)
        return 1
    if len(model_keys) < 3:
        print("Need >= 3 models for a bake-off correlation to mean anything.", file=sys.stderr)
        return 1
    if not _guard_cache(model_keys, units, args.allow_embed):
        return 1

    dims = [int(d) for d in args.resid_dims.split(",") if d.strip()]

    # Peak memory is the binding constraint here (all models resident at once for
    # the pairwise geometry), so keep only unit-normalised matrices and the arm
    # scores; the raw vectors go as soon as both are derived.
    unit_docs: Dict[str, Dict[str, np.ndarray]] = {}
    unit_queries: Dict[str, np.ndarray] = {}
    all_singles: Dict[str, Dict[Tuple[str, str], ArmScores]] = {}
    embed_stats: Dict[str, object] = {}
    for mkey in model_keys:
        spec = MODEL_POOL[mkey]
        logger.info("=== loading %s (%s) ===", mkey, spec.model)
        vec, stats = embed_model(spec, units)
        assert vec is not None
        embed_stats[mkey] = stats
        if not args.skip_fusion:
            all_singles[mkey] = build_singles(vec, poolers, qrels_by_target, qids)
        unit_docs[mkey] = {"chunk": _unit(vec.chunks), "section": _unit(vec.sections)}
        unit_queries[mkey] = _unit(vec.queries)
        del vec

    logger.info("Tier 0 geometry over %d models ...", len(model_keys))
    geom = run_geometry(unit_docs, unit_queries, args.sample, dims, args.folds, args.skip_residual)
    del unit_docs, unit_queries

    fusion: Dict[str, object] = {}
    bake: Dict[str, object] = {}
    if not args.skip_fusion:
        logger.info("Realized fusion targets: %d^2 ordered pairs x %d targets ...", len(model_keys), len(TARGETS))
        fusion = run_align_fusion(all_singles, qids, qrels_by_target)
        bake = run_bakeoff(geom, fusion)

    if fusion:
        print_align_report(geom, fusion, bake)
    else:
        print_align_report(geom, {t: {"best_single": {"arm": "-", "ndcg": 0.0}, "pairs": {}} for t in TARGETS}, {})

    full = {
        "schema": 1,
        "phase": "tier0-align",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "papers": len(bench.corpus),
            "queries": len(qids),
            "chunks": len(units.chunk_texts),
            "sections": len(units.section_texts),
        },
        "chunk_tokens": args.chunk_tokens or 500,
        "params": {
            "sample": args.sample,
            "folds": args.folds,
            "resid_dims": dims,
            "svcca_var": SVCCA_VAR,
            "ridge_lam_mults": list(RIDGE_LAM_MULTS),
        },
        "models": {
            m: {
                "model": MODEL_POOL[m].model,
                "num_ctx": MODEL_POOL[m].num_ctx,
                "query_prefix": MODEL_POOL[m].query_prefix,
                "doc_prefix": MODEL_POOL[m].doc_prefix,
                "dim": MODEL_POOL[m].dim,
                "note": MODEL_POOL[m].note,
                "cache": embed_stats[m],
            }
            for m in model_keys
        },
        "geometry": geom,
        "fusion": fusion,
        "bakeoff": bake,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS_DIR / f"dual_{args.tag}_{args.dataset}_{args.split}_{len(bench.corpus)}p_{stamp}.json"
    out.write_text(json.dumps(full, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


def _cmd_p2(
    args: argparse.Namespace,
    bench,
    units: CorpusUnits,
    poolers: Dict[Tuple[str, str], Pooler],
    qrels_by_target: Dict[str, Dict[str, Dict[str, int]]],
    qids: List[str],
) -> int:
    pairs: List[Tuple[str, str]] = []
    for spec_str in args.pairs.split(","):
        cm, _, sm = spec_str.strip().partition(":")
        if cm not in MODEL_POOL or sm not in MODEL_POOL:
            print(f"Bad pair spec {spec_str!r} (want chunkmodel:sectionmodel)", file=sys.stderr)
            return 1
        pairs.append((cm, sm))
    needed = list(dict.fromkeys(m for pair in pairs for m in pair))

    # The Qasper leg is supposed to be zero-embedding; a cache miss means a
    # chunking/prefix drift, not new work to do.
    if not _guard_cache(needed, units, args.allow_embed):
        return 1

    bkey = str(args.budget_dim)
    singles_by_budget: Dict[str, Dict[str, Dict[Tuple[str, str], ArmScores]]] = {"full": {}, bkey: {}}
    embed_stats: Dict[str, object] = {}
    for mkey in needed:
        spec = MODEL_POOL[mkey]
        logger.info("=== loading %s (%s) ===", mkey, spec.model)
        vec, stats = embed_model(spec, units)
        assert vec is not None
        embed_stats[mkey] = stats
        singles_by_budget["full"][mkey] = build_singles(vec, poolers, qrels_by_target, qids)
        sliced = ModelVectors(
            vec.chunks[:, : args.budget_dim], vec.sections[:, : args.budget_dim], vec.queries[:, : args.budget_dim]
        )
        singles_by_budget[bkey][mkey] = build_singles(sliced, poolers, qrels_by_target, qids)
        del vec, sliced

    logger.info("P2' cross-model unit fusion: %d pairs x 2 budgets ...", len(pairs))
    p2 = run_p2(pairs, singles_by_budget, qids, qrels_by_target)
    print_p2_report(p2)

    full = {
        "schema": 1,
        "phase": f"P2-{args.dataset}",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "papers": len(bench.corpus),
            "queries": len(qids),
            "chunks": len(units.chunk_texts),
            "sections": len(units.section_texts),
        },
        "chunk_tokens": args.chunk_tokens or 500,
        "budget_dim": args.budget_dim,
        "pairs": [f"{cm}:{sm}" for cm, sm in pairs],
        "models": {
            m: {
                "model": MODEL_POOL[m].model,
                "num_ctx": MODEL_POOL[m].num_ctx,
                "query_prefix": MODEL_POOL[m].query_prefix,
                "doc_prefix": MODEL_POOL[m].doc_prefix,
                "note": MODEL_POOL[m].note,
                "cache": embed_stats[m],
            }
            for m in needed
        },
        "p2": p2,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS_DIR / f"dual_{args.tag}_{args.dataset}_{args.split}_{len(bench.corpus)}p_{stamp}.json"
    out.write_text(json.dumps(full, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    global TARGETS, SCOPE_QID
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("localvectordb.chunking").setLevel(logging.ERROR)
    _load_experiments_env()
    args = parse_args(argv)

    if args.dataset == "maud":
        from benchmarks.maud_data import detect_contract_sections, load_maud

        # Per-contract retrieval: single SECTION target (the doc target is
        # degenerate when every query is scoped to its own contract), scoped
        # ranking + within-contract normalization via SCOPE_QID.
        bench = load_maud(max_contracts=args.max_papers)
        TARGETS = ("section",)
        SCOPE_QID = {qid: qid.split("||", 1)[0] for qid in bench.queries}
        args.split = "all"  # MAUD's own splits are per-example; keep filenames honest
        units = load_units(bench, args.chunk_tokens, detect_contract_sections)
    else:
        from benchmarks.qasper_data import load_qasper

        bench = load_qasper(split=args.split, max_papers=args.max_papers)
        units = load_units(bench, args.chunk_tokens)
    logger.info(
        "Units: %d chunks, %d sections, %d queries over %d papers",
        len(units.chunk_texts),
        len(units.section_texts),
        len(units.query_ids),
        len(bench.corpus),
    )

    if args.cmd == "embed":
        spec = MODEL_POOL[args.model_key]
        _, stats = embed_model(spec, units, dry_run=args.dry_run)
        print(json.dumps({"model": spec.model, "dry_run": args.dry_run, "stats": stats}, indent=2))
        return 0

    qrels_by_target = {"doc": bench.doc_qrels, "section": bench.section_qrels}
    qids = units.query_ids
    poolers = {
        ("chunk", "doc"): Pooler(units.chunk_doc),
        ("chunk", "section"): Pooler(units.chunk_section),
        ("section", "doc"): Pooler(units.section_doc),
        ("section", "section"): Pooler(units.section_ids),
    }

    if args.cmd == "p2":
        return _cmd_p2(args, bench, units, poolers, qrels_by_target, qids)
    if args.cmd == "align":
        return _cmd_align(args, bench, units, poolers, qrels_by_target, qids)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODEL_POOL]
    if unknown:
        print(f"Unknown model keys: {unknown}", file=sys.stderr)
        return 1

    all_singles: Dict[str, Dict[Tuple[str, str], ArmScores]] = {}
    p0b: Dict[str, object] = {}
    p0c: Dict[str, object] = {}
    embed_stats: Dict[str, object] = {}
    for mkey in model_keys:
        spec = MODEL_POOL[mkey]
        logger.info("=== loading %s (%s) ===", mkey, spec.model)
        vec, stats = embed_model(spec, units)
        assert vec is not None
        embed_stats[mkey] = stats
        singles = build_singles(vec, poolers, qrels_by_target, qids)
        all_singles[mkey] = singles
        logger.info("[%s] P0-B fusion-rule ablation ...", mkey)
        p0b[mkey] = run_p0b(vec, units, singles, qrels_by_target, qids)
        logger.info("[%s] P0-C truncation ladder ...", mkey)
        p0c[mkey] = run_p0c(spec, vec, poolers, qrels_by_target, qids)
        del vec  # free the raw arrays before the next model loads

    logger.info("P0-A complementarity/oracle across %d models ...", len(model_keys))
    p0a = run_p0a(all_singles, qids)

    print_report(model_keys, p0a, p0b, p0c)

    per_query = {
        target: {
            f"{mkey}/{level}": [round(float(v), 4) for v in all_singles[mkey][(level, target)].pq_ndcg]
            for mkey in model_keys
            for level in ("chunk", "section")
        }
        for target in TARGETS
    }
    full = {
        "schema": 1,
        "phase": "P0",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "papers": len(bench.corpus),
            "queries": len(qids),
            "chunks": len(units.chunk_texts),
            "sections": len(units.section_texts),
        },
        "chunk_tokens": args.chunk_tokens or 500,
        "models": {
            m: {
                "model": MODEL_POOL[m].model,
                "num_ctx": MODEL_POOL[m].num_ctx,
                "query_prefix": MODEL_POOL[m].query_prefix,
                "doc_prefix": MODEL_POOL[m].doc_prefix,
                "mrl_dims": list(MODEL_POOL[m].mrl_dims),
                "note": MODEL_POOL[m].note,
                "cache": embed_stats[m],
            }
            for m in model_keys
        },
        "p0a": p0a,
        "p0b": p0b,
        "p0c": p0c,
        "query_ids": qids,
        "per_query_ndcg": per_query,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    papers_tag = f"{len(bench.corpus)}p"
    out = RESULTS_DIR / f"dual_{args.tag}_{args.dataset}_{args.split}_{papers_tag}_{stamp}.json"
    out.write_text(json.dumps(full, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
