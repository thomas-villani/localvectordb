"""Section-level regression gate for the hierarchical retrieval path.

WHY THIS EXISTS
---------------
``eval_retrieval.py`` -- the gate this project uses for ``src/`` default changes --
contains **zero** occurrences of ``hierarchical``, ``section`` or ``search_level``.
It is BEIR SciFact at chunk level. So every section-level default is currently
ungated: a change to ``section_vector_strategy``, to ``_span_embed`` window
sizing, or to chunk->section attribution passes ``--check`` at +0.0000 while
moving section retrieval by up to 0.36 nDCG (``span-length-crossover-findings``
§6.26). This file closes that hole.

Unlike ``eval_levels.py``, which reimplements the pooling maths in numpy, this
gate drives the **real** ``LocalVectorDB.query()`` path -- ingest, FAISS section
index, ``search_level``, the lot -- so it fails when ``src/`` breaks, not when a
harness diverges.

THE TWO LEGS
------------
``qasper``
    Real NLP papers, sections averaging ~190 tokens. The **short-section**
    regime, where a raw-span section vector really is one global embedding that
    never window-pools, and where raw-span is defensible (egemma +0.0818, §6.23).

``superdocs``
    Synthetic FiQA super-documents, sections of ~24k chars (~7k tokens). The
    **long-section** regime the headline finding is actually about, where
    raw-span loses 0.25-0.36 nDCG to the centroid.

Why synthetic for the long leg: §6.30 established that ``src/``'s
``SectionDetector`` is a two-group, line-anchored Markdown regex, so it finds
essentially no sections in MAUD contracts -- the corpus the long-section result
was measured on cannot be used to gate ``src/`` at all. ``superdocs.py`` emits
``## Section N`` headings and *asserts* every gold span aligns to a
detector-assigned section, so the ground truth is src-detectable by construction.
A gate needs sensitivity to the change under test, not external validity; the
external-validity claim lives in the findings doc, on real corpora.

READ THE LONG LEG'S DELTAS, NOT ITS ABSOLUTE NUMBERS
----------------------------------------------------
``mode="point"`` places **one** gold passage per super-document, so the answer is
a median 825 chars inside a ~24.5k-char section (**3.4%**) and a 74k-char
document (**1.1%**). Every arm is therefore scored under near-worst-case
dilution, and the section arms sit low in absolute terms by construction -- that
is the design, not a defect, because a gate needs to move when ``src/`` changes,
not to flatter the section level. The question "can a section vector ever beat a
diluted chunk average when the gold is *dense*" is a different experiment
(``mode="section"`` clusters a query's golds into one section); it is not what
this file measures.

WHAT THE LONG LEG IS SENSITIVE TO
---------------------------------
With the default eval model (all-MiniLM-L6-v2, ``max_seq_length`` 256) the
raw-span path currently encodes **7.3%** of a 24.5k-char section: ``_span_embed``
asks the provider for its context window, ``SentenceTransformerEmbeddings``
exposes neither ``num_ctx`` nor ``max_input_tokens``, so the fixed 24,000-char
default wins and each window is then truncated to 256 tokens by the model. Both
pending ``src/`` fixes -- the ``section_vector_strategy`` default and the window
sizing -- move this leg hard and move the qasper leg barely at all. That contrast
is the point of having two legs rather than one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.config import DATA_DIR, EVAL_EMBEDDING_MODEL, EVAL_EMBEDDING_PROVIDER  # noqa: E402
from benchmarks.metrics import ndcg_at_k, recall_at_k  # noqa: E402

logger = logging.getLogger("eval_hier_gate")

BASELINE_JSON = _ROOT / "benchmarks" / "hier_gate_baseline.json"
RECALL_K = (1, 5, 10)
K = 10
TOLERANCE = 0.005

# Per-metric tolerance, set from MEASURED rebuild variance rather than taste.
#
# Two independent rebuilds of the same code (§6.31/5) agree exactly at document
# level -- 0.1838 and every recall identical -- but differ by 0.0044 on
# ndcg@10_sections. The cause is exact ties: sections that own no chunk of their
# own inherit their neighbours' chunk vectors, so 17.9% of qasper's section
# vectors are duplicates. Ordering *within* the returned candidates is now
# deterministic (results sort on (-score, id)), but which tied candidates FAISS
# RETURNS at the top-k boundary still depends on index order, which the threaded
# ingest assigns differently per build.
#
# So the section metric gets a tolerance above its noise floor, and is reported
# as advisory rather than authoritative. Do not tighten it without either
# over-fetching and selecting ties deterministically, or making faiss_id
# assignment reproducible -- otherwise the gate will fail on rebuilds that
# changed nothing. A corpus with no duplicate section vectors (superdocs: 0
# duplicates) is exactly reproducible and does not need the headroom.
SECTION_TOLERANCE = 0.015
METRIC_TOLERANCE = {"ndcg@10_sections": SECTION_TOLERANCE}

# Long-leg grid. S x P FiQA passages (~767 chars each) per super-document, so
# P=32 puts a section at ~24.5k chars -- just past the 24,000-char window
# ``_span_embed`` falls back to, which is what makes the leg exercise multi-window
# pooling and not only truncation. 200 queries keeps a rebuild near ~8 min/arm on
# CPU at ~8.8k tok/s; the distractor pool needs S*P*queries <= 57,638 passages.
SUPERDOC_SECTIONS = 3
SUPERDOC_PASSAGES = 32
SUPERDOC_QUERIES = 200
SUPERDOC_SEED = 0


@dataclass(frozen=True)
class Leg:
    """One corpus the gate scores, with the cache key that identifies its index."""

    name: str
    cache_key: str
    load: Callable[[], Any]


@dataclass(frozen=True)
class HierConfig:
    search_level: str
    strategy: str  # section_vector_strategy the DB was built with

    @property
    def label(self) -> str:
        # The chunk leg does not read section vectors, so the strategy is not
        # part of its identity -- labelling it would imply two distinct arms.
        return self.search_level if self.search_level == "chunks" else f"{self.search_level} · {self.strategy}"


def build_configs() -> List[HierConfig]:
    out = [HierConfig("chunks", "centroid")]
    for strategy in ("rawspan", "centroid"):
        for level in ("sections", "fused"):
            out.append(HierConfig(level, strategy))
    return out


# Settings that change only how we TALK to the provider, never what it returns.
# See the cache-key discussion in build_db before adding to this set.
_TRANSPORT_ONLY_EMBEDDING_KEYS = frozenset(
    {"max_concurrent_requests", "timeout", "max_retries", "retry_delay", "base_url"}
)


def qasper_leg(max_papers: Optional[int]) -> Leg:
    from benchmarks.qasper_data import load_qasper

    key = "qasper" if max_papers is None else f"qasper_max{max_papers}"
    return Leg(name="qasper", cache_key=key, load=lambda: load_qasper(split="dev", max_papers=max_papers))


def superdocs_leg(sections: int, passages: int, max_queries: int, seed: int) -> Leg:
    from benchmarks import beir_data
    from benchmarks.superdocs import build_synthetic_benchmark

    def _load():
        source = beir_data.load("fiqa")
        return build_synthetic_benchmark(
            source,
            sections_per_doc=sections,
            passages_per_section=passages,
            seed=seed,
            max_queries=max_queries,
            mode="point",
        )

    key = f"superdocs_fiqa_s{sections}p{passages}q{max_queries}seed{seed}"
    return Leg(name="superdocs", cache_key=key, load=_load)


def build_db(
    bench,
    *,
    leg: Leg,
    provider: str,
    model: str,
    strategy: str,
    rebuild: bool,
    chunk_size: Optional[int] = None,
    embedding_config: Optional[Dict[str, Any]] = None,
):
    """Build (or reopen) a hierarchical DB for one section_vector_strategy.

    The strategy is baked in at ingest -- section vectors are written once -- so
    each arm needs its own database, and the key must carry the strategy or the
    two arms silently share an index and the A/B compares a build to itself.

    The key must ALSO carry every parameter that changes the CONTENT of the
    index. Omitting ``max_papers`` once made a ``--max-papers 12`` smoke build get
    reused by the full 275-paper run: 263 gold documents were simply absent and
    every arm scored at chance (nDCG 0.0357 ~= 10/275) while looking like a real
    result. ``Leg.cache_key`` carries those parameters now, so a new grid cannot
    collide with an old one.

    ``chunk_size`` follows the same rule but suffixes the key **only when set**,
    so ``None`` (= "whatever the library defaults to", currently 500) keeps the
    historical key and every already-built gate/density index still hits cache.
    The consequence is that ``chunk_size=None`` and ``chunk_size=500`` are two
    keys for the same content; pass ``None`` unless you are deliberately sweeping.

    ``embedding_config`` is the opposite case: it is deliberately kept OUT of the
    key, because it may only carry TRANSPORT settings (how fast we talk to the
    provider), which cannot change a single stored vector. Ollama's default of 3
    concurrent requests leaves this box at ~250 tok/s against ~600 available, so
    a build that should take an hour takes two and a half. The allowlist below is
    what keeps that shortcut honest -- anything content-affecting (a prefix, a
    dimension, a task type) would produce different vectors under an unchanged
    key, which is precisely the cache poisoning the key rules exist to prevent.
    """
    from localvectordb import LocalVectorDB

    key = f"hiergate__{leg.cache_key}__{model.replace('/', '_').replace(':', '-')}__{strategy}"
    if chunk_size is not None:
        key += f"__c{chunk_size}"
    base = DATA_DIR / "db"
    base.mkdir(parents=True, exist_ok=True)
    sentinel = base / f"{key}.complete"

    def _discard() -> None:
        for path in base.glob(f"{key}.*"):
            path.unlink(missing_ok=True)

    if rebuild:
        _discard()

    kwargs: Dict[str, Any] = dict(
        embedding_provider=provider,
        embedding_model=model,
        hierarchical_embeddings=True,
        section_vector_strategy=strategy,
    )
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    if embedding_config:
        unsafe = set(embedding_config) - _TRANSPORT_ONLY_EMBEDDING_KEYS
        if unsafe:
            raise ValueError(
                f"embedding_config carries content-affecting keys {sorted(unsafe)}; these would change "
                "the stored vectors without changing the cache key. Add them to the key, not the allowlist."
            )
        kwargs["embedding_config"] = dict(embedding_config)
    if sentinel.exists():
        logger.info("Reusing cached DB %s", key)
        return LocalVectorDB(key, base, **kwargs)

    if any(base.glob(f"{key}.*")):
        logger.warning("Discarding incomplete DB %s", key)
        _discard()

    logger.info("Building %s (%d docs) -- slow part", key, len(bench.corpus))
    db = LocalVectorDB(key, base, **kwargs)
    doc_ids = list(bench.corpus)
    slab = 100
    for start in range(0, len(doc_ids), slab):
        batch = doc_ids[start : start + slab]
        db.upsert([bench.corpus[d] for d in batch], ids=batch)
        logger.info("  ingested %d/%d", min(start + slab, len(doc_ids)), len(doc_ids))
    db.save()
    # Confirm the strategy actually persisted; a silently-defaulted DB would make
    # the two arms identical and the whole comparison vacuous.
    got = getattr(db, "section_vector_strategy", None)
    if got != strategy:
        raise RuntimeError(f"DB reports section_vector_strategy={got!r}, expected {strategy!r}")
    # Same reasoning for the sweep parameter: a silently-defaulted chunk_size
    # would make every rung of a chunk_size ladder the same build.
    if chunk_size is not None and db.chunk_size != chunk_size:
        raise RuntimeError(f"DB reports chunk_size={db.chunk_size!r}, expected {chunk_size!r}")
    sentinel.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return db


def _section_qrel_id(result_id: str) -> str:
    """``{doc}:section:{i}`` (what ``query`` returns) -> ``{doc}#s{i}`` (what qrels use)."""
    doc_id, _, index = result_id.rpartition(":section:")
    return f"{doc_id}#s{index}"


def run_config(db, bench, config: HierConfig) -> Dict[str, float]:
    """Score one arm at document level, and -- where meaningful -- at section level.

    Document level is the common unit every arm can be measured in, so it is the
    primary metric. But it is **structurally blind to whether a given section is
    retrievable at all**: ``search_level="sections", return_type="documents"``
    rolls a document up to the score of its *best* section, so a document with one
    good section is found whether its other nine are reachable or dead. A defect
    that makes 26% of gold sections unretrievable therefore barely moves this
    number -- which is exactly what happened when the chunkless-section fix landed
    (-0.0066 on qasper, entirely from added vectors reshuffling the ranking).

    ``ndcg@10_sections`` scores the sections themselves against ``section_qrels``.
    It is the only metric here that can see a section become reachable, so any
    change to section *vectors* must be read on it rather than on the doc-level
    column. Only computed for ``search_level="sections"``: ``fused`` mixes chunk
    and section hits and has no section-level ground truth to be scored against.
    """
    scores: Dict[str, List[float]] = {f"recall@{k}": [] for k in RECALL_K}
    scores["ndcg@10"] = []
    section_scores: List[float] = []
    score_sections = config.search_level == "sections" and bool(getattr(bench, "section_qrels", None))

    for qid, text in bench.queries.items():
        rel = bench.doc_qrels.get(qid, {})
        if not any(r > 0 for r in rel.values()):
            continue
        hits = db.query(
            text,
            search_level=config.search_level,
            return_type="documents",
            k=K,
        )
        ranked = [h.id for h in hits]
        scores["ndcg@10"].append(ndcg_at_k(ranked, rel, K))
        for k in RECALL_K:
            scores[f"recall@{k}"].append(recall_at_k(ranked, rel, k))

        if score_sections:
            sec_rel = bench.section_qrels.get(qid, {})
            if any(r > 0 for r in sec_rel.values()):
                sec_hits = db.query(text, search_level="sections", k=K)
                sec_ranked = [_section_qrel_id(h.id) for h in sec_hits]
                section_scores.append(ndcg_at_k(sec_ranked, sec_rel, K))

    out = {m: (sum(v) / len(v) if v else 0.0) for m, v in scores.items()}
    if section_scores:
        out["ndcg@10_sections"] = sum(section_scores) / len(section_scores)
    return out


def run_leg(leg: Leg, args: argparse.Namespace) -> Dict[str, Any]:
    """Load one corpus, build both strategy DBs, and score every arm."""
    bench = leg.load()
    logger.info("[%s] %d docs, %d queries", leg.name, len(bench.corpus), len(bench.queries))

    dbs = {
        s: build_db(
            bench,
            leg=leg,
            provider=args.embedding_provider,
            model=args.embedding_model,
            strategy=s,
            rebuild=args.rebuild,
        )
        for s in ("rawspan", "centroid")
    }

    # Belt and braces on the cache key: assert the index actually holds the corpus
    # being scored. A stale-but-valid database is the failure mode here -- it does
    # not error, it just returns chance-level numbers that read as a real result.
    for strategy, db in dbs.items():
        n = db.count()
        if n != len(bench.corpus):
            raise RuntimeError(
                f"[{leg.name}] DB[{strategy}] holds {n} documents but the benchmark has "
                f"{len(bench.corpus)}. Stale cached index -- re-run with --rebuild."
            )
    logger.info("[%s] index check: both DBs hold %d documents", leg.name, len(bench.corpus))

    results: Dict[str, Dict[str, float]] = {}
    for config in build_configs():
        m = run_config(dbs[config.strategy], bench, config)
        results[config.label] = m
        logger.info(
            "[%s] %-24s ndcg@10 %.4f  r@1 %.4f  r@10 %.4f%s",
            leg.name,
            config.label,
            m["ndcg@10"],
            m["recall@1"],
            m["recall@10"],
            f"  sec-ndcg {m['ndcg@10_sections']:.4f}" if "ndcg@10_sections" in m else "",
        )
    return {"n_docs": len(bench.corpus), "n_queries": len(bench.queries), "results": results}


def compare_to_baseline(legs: Dict[str, Any], path: Path, tolerance: float) -> int:
    if not path.exists():
        print(f"No baseline at {path}; run with --save-baseline first.", file=sys.stderr)
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = payload.get("legs", {})
    if not base:
        # The pre-two-leg format stored a flat top-level "results". Reading it as
        # a leg map yields {}, every arm reports "NEW ... not gated", and --check
        # returns 0 -- a gate that passes because it compared nothing. Fail loudly.
        print(
            f"Baseline {path} has no 'legs' section"
            + (" (pre-two-leg format)" if "results" in payload else "")
            + "; regenerate it with --leg both --save-baseline.",
            file=sys.stderr,
        )
        return 1
    worst, failed = 0.0, []
    for leg_name, payload in legs.items():
        print(f"\n{leg_name}:")
        if leg_name not in base:
            print(f"  NEW  leg {leg_name} (not in baseline) -- not gated")
            continue
        base_results = base[leg_name]["results"]
        for label, metrics in payload["results"].items():
            if label not in base_results:
                print(f"  NEW  {label} (not in baseline)")
                continue
            # Both metrics are gated. Document level alone would miss a change that
            # makes sections unreachable, because a rolled-up document takes the max
            # over its sections and survives losing most of them.
            for metric in ("ndcg@10", "ndcg@10_sections"):
                if metric not in metrics or metric not in base_results[label]:
                    continue
                delta = metrics[metric] - base_results[label][metric]
                limit = METRIC_TOLERANCE.get(metric, tolerance)
                worst = min(worst, delta)
                flag = "FAIL" if delta < -limit else "ok"
                if delta < -limit:
                    failed.append(f"{leg_name}/{label}/{metric}")
                suffix = "" if metric == "ndcg@10" else " [sec]"
                print(f"  {flag:4s} {label:24s}{suffix:6s} {metrics[metric]:.4f}  ({delta:+.4f})")
    print(f"\nworst delta {worst:+.4f} (tolerance {tolerance})")
    if failed:
        print("REGRESSED: " + ", ".join(failed))
    return 1 if failed else 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-provider", default=EVAL_EMBEDDING_PROVIDER)
    p.add_argument("--embedding-model", default=EVAL_EMBEDDING_MODEL)
    p.add_argument("--leg", choices=("qasper", "superdocs", "both"), default="both")
    p.add_argument("--max-papers", type=int, default=None, help="qasper leg: cap papers (smoke only)")
    p.add_argument("--sections", type=int, default=SUPERDOC_SECTIONS, help="superdocs leg: sections per doc")
    p.add_argument("--passages", type=int, default=SUPERDOC_PASSAGES, help="superdocs leg: passages per section")
    p.add_argument("--max-queries", type=int, default=SUPERDOC_QUERIES, help="superdocs leg: docs to build")
    p.add_argument("--seed", type=int, default=SUPERDOC_SEED)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--baseline", type=Path, default=BASELINE_JSON)
    p.add_argument("--tolerance", type=float, default=TOLERANCE)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    # LocalVectorDB.__init__ turns ANY provider failure into a bare "model is not
    # available" (a cold HF cache included), so surface the real error first.
    from benchmarks.eval_retrieval import preflight_embedding_model

    preflight_embedding_model(args.embedding_provider, args.embedding_model)

    wanted = ("qasper", "superdocs") if args.leg == "both" else (args.leg,)
    legs: List[Leg] = []
    if "qasper" in wanted:
        legs.append(qasper_leg(args.max_papers))
    if "superdocs" in wanted:
        legs.append(superdocs_leg(args.sections, args.passages, args.max_queries, args.seed))

    payloads: Dict[str, Any] = {leg.name: run_leg(leg, args) for leg in legs}

    print("\n" + "=" * 72)
    for leg_name, payload in payloads.items():
        print(f"{leg_name}  ({payload['n_docs']} docs, {payload['n_queries']} queries)")
        print(f"  {'arm':24s} {'ndcg@10':>9s} {'r@1':>8s} {'r@5':>8s} {'r@10':>8s} {'sec-ndcg':>9s}")
        for label, m in payload["results"].items():
            sec = f"{m['ndcg@10_sections']:9.4f}" if "ndcg@10_sections" in m else f"{'-':>9s}"
            print(
                f"  {label:24s} {m['ndcg@10']:9.4f} {m['recall@1']:8.4f} "
                f"{m['recall@5']:8.4f} {m['recall@10']:8.4f} {sec}"
            )
    print("=" * 72)

    if args.save_baseline:
        if args.leg != "both":
            # A partial save would silently drop the other leg's baseline and
            # leave it ungated from then on, with nothing in the file to say so.
            print("Refusing to save a partial baseline; re-run with --leg both.", file=sys.stderr)
            return 1
        args.baseline.write_text(
            json.dumps(
                {
                    "generated": datetime.now(timezone.utc).isoformat(),
                    "model": args.embedding_model,
                    "legs": {
                        "qasper": {
                            "dataset": "qasper-dev",
                            "regime": "SHORT sections (~190 tok mean); raw-span never window-pools",
                            **payloads["qasper"],
                        },
                        "superdocs": {
                            "dataset": (
                                f"fiqa-superdocs s{args.sections}p{args.passages} " f"seed{args.seed} (SYNTHETIC)"
                            ),
                            "regime": (
                                f"LONG sections (~{args.passages * 767 // 1000}k chars); "
                                "raw-span truncates and window-pools"
                            ),
                            **payloads["superdocs"],
                        },
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote baseline {args.baseline}")
    if args.check:
        return compare_to_baseline(payloads, args.baseline, args.tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
