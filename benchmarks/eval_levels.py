#!/usr/bin/env python
"""Level-representation and per-query-routing analyses over the dual-embedding cache.

Two questions the shipped experiments left open, both answerable with ZERO new
embedding from the ``eval_dual`` per-(model, text) disk cache.

``hier`` -- **centroid vs raw-span at the section level, stratified by span
length.** ``hierarchical-test-findings.md`` F2 ("raw-span beats centroid") is
load-bearing for the shipped ``search_level=fused`` work, but it was only ever
measured on one encoder over Qasper and synthetic super-docs, whose sections
mostly fit in 2k tokens. MAUD contracts are the first corpus here with genuinely
varied section lengths, so this subcommand adds the missing ``centroid`` arm --
free, since a centroid is the mean of chunk vectors already in the cache -- and
reads every arm inside gold-section length bands.

The bands do double duty. Each model window-mean-pools any span past its context
cap, so a band straddling a model's cap is a **natural experiment on context
length**: over MAUD@50 egemma (2k ctx) pools 755 sections while openai (8k ctx)
pools 77, which makes the 2k-8k band exactly the set where egemma pools and
openai does not. Comparing the egemma-vs-openai gap inside that band against the
same gap in the <=2k band (where neither pools) is a difference-in-differences
estimate of what pooling costs -- the question ``dual-embedding-findings.md`` §2
records Qasper as structurally unable to answer.

``route`` -- **can a label-free per-query signal beat a single global fusion
weight?** Three independent threads converged on one wall: fusion realizes only
20-45% of the per-query oracle (``dual-embedding-findings.md`` §8.2/4), the
oracle beat fusion everywhere the hierarchical study measured it (F7), and
corpus-level geometry provably cannot express a per-query decision
(``tier0-selection-findings.md`` §6). This subcommand first prices the ceiling
(what a perfect per-query weight would buy over the best global weight), then
tests realizable routers -- score margin, cross-leg agreement, top-1 confidence,
softmax entropy, query length -- as both hard routers and continuous adaptive
weights. Everything is **cross-validated**: a threshold fitted and read on the
same queries is guaranteed to flatter itself, and that artifact is the single
most likely way this thread produces a false positive.

The routing sweep is cheap because of one identity: the shipped fusion weight
``w`` is the SECTION-leg weight, so w=0 is the chunk arm alone and w=1 is the
section arm alone, and *every* router in this family -- hard routing included --
is just "choose w per query from the grid". Pre-computing per-query nDCG once
per grid point reduces the whole study to array indexing.

Layering: primitives come from ``eval_dual`` (corpus units, cache-backed
encoders, poolers, scoped scoring, paired bootstrap) exactly as ``eval_dual``
imports from ``eval_hierarchical``, so arms here are directly comparable to the
p2/align tables and cannot drift from them.

Usage (both are analysis-only; they refuse to embed):

    ./.venv/Scripts/python.exe benchmarks/eval_levels.py hier \\
        --dataset maud --max-papers 50 --models openai,egemma
    ./.venv/Scripts/python.exe benchmarks/eval_levels.py route --dataset qasper
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from benchmarks import eval_dual as ed  # noqa: E402
from benchmarks.config import RESULTS_DIR  # noqa: E402
from benchmarks.eval_hierarchical import _estimate_tokens, _unit  # noqa: E402

logger = logging.getLogger("benchmarks.eval_levels")

# Bands are the model context caps in the pool (egemma/nomic 2k; openai/arctic/
# qwen3 8k), so a band boundary is exactly where a model starts window-pooling.
BANDS: Tuple[Tuple[str, int, float], ...] = (
    ("<=2k", 0, 2048),
    ("2k-8k", 2048, 8192),
    (">8k", 8192, float("inf")),
)

# Continuous-router grid. g=0 reduces exactly to the global-weight baseline, so
# the adaptive family provably contains it and cannot lose on training data.
ROUTER_GAINS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
CV_FOLDS = 5
TOPK_SIGNAL = 10


# ---------------------------------------------------------------------------
# Shared setup (mirrors eval_dual.main's dataset branch, including the MAUD
# single-target + per-contract scoping globals).
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    bench: Any
    units: ed.CorpusUnits
    qids: List[str]
    poolers: Dict[Tuple[str, str], ed.Pooler]
    qrels_by_target: Dict[str, Dict[str, Dict[str, int]]]


def setup(args: argparse.Namespace) -> Corpus:
    if args.dataset == "maud":
        from benchmarks.maud_data import detect_contract_sections, load_maud

        bench = load_maud(max_contracts=args.max_papers)
        ed.TARGETS = ("section",)
        ed.SCOPE_QID = {qid: qid.split("||", 1)[0] for qid in bench.queries}
        args.split = "all"
        units = ed.load_units(bench, args.chunk_tokens, detect_contract_sections)
    else:
        from benchmarks.qasper_data import load_qasper

        bench = load_qasper(split=args.split, max_papers=args.max_papers)
        ed.TARGETS = ("doc", "section")
        ed.SCOPE_QID = None
        units = ed.load_units(bench, args.chunk_tokens)

    logger.info(
        "Units: %d chunks, %d sections, %d queries over %d docs",
        len(units.chunk_texts),
        len(units.section_texts),
        len(units.query_ids),
        len(bench.corpus),
    )
    return Corpus(
        bench=bench,
        units=units,
        qids=units.query_ids,
        poolers={
            ("chunk", "doc"): ed.Pooler(units.chunk_doc),
            ("chunk", "section"): ed.Pooler(units.chunk_section),
            ("section", "doc"): ed.Pooler(units.section_doc),
            ("section", "section"): ed.Pooler(units.section_ids),
        },
        qrels_by_target={"doc": bench.doc_qrels, "section": bench.section_qrels},
    )


def load_vectors(mkey: str, units: ed.CorpusUnits, allow_embed: bool) -> ed.ModelVectors:
    """Load one model's cached vectors, refusing to silently embed on cache drift."""
    _, dry = ed.embed_model(ed.MODEL_POOL[mkey], units, dry_run=True)
    misses = sum(int(s["misses"]) for s in dry.values())  # type: ignore[index]
    if misses and not allow_embed:
        raise SystemExit(
            f"[{mkey}] {misses} vectors missing from cache. This analysis is meant to be "
            f"zero-compute; pass --allow-embed only if you intend to embed them."
        )
    vec, _ = ed.embed_model(ed.MODEL_POOL[mkey], units)
    assert vec is not None
    return vec


# ---------------------------------------------------------------------------
# split: does a chunker's win come from not cutting gold evidence in half?
#
# Qasper's gold evidence is a paragraph with known char offsets, so "did this
# chunker split the relevant span" is directly checkable -- no embedding, and no
# appeal to aggregate margin statistics (which failed to explain the crossover in
# §3, and failed again for straddle in §6.4).
# ---------------------------------------------------------------------------


def qasper_gold_spans(bench, split: str) -> Dict[str, List[Tuple[str, int, int]]]:
    """Rebuild ``qid -> [(doc_id, start, end)]``; ``load_qasper`` keeps only the section id."""
    from benchmarks import qasper_data as qd
    from localvectordb.section_detection import SectionDetector

    root = qd.download()
    data = json.loads(sorted(root.glob(f"*{split}*.json"))[0].read_text(encoding="utf-8"))
    det = SectionDetector()
    out: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    for pid in sorted(data):
        if pid not in bench.corpus:
            continue
        text, para_spans = qd._render_paper(data[pid])
        detected = det.detect_sections(text)
        for qa in data[pid].get("qas") or []:
            qid = qa.get("question_id")
            if not qid or qid not in bench.queries:
                continue
            for ev in qd._evidence_strings(qa):
                span = para_spans.get(ev) or para_spans.get(ev.strip())
                if span is not None and qd._owner_section_index(detected, span) is not None:
                    out[qid].append((pid, span[0], span[1]))
    return dict(out)


def _chunk_starts(
    bench,
    arm: ChunkArm,
    ck: int,
    heading_finder: Optional[Callable[[str], Iterable[int]]] = None,
) -> Dict[str, List[int]]:
    ckw = {"heading_finder": heading_finder} if (heading_finder and arm.method == "structure") else {}
    chunker = ed._build_chunker(arm.size_for(ck), arm.method, arm.overlap, **ckw)
    return {doc: sorted({c.position.start for c in chunker.chunk(text)}) for doc, text in bench.corpus.items()}


def _splits_gold(starts: Dict[str, List[int]], doc: str, s: int, e: int) -> bool:
    """True if a chunk boundary falls strictly inside the gold span [s, e)."""
    b = starts.get(doc) or []
    i = bisect.bisect_right(b, s)
    return i < len(b) and b[i] < e


def run_split(
    corpus: Corpus,
    mkey: str,
    arms: List[ChunkArm],
    ck: int,
    split: str,
    allow_embed: bool,
    heading_finder: Optional[Callable[[str], Iterable[int]]] = None,
    pooling: str = "max",
) -> Dict:
    if len(arms) != 2:
        raise SystemExit("split needs exactly two arms: --chunk-methods 'baseline,candidate'")
    spans = gold_spans_for(corpus, split)
    logger.info("gold spans for %d/%d queries", len(spans), len(corpus.bench.queries))
    detector = _section_detector(corpus)

    pq: Dict[str, Dict[str, float]] = {}
    starts: Dict[str, Dict[str, List[int]]] = {}
    base_chunks: Dict[str, List[Tuple[int, int, str]]] = {}
    for arm in arms:
        ckw = {"heading_finder": heading_finder} if (heading_finder and arm.method == "structure") else None
        units = ed.load_units(corpus.bench, arm.size_for(ck), detector, arm.method, arm.overlap, ckw)
        vec = load_vectors(mkey, units, allow_embed)
        pooler = ed.Pooler(units.chunk_section, mode=pooling)
        sims = (_unit(vec.queries) @ _unit(vec.chunks).T).astype(np.float32)
        s, _ = ed.score_arm(pooler.units, pooler.pool(sims), units.query_ids, corpus.bench.section_qrels)
        pq[arm.label] = dict(zip(units.query_ids, s, strict=True))
        starts[arm.label] = _chunk_starts(corpus.bench, arm, ck, heading_finder)
        if arm is arms[0]:  # baseline attribution, for the per-query recoverability split
            for (a, b), doc, sid in zip(units.chunk_spans, units.chunk_doc, units.chunk_section, strict=True):
                base_chunks.setdefault(doc, []).append((a, b, sid))
        del vec

    base, cand = arms[0].label, arms[1].label

    def _base_split_shape(sp: List[Tuple[str, int, int]]) -> Tuple[bool, int]:
        """(did the BASELINE scatter gold across >1 section, most chunks any span was cut into).

        The harm-relevant reading of the flag: one span landing in two sections means
        the gold section never sees all of the evidence, however the query's other
        spans fell. The count is the free-redundancy dose -- a 3-way split hands
        max-pooling three shots where a 2-way split hands it two.
        """
        scattered, most = False, 1
        for doc, s, e in sp:
            touched = [c for c in base_chunks.get(doc, ()) if c[0] < e and c[1] > s]
            most = max(most, len(touched))
            if len(touched) > 1 and len({sid for _, _, sid in touched}) > 1:
                scattered = True
        return scattered, most

    rows = []
    for qid, sp in spans.items():
        if qid not in pq[base]:
            continue
        sb = any(_splits_gold(starts[base], d, s, e) for d, s, e in sp)
        sc = any(_splits_gold(starts[cand], d, s, e) for d, s, e in sp)
        scattered, most = _base_split_shape(sp)
        rows.append((sb, sc, pq[base][qid], pq[cand][qid], scattered, most))

    grand = float(np.mean([r[3] - r[2] for r in rows]))
    cells = {
        "base_splits_cand_whole": lambda r: r[0] and not r[1],
        "both_split": lambda r: r[0] and r[1],
        "both_whole": lambda r: not r[0] and not r[1],
        "cand_splits_base_whole": lambda r: not r[0] and r[1],
        # The identified test: inside the rescue cell, ONE corpus, ONE chunk size,
        # ONE ceiling -- only whether pooling could have repaired the split varies.
        # The mechanism says the unrecoverable half gains more.
        "rescue_UNrecoverable": lambda r: r[0] and not r[1] and r[4],
        "rescue_recoverable": lambda r: r[0] and not r[1] and not r[4],
        # Free-redundancy dose-response. If the recoverable subgroup loses because
        # it surrenders extra shots at max-pooling, the loss must GROW with the
        # number of pieces. Query difficulty predicts no such gradient, so this is
        # what separates the mechanism from "recoverable spans are just easier".
        "rescue_recov_2chunks": lambda r: r[0] and not r[1] and not r[4] and r[5] == 2,
        "rescue_recov_3plus": lambda r: r[0] and not r[1] and not r[4] and r[5] >= 3,
        "rescue_UNrecov_2chunks": lambda r: r[0] and not r[1] and r[4] and r[5] == 2,
        "rescue_UNrecov_3plus": lambda r: r[0] and not r[1] and r[4] and r[5] >= 3,
    }
    out: Dict[str, object] = {
        "base": base,
        "candidate": cand,
        "chunk_tokens": ck,
        "n_queries": len(rows),
        "gold_split_rate": {
            base: float(np.mean([r[0] for r in rows])),
            cand: float(np.mean([r[1] for r in rows])),
        },
        "overall": ed.paired_bootstrap(np.array([r[3] for r in rows]), np.array([r[2] for r in rows])),
        "cells": {},
    }
    for name, fn in cells.items():
        sel = [r for r in rows if fn(r)]
        if not sel:
            out["cells"][name] = {"n": 0}  # type: ignore[index]
            continue
        b = ed.paired_bootstrap(np.array([r[3] for r in sel]), np.array([r[2] for r in sel]))
        out["cells"][name] = {  # type: ignore[index]
            "n": len(sel),
            "base_ndcg": float(np.mean([r[2] for r in sel])),
            "cand_ndcg": float(np.mean([r[3] for r in sel])),
            **b,
            # Fraction of the OVERALL delta this cell accounts for.
            "share_of_gain": float((len(sel) / len(rows)) * b["delta"] / grand) if grand else float("nan"),
        }
    return out


def print_split(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 92)
    print("GOLD-EVIDENCE SPLIT: is the win 'did not cut the relevant span in half'?")
    print("=" * 92)
    print(f"  {res['candidate']} vs {res['base']} @ {res['chunk_tokens']} tokens, {res['n_queries']} queries")
    for k, v in res["gold_split_rate"].items():
        print("    gold split rate  %-18s %5.1f%%" % (k, 100 * v))
    print("\n  %-26s %6s %8s %8s %9s %19s %6s %8s" % ("cell", "n", "base", "cand", "delta", "95% CI", "p_win", "share"))
    for name, c in res["cells"].items():
        if not c["n"]:
            print("  %-26s %6d" % (name, 0))
            continue
        print(
            "  %-26s %6d %8.4f %8.4f %+9.4f  [%+.4f,%+.4f] %6.2f %7.0f%%"
            % (
                name,
                c["n"],
                c["base_ndcg"],
                c["cand_ndcg"],
                c["delta"],
                c["ci_lo"],
                c["ci_hi"],
                c["p_win"],
                100 * c["share_of_gain"],
            )
        )
    o = res["overall"]
    print("\n  overall %+.4f [%+.4f,%+.4f] p_win %.2f" % (o["delta"], o["ci_lo"], o["ci_hi"], o["p_win"]))


# ---------------------------------------------------------------------------
# ratio: is the chunk:section size ratio what decides whether a split is fatal?
#
# §6.10 explains the Qasper-wins / MAUD-null gap with a mechanism: cutting a gold
# span is harmless when every piece pools back to the SAME section (max-pooling
# reassembles it) and harmful when the pieces land in different sections. Which
# one happens is governed by chunk tokens / section tokens -- Qasper 1.92 (20.3%
# recoverable), MAUD 1.43 (56.8%).
#
# That is two corpora, i.e. an association with n=2 and everything else differing
# too. This subcommand turns it into a manipulation: hold the corpus, the
# sections and the gold fixed, sweep chunk size, and watch the ratio move. The
# prediction is directional and stated before running -- recoverability FALLS as
# the ratio rises, and the structure chunker's benefit RISES with it. A benefit
# that is flat across a several-fold swing in ratio falsifies the mechanism.
#
# Structural stats need no embedding; --score adds the paired nDCG delta from
# cached vectors so ratio, recoverability and benefit land in ONE table (joining
# them by hand across runs is how a size gets mismatched).
# ---------------------------------------------------------------------------


def gold_spans_for(corpus: Corpus, split: str) -> Dict[str, List[Tuple[str, int, int]]]:
    """``qid -> [(doc_id, start, end)]`` in corpus-text coordinates, either corpus.

    MAUD carries clause spans on the benchmark already (``load_maud`` maps them
    through ``pos_map``); Qasper's have to be rebuilt from the raw JSON.
    """
    if ed.SCOPE_QID is not None:
        return {
            qid: [(qid.split("||", 1)[0], s, e) for s, e in spans]
            for qid, spans in corpus.bench.gold_spans.items()
            if spans
        }
    return qasper_gold_spans(corpus.bench, split)


def _section_detector(corpus: Corpus):
    """The detector whose boundaries this corpus' section qrels were built from."""
    if corpus.bench.__class__.__module__.endswith("maud_data") or ed.SCOPE_QID is not None:
        from benchmarks.maud_data import detect_contract_sections

        return detect_contract_sections
    return None


def _recoverability(units: ed.CorpusUnits, spans: Dict[str, List[Tuple[str, int, int]]]) -> Dict[str, float]:
    """How often a cut gold span survives because both halves pool to one section."""
    by_doc: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    for (a, b), doc, sid in zip(units.chunk_spans, units.chunk_doc, units.chunk_section, strict=True):
        by_doc[doc].append((a, b, sid))

    n_gold = n_split = 0
    touched, sections = [], []
    for ranges in spans.values():
        for doc, s, e in ranges:
            chunks = [c for c in by_doc.get(doc, ()) if c[0] < e and c[1] > s]
            if not chunks:
                continue
            n_gold += 1
            if len(chunks) <= 1:
                continue
            n_split += 1
            touched.append(len(chunks))
            sections.append(len({sid for _, _, sid in chunks}))

    ns = np.array(sections)
    return {
        "n_gold": n_gold,
        "n_split": n_split,
        "split_rate": n_split / n_gold if n_gold else float("nan"),
        "mean_chunks_touched": float(np.mean(touched)) if touched else float("nan"),
        "mean_sections_touched": float(ns.mean()) if n_split else float("nan"),
        # The mechanism's quantity: of the spans this chunker cut, how many can
        # max-pooling put back together?
        "recoverable": float(np.mean(ns == 1)) if n_split else float("nan"),
    }


def run_ratio(
    corpus: Corpus,
    arms: List[ChunkArm],
    sizes: List[int],
    split: str,
    mkey: str,
    score: bool,
    allow_embed: bool,
    heading_finder: Optional[Callable[[str], Iterable[int]]] = None,
) -> Dict[str, object]:
    spans = gold_spans_for(corpus, split)
    logger.info("gold spans for %d/%d queries", len(spans), len(corpus.bench.queries))
    detector = _section_detector(corpus)
    target = "section" if "section" in ed.TARGETS else ed.TARGETS[0]

    out: Dict[str, object] = {
        "target": target,
        "sizes": sizes,
        "arms": [a.label for a in arms],
        "scored": bool(score),
        "model": mkey if score else None,
        "n_gold_queries": len(spans),
        "by_arm": {},
    }
    pq_by: Dict[Tuple[str, int], np.ndarray] = {}
    sec_tokens: Optional[float] = None

    for arm in arms:
        rows: Dict[str, object] = {}
        for ck in sizes:
            actual = arm.size_for(ck)
            logger.info("=== %s @ chunk_tokens=%d (arm size %d) ===", arm.label, ck, actual)
            ckw = {"heading_finder": heading_finder} if (heading_finder and arm.method == "structure") else None
            units = ed.load_units(corpus.bench, actual, detector, arm.method, arm.overlap, ckw)
            if sec_tokens is None:  # sections do not move with chunk size
                sec_tokens = float(np.median([_estimate_tokens(t) for t in units.section_texts]))
            chunk_tok = float(np.median([_estimate_tokens(t) for t in units.chunk_texts]))
            row: Dict[str, object] = {
                "arm_chunk_tokens": actual,
                "median_chunk_tokens": chunk_tok,
                "median_section_tokens": sec_tokens,
                "ratio": chunk_tok / max(sec_tokens, 1.0),
                "n_vectors": len(units.chunk_texts),
                **_recoverability(units, spans),
            }
            if score:
                vec = load_vectors(mkey, units, allow_embed)
                pooler = ed.Pooler(units.chunk_section if target == "section" else units.chunk_doc)
                sims = (_unit(vec.queries) @ _unit(vec.chunks).T).astype(np.float32)
                pq, _ = ed.score_arm(pooler.units, pooler.pool(sims), corpus.qids, corpus.qrels_by_target[target])
                pq_by[(arm.label, ck)] = pq
                row["ndcg"] = float(pq.mean())
                del vec
            rows[str(ck)] = row
        out["by_arm"][arm.label] = rows  # type: ignore[index]

    if score and len(arms) > 1:
        base = arms[0].label
        out["vs_first_arm"] = {
            "base": base,
            "deltas": {
                a.label: {str(ck): ed.paired_bootstrap(pq_by[(a.label, ck)], pq_by[(base, ck)]) for ck in sizes}
                for a in arms[1:]
            },
        }
    return out


def print_ratio(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("CHUNK:SECTION RATIO -- does recoverability (and the benefit) track it?")
    print("=" * 100)
    scored = res.get("scored")
    for label, rows in res["by_arm"].items():
        print(f"\n--- arm {label} ---")
        hdr = "  %8s %9s %9s %7s %9s %8s %8s %9s" % (
            "chunk",
            "med_tok",
            "sec_tok",
            "ratio",
            "gold_split",
            "chunks",
            "sects",
            "recover",
        )
        print(hdr + ("%9s" % "nDCG@10" if scored else ""))
        for ck, r in rows.items():
            actual = r.get("arm_chunk_tokens")
            shown = ck if actual in (None, int(ck)) else "%s(%s)" % (ck, actual)
            line = "  %8s %9.0f %9.0f %7.2f %8.1f%% %8.2f %8.2f %8.1f%%" % (
                shown,
                r["median_chunk_tokens"],
                r["median_section_tokens"],
                r["ratio"],
                100 * r["split_rate"],
                r["mean_chunks_touched"],
                r["mean_sections_touched"],
                100 * r["recoverable"],
            )
            print(line + ("%9.4f" % r["ndcg"] if scored else ""))
    vs = res.get("vs_first_arm")
    if vs:
        print(f"\n  paired vs {vs['base']} (same queries, matched nominal size):")
        print("    %-20s %8s %9s %20s %7s" % ("arm", "size", "dnDCG", "95% CI", "p_win"))
        for label, by_size in vs["deltas"].items():
            for ck, b in by_size.items():
                print(
                    "    %-20s %8s %+9.4f  [%+.4f,%+.4f] %7.2f"
                    % (label, ck, b["delta"], b["ci_lo"], b["ci_hi"], b["p_win"])
                )
        print(
            "\n  PREDICTION (§6.10): recoverable falls as ratio rises; the benefit rises with it.\n"
            "  A benefit flat across the ratio swing falsifies the mechanism."
        )


# ---------------------------------------------------------------------------
# hier: centroid vs raw-span, banded by gold-section length
# ---------------------------------------------------------------------------


@dataclass
class Centroids:
    """Section centroids plus the free by-products of building them.

    ``coherence`` is the norm of the mean of the member unit vectors BEFORE
    re-normalisation: 1.0 when every chunk points the same way, → 0 as they
    spread out. It is a label-free, ingest-time measure of how well one vector
    can stand for the span -- the natural criterion for a dynamic chunker, and
    free because the chunk vectors already exist.
    """

    vectors: np.ndarray
    empty: np.ndarray
    coherence: np.ndarray
    n_chunks: np.ndarray


def section_centroids(chunk_vecs: np.ndarray, units: ed.CorpusUnits) -> Centroids:
    """Mean of member chunk vectors per section, in ``units.section_ids`` row order.

    Members are unit-normalised before averaging so a long chunk cannot dominate
    by norm, then the mean is unit-normalised at the boundary (the T1.5 rule).
    Sections owning no chunk get a zero row: ``_unit`` leaves those at zero, so
    they score cosine 0 against every query and are effectively unreachable --
    the same structural cost P0-B found for concat's chunkless sections, here
    measured rather than assumed.
    """
    row = {sid: i for i, sid in enumerate(units.section_ids)}
    acc = np.zeros((len(units.section_ids), chunk_vecs.shape[1]), dtype=np.float64)
    cnt = np.zeros(len(units.section_ids), dtype=np.int64)
    member = _unit(chunk_vecs.astype(np.float64))
    for j, sid in enumerate(units.chunk_section):
        i = row.get(sid)
        if i is None:  # chunk owned by a heading-less span (never embedded as a section)
            continue
        acc[i] += member[j]
        cnt[i] += 1
    nonempty = cnt > 0
    acc[nonempty] /= cnt[nonempty, None]
    coherence = np.linalg.norm(acc, axis=1)  # pre-normalisation norm == part agreement
    return Centroids(acc.astype(np.float32), ~nonempty, coherence, cnt)


def _arm(
    level: str,
    target: str,
    vecs: np.ndarray,
    queries: np.ndarray,
    corpus: Corpus,
) -> ed.ArmScores:
    pooler = corpus.poolers[(level, target)]
    sims = (_unit(queries) @ _unit(vecs).T).astype(np.float32)
    pooled = pooler.pool(sims)
    pq, rec = ed.score_arm(pooler.units, pooled, corpus.qids, corpus.qrels_by_target[target])
    return ed.ArmScores(pooler.units, pooled, pq, float(pq.mean()), float(rec.mean()))


def _fuse_best(a: ed.ArmScores, b: ed.ArmScores, corpus: Corpus, target: str) -> Tuple[float, float, np.ndarray]:
    """Tuned min-max fusion of two arms; returns (best_w, best_ndcg, per-query nDCG)."""
    union, ma, mb = ed._union_fill(a.units, a.mat, b.units, b.mat)
    ma, mb = ed._scope_nan((ma, mb), corpus.qids, union)
    means, bi, pqs = ed._sweep(
        ed._minmax_rows(ma), ed._minmax_rows(mb), union, corpus.qids, corpus.qrels_by_target[target]
    )
    return float(ed.ALPHAS[bi]), means[bi], pqs[bi]


def band_of(tokens: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= tokens < hi:
            return name
    return BANDS[-1][0]


def query_bands(corpus: Corpus) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Band each query by its LONGEST gold section.

    Rationale: a query is "hard for a short-context model" when its answer lives
    in a span that model cannot see whole. Queries whose gold sections are all
    absent from the section universe are banded ``unknown`` and excluded from
    banded reads (they are not excluded from the overall numbers).
    """
    tokens = {
        sid: _estimate_tokens(t) for sid, t in zip(corpus.units.section_ids, corpus.units.section_texts, strict=True)
    }
    qrels = corpus.qrels_by_target["section"]
    bands: Dict[str, str] = {}
    for qid in corpus.qids:
        gold = [tokens[s] for s, r in qrels.get(qid, {}).items() if r > 0 and s in tokens]
        bands[qid] = band_of(max(gold)) if gold else "unknown"
    counts: Dict[str, int] = {}
    for b in bands.values():
        counts[b] = counts.get(b, 0) + 1
    return bands, counts


def _band_means(pq: np.ndarray, qids: Sequence[str], bands: Dict[str, str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, _, _ in BANDS:
        sel = np.array([bands[q] == name for q in qids])
        out[name] = float(pq[sel].mean()) if sel.any() else float("nan")
    return out


def _band_boot(
    pq_a: np.ndarray, pq_b: np.ndarray, qids: Sequence[str], bands: Dict[str, str], band: str
) -> Optional[Dict[str, float]]:
    sel = np.array([bands[q] == band for q in qids])
    if sel.sum() < 10:
        return None
    return ed.paired_bootstrap(pq_a[sel], pq_b[sel])


def did_pooling(
    pq_short: np.ndarray,
    pq_long: np.ndarray,
    qids: Sequence[str],
    bands: Dict[str, str],
    inner: str = "<=2k",
    outer: str = "2k-8k",
    n: int = ed.BOOTSTRAP_N,
) -> Optional[Dict[str, float]]:
    """Difference-in-differences: does the short-ctx model fall further behind
    in the band where it window-pools and the long-ctx model does not?

    DiD = (short - long | outer band) - (short - long | inner band). A
    significantly negative DiD is direct evidence that window pooling costs
    retrieval quality; ~zero says pooling is benign and the "context length is
    not the bottleneck" reading survives on a corpus that could have refuted it.
    Bands are resampled independently (they are disjoint query sets).
    """
    si = np.array([bands[q] == inner for q in qids])
    so = np.array([bands[q] == outer for q in qids])
    if si.sum() < 10 or so.sum() < 10:
        return None
    di, do = (pq_short - pq_long)[si], (pq_short - pq_long)[so]
    rng = np.random.default_rng(ed.BOOTSTRAP_SEED)
    bi = di[rng.integers(0, len(di), size=(n, len(di)))].mean(axis=1)
    bo = do[rng.integers(0, len(do), size=(n, len(do)))].mean(axis=1)
    dd = bo - bi
    return {
        "gap_inner": float(di.mean()),
        "gap_outer": float(do.mean()),
        "did": float(do.mean() - di.mean()),
        "ci_lo": float(np.percentile(dd, 2.5)),
        "ci_hi": float(np.percentile(dd, 97.5)),
        "p_negative": float((dd < 0).mean()),
        "n_inner": int(si.sum()),
        "n_outer": int(so.sum()),
    }


def run_hier(corpus: Corpus, model_keys: List[str], allow_embed: bool) -> Dict[str, object]:
    bands, band_counts = query_bands(corpus)
    logger.info("Query bands by longest gold section: %s", band_counts)

    sec_tokens = np.array([_estimate_tokens(t) for t in corpus.units.section_texts])
    section_band_counts = {name: int(((sec_tokens >= lo) & (sec_tokens < hi)).sum()) for name, lo, hi in BANDS}

    out: Dict[str, object] = {
        "bands": {name: {"lo": lo, "hi": None if hi == float("inf") else hi} for name, lo, hi in BANDS},
        "query_band_counts": band_counts,
        "section_band_counts": section_band_counts,
        "models": {},
    }
    per_model_pq: Dict[str, Dict[str, np.ndarray]] = {}

    for mkey in model_keys:
        logger.info("=== %s ===", mkey)
        vec = load_vectors(mkey, corpus.units, allow_embed)
        cstats = section_centroids(vec.chunks, corpus.units)
        cent, empty = cstats.vectors, cstats.empty
        n_empty = int(empty.sum())

        # How much of the gold is unreachable through centroids at all?
        empty_ids = {sid for sid, e in zip(corpus.units.section_ids, empty, strict=True) if e}
        gold_ids = {s for q in corpus.qids for s, r in corpus.qrels_by_target["section"].get(q, {}).items() if r > 0}
        gold_empty = len(gold_ids & empty_ids)

        entry: Dict[str, object] = {
            "model": ed.MODEL_POOL[mkey].model,
            "num_ctx": ed.MODEL_POOL[mkey].num_ctx,
            "chunkless_sections": n_empty,
            "chunkless_gold_sections": gold_empty,
            "gold_sections": len(gold_ids),
            "targets": {},
        }

        for target in ed.TARGETS:
            chunk = _arm("chunk", target, vec.chunks, vec.queries, corpus)
            raw = _arm("section", target, vec.sections, vec.queries, corpus)
            ctr = _arm("section", target, cent, vec.queries, corpus)
            w_raw, nd_raw, pq_fraw = _fuse_best(chunk, raw, corpus, target)
            w_ctr, nd_ctr, pq_fctr = _fuse_best(chunk, ctr, corpus, target)

            arms = {
                "chunk": chunk.pq_ndcg,
                "section_rawspan": raw.pq_ndcg,
                "section_centroid": ctr.pq_ndcg,
                "fused_rawspan": pq_fraw,
                "fused_centroid": pq_fctr,
            }
            entry["targets"][target] = {  # type: ignore[index]
                "overall": {k: float(v.mean()) for k, v in arms.items()},
                "recall10": {
                    "chunk": chunk.recall10,
                    "section_rawspan": raw.recall10,
                    "section_centroid": ctr.recall10,
                },
                "best_weight": {"fused_rawspan": w_raw, "fused_centroid": w_ctr},
                "banded": {k: _band_means(v, corpus.qids, bands) for k, v in arms.items()},
                "boot": {
                    # F2, the headline: does embedding the span beat averaging it?
                    "rawspan_vs_centroid": ed.paired_bootstrap(raw.pq_ndcg, ctr.pq_ndcg),
                    "rawspan_vs_chunk": ed.paired_bootstrap(raw.pq_ndcg, chunk.pq_ndcg),
                    "fused_rawspan_vs_chunk": ed.paired_bootstrap(pq_fraw, chunk.pq_ndcg),
                    "fused_rawspan_vs_rawspan": ed.paired_bootstrap(pq_fraw, raw.pq_ndcg),
                },
                "boot_banded_rawspan_vs_centroid": {
                    name: _band_boot(raw.pq_ndcg, ctr.pq_ndcg, corpus.qids, bands, name) for name, _, _ in BANDS
                },
            }
            per_model_pq.setdefault(target, {})[f"{mkey}/section_rawspan"] = raw.pq_ndcg
            per_model_pq[target][f"{mkey}/chunk"] = chunk.pq_ndcg

        out["models"][mkey] = entry  # type: ignore[index]
        del vec

    # Context-length natural experiment: short-ctx vs long-ctx model, inside vs
    # outside the band where only the short model window-pools.
    caps = {m: (ed.MODEL_POOL[m].num_ctx or 8192) for m in model_keys}
    did: Dict[str, object] = {}
    for target, pqs in per_model_pq.items():
        for short in model_keys:
            for long in model_keys:
                if caps[short] >= caps[long]:
                    continue
                for arm in ("section_rawspan", "chunk"):
                    key = f"{target}|{short}-vs-{long}|{arm}"
                    res = did_pooling(pqs[f"{short}/{arm}"], pqs[f"{long}/{arm}"], corpus.qids, bands)
                    if res is not None:
                        res["short_ctx"] = caps[short]
                        res["long_ctx"] = caps[long]
                        did[key] = res
    out["pooling_did"] = did
    return out


# ---------------------------------------------------------------------------
# diag: WHY raw-span loses past ~2k, and whether a dynamic chunker can see it
#
# T1 (dilution)   -- if the bottleneck is a fixed-width OUTPUT rather than the
#                    context window, the raw-span/centroid crossover must move
#                    LEFT as MRL dimension shrinks. Slicing is free.
# T2 (collapse)   -- diluted long-span vectors should be mutually MORE similar
#                    (and span fewer effective directions) than short ones.
#                    Pure geometry, no qrels.
# T3 (margin)     -- the decision-relevant consequence: does the gold section
#                    still stand above its own contract's pool? Measured as a
#                    z-score of the gold cosine against the in-scope pool.
# T4 (dynamic)    -- Tom's dynamic-chunking hypothesis. Does span COHERENCE
#                    (agreement of the member chunk vectors) predict whether
#                    raw-span or centroid wins for that span -- and does it add
#                    anything over just knowing the span's LENGTH? If coherence
#                    survives as a partial, a dynamic splitter beats a length
#                    rule; if not, fixed-length chunking with a cutoff is enough.
# ---------------------------------------------------------------------------


def _slice(v: np.ndarray, d: int) -> np.ndarray:
    """MRL truncate + renormalise (prefix slice is the whole trick)."""
    return _unit(v[:, :d].astype(np.float32))


def _section_scopes(corpus: Corpus) -> Dict[str, List[int]]:
    """Row indices of ``units.section_ids`` grouped by retrieval scope."""
    groups: Dict[str, List[int]] = {}
    for i, sid in enumerate(corpus.units.section_ids):
        groups.setdefault(ed._unit_scope(sid), []).append(i)
    return groups


def _participation_ratio(x: np.ndarray) -> float:
    """Effective number of directions spanned (PR = (tr S)^2 / ||S||_F^2)."""
    if x.shape[0] < 2:
        return float("nan")
    xc = x - x.mean(axis=0, keepdims=True)
    s = xc.T @ xc
    fro = float((s * s).sum())
    return float(np.trace(s) ** 2 / fro) if fro > 0 else float("nan")


def _mean_pairwise_cos(x: np.ndarray) -> float:
    """Mean off-diagonal cosine of unit-normalised rows."""
    if x.shape[0] < 2:
        return float("nan")
    u = _unit(x.astype(np.float64))
    g = u @ u.T
    n = g.shape[0]
    return float((g.sum() - np.trace(g)) / (n * (n - 1)))


def _geometry_by_band(
    vecs: np.ndarray, sec_band: List[str], scopes: Dict[str, List[int]]
) -> Dict[str, Dict[str, float]]:
    """Within-scope mutual similarity + effective rank, per length band.

    Within-scope because that is the pool retrieval actually discriminates
    against (MAUD queries are scoped to their own contract); a global figure
    would mostly measure how similar all merger agreements are to each other.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, _, _ in BANDS:
        cosines: List[float] = []
        prs: List[float] = []
        for rows in scopes.values():
            sel = [i for i in rows if sec_band[i] == name]
            if len(sel) < 3:
                continue
            block = vecs[sel]
            cosines.append(_mean_pairwise_cos(block))
            prs.append(_participation_ratio(_unit(block.astype(np.float64))))
        out[name] = {
            "mean_pairwise_cos": float(np.nanmean(cosines)) if cosines else float("nan"),
            "participation_ratio": float(np.nanmean(prs)) if prs else float("nan"),
            "n_scopes": len(cosines),
        }
    return out


def _gold_margin_by_band(
    vecs: np.ndarray,
    queries: np.ndarray,
    corpus: Corpus,
    qband: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """z-score of the gold section's cosine against its own in-scope pool."""
    sims = _unit(queries) @ _unit(vecs).T
    sid_row = {sid: i for i, sid in enumerate(corpus.units.section_ids)}
    scopes = _section_scopes(corpus)
    qrels = corpus.qrels_by_target["section"]

    per_band: Dict[str, List[float]] = {name: [] for name, _, _ in BANDS}
    raw_gold: Dict[str, List[float]] = {name: [] for name, _, _ in BANDS}
    for qi, qid in enumerate(corpus.qids):
        band = qband.get(qid, "unknown")
        if band not in per_band:
            continue
        gold = [sid_row[s] for s, r in qrels.get(qid, {}).items() if r > 0 and s in sid_row]
        if not gold:
            continue
        pool = scopes.get(ed.SCOPE_QID[qid], []) if ed.SCOPE_QID else list(range(len(corpus.units.section_ids)))
        if len(pool) < 3:
            continue
        pv = sims[qi, pool]
        sd = float(pv.std())
        g = float(sims[qi, gold].mean())
        raw_gold[band].append(g)
        per_band[band].append((g - float(pv.mean())) / (sd if sd > 0 else 1.0))
    return {
        name: {
            "gold_z_vs_pool": float(np.mean(per_band[name])) if per_band[name] else float("nan"),
            "gold_cos": float(np.mean(raw_gold[name])) if raw_gold[name] else float("nan"),
            "n": len(per_band[name]),
        }
        for name, _, _ in BANDS
    }


def _dynamic_criterion(
    pq_raw: np.ndarray,
    pq_cent: np.ndarray,
    cstats: Centroids,
    raw_vecs: np.ndarray,
    corpus: Corpus,
    qband: Dict[str, str],
) -> Dict[str, object]:
    """Does span coherence predict raw-span-vs-centroid, beyond span length?

    Outcome per query: nDCG(raw-span) - nDCG(centroid), attributed to the
    query's longest gold section. Predictors are all label-free and available at
    ingest: coherence (member-vector agreement), drift (how far the whole-span
    embedding sits from the centroid), length, and chunk count.
    """
    sid_row = {sid: i for i, sid in enumerate(corpus.units.section_ids)}
    tokens = [_estimate_tokens(t) for t in corpus.units.section_texts]
    cent_u = _unit(cstats.vectors.astype(np.float64))
    raw_u = _unit(raw_vecs.astype(np.float64))
    drift = 1.0 - np.einsum("ij,ij->i", cent_u, raw_u)  # 0 = identical, 2 = opposed
    qrels = corpus.qrels_by_target["section"]

    y: List[float] = []
    cols: Dict[str, List[float]] = {"coherence": [], "drift": [], "length": [], "n_chunks": []}
    for qi, qid in enumerate(corpus.qids):
        if qband.get(qid) == "unknown":
            continue
        gold = [sid_row[s] for s, r in qrels.get(qid, {}).items() if r > 0 and s in sid_row]
        gold = [i for i in gold if not cstats.empty[i]]
        if not gold:
            continue
        g = max(gold, key=lambda i: tokens[i])
        y.append(float(pq_raw[qi] - pq_cent[qi]))
        cols["coherence"].append(float(cstats.coherence[g]))
        cols["drift"].append(float(drift[g]))
        cols["length"].append(float(tokens[g]))
        cols["n_chunks"].append(float(cstats.n_chunks[g]))

    if len(y) < 30:
        return {"n": len(y)}
    out: Dict[str, object] = {"n": len(y), "predictors": {}}
    for name, x in cols.items():
        entry: Dict[str, object] = {
            "spearman": ed._spearman(x, y),
            "boot": ed._spearman_boot(x, y),
        }
        if name != "length":
            # The question that decides dynamic-vs-fixed chunking: does this
            # signal still predict once length is held constant?
            entry["partial_given_length"] = ed._partial_spearman(x, y, cols["length"])
            entry["partial_boot"] = ed._partial_boot(x, y, cols["length"])
        out["predictors"][name] = entry  # type: ignore[index]

    # Practical read: split the corpus on the median of each signal and report
    # the raw-span advantage on each side. A usable splitter needs opposite signs.
    splits: Dict[str, object] = {}
    yv = np.asarray(y)
    for name, x in cols.items():
        xv = np.asarray(x)
        med = float(np.median(xv))
        hi, lo = xv > med, xv <= med
        if hi.sum() >= 10 and lo.sum() >= 10:
            splits[name] = {
                "median": med,
                "raw_minus_centroid_above": float(yv[hi].mean()),
                "raw_minus_centroid_below": float(yv[lo].mean()),
                "separation": float(yv[hi].mean() - yv[lo].mean()),
            }
    out["median_splits"] = splits
    return out


def run_diag(corpus: Corpus, model_keys: List[str], allow_embed: bool) -> Dict[str, object]:
    qband, band_counts = query_bands(corpus)
    tokens = [_estimate_tokens(t) for t in corpus.units.section_texts]
    sec_band = [band_of(t) for t in tokens]
    scopes = _section_scopes(corpus)
    target = "section" if "section" in ed.TARGETS else ed.TARGETS[0]

    out: Dict[str, object] = {"query_band_counts": band_counts, "target": target, "models": {}}
    for mkey in model_keys:
        logger.info("=== %s ===", mkey)
        spec = ed.MODEL_POOL[mkey]
        vec = load_vectors(mkey, corpus.units, allow_embed)
        cstats = section_centroids(vec.chunks, corpus.units)

        # T1 -- crossover vs MRL dimension.
        ladder: Dict[str, object] = {}
        for d in spec.mrl_dims:
            q_d = _slice(vec.queries, d)
            raw = _arm("section", target, _slice(vec.sections, d), q_d, corpus)
            cen = _arm("section", target, _slice(cstats.vectors, d), q_d, corpus)
            ladder[str(d)] = {
                "rawspan": raw.mean_ndcg,
                "centroid": cen.mean_ndcg,
                "delta_overall": raw.mean_ndcg - cen.mean_ndcg,
                "delta_banded": {
                    name: (
                        _band_means(raw.pq_ndcg, corpus.qids, qband)[name]
                        - _band_means(cen.pq_ndcg, corpus.qids, qband)[name]
                    )
                    for name, _, _ in BANDS
                },
            }

        # T2/T3 at full dim.
        geom = {
            "rawspan": _geometry_by_band(vec.sections, sec_band, scopes),
            "centroid": _geometry_by_band(cstats.vectors, sec_band, scopes),
        }
        margin = {
            "rawspan": _gold_margin_by_band(vec.sections, vec.queries, corpus, qband),
            "centroid": _gold_margin_by_band(cstats.vectors, vec.queries, corpus, qband),
        }

        # T4 -- dynamic-chunking criterion.
        raw_full = _arm("section", target, vec.sections, vec.queries, corpus)
        cen_full = _arm("section", target, cstats.vectors, vec.queries, corpus)
        dyn = _dynamic_criterion(raw_full.pq_ndcg, cen_full.pq_ndcg, cstats, vec.sections, corpus, qband)

        out["models"][mkey] = {  # type: ignore[index]
            "model": spec.model,
            "num_ctx": spec.num_ctx,
            "mrl_ladder": ladder,
            "geometry": geom,
            "gold_margin": margin,
            "dynamic": dyn,
            "coherence_by_band": {
                name: float(
                    np.mean(
                        [
                            cstats.coherence[i]
                            for i in range(len(sec_band))
                            if sec_band[i] == name and not cstats.empty[i]
                        ]
                    )
                )
                for name, _, _ in BANDS
            },
        }
        del vec
    return out


# ---------------------------------------------------------------------------
# confound: is the crossover an artifact of chunkless sections?
#
# A centroid needs member chunks. Under midpoint assignment 1,494 / 4,709 MAUD
# sections (32%) own none, so they get a zero vector and sink to the bottom of
# the centroid ranking -- which silently DELETES 32% of the distractor pool for
# the centroid arm only. Those sections are short (shorter than half a chunk),
# so the deletion helps whenever the gold is long and hurts when the gold is
# itself short -- exactly the shape of the reported crossover.
#
# The control: restrict BOTH arms to sections owning >=1 chunk (and drop the
# queries whose gold is entirely chunkless), so the two arms rank an identical
# candidate set. If the crossover survives, it is a representation effect. If it
# collapses, it was a bookkeeping artifact of the chunk-to-section assignment.
# ---------------------------------------------------------------------------


def run_confound(corpus: Corpus, model_keys: List[str], allow_embed: bool) -> Dict[str, object]:
    qband, _ = query_bands(corpus)
    target = "section" if "section" in ed.TARGETS else ed.TARGETS[0]
    qrels_full = corpus.qrels_by_target[target]

    out: Dict[str, object] = {"models": {}}
    for mkey in model_keys:
        vec = load_vectors(mkey, corpus.units, allow_embed)
        cstats = section_centroids(vec.chunks, corpus.units)
        keep = ~cstats.empty
        kept_ids = {sid for sid, k in zip(corpus.units.section_ids, keep, strict=True) if k}

        # Restricted pool: only sections a centroid can actually represent.
        sub_ids = [sid for sid, k in zip(corpus.units.section_ids, keep, strict=True) if k]
        pooler_k = ed.Pooler(sub_ids)
        qrels_k = {q: {s: r for s, r in d.items() if s in kept_ids} for q, d in qrels_full.items()}
        qids_k = [q for q in corpus.qids if any(r > 0 for r in qrels_k.get(q, {}).values())]
        keep_q = np.array([q in set(qids_k) for q in corpus.qids])

        variants: Dict[str, object] = {}
        for label, (ids, pooler, qrels, qids, rows) in {
            "full_pool": (
                corpus.units.section_ids,
                corpus.poolers[("section", target)],
                qrels_full,
                corpus.qids,
                slice(None),
            ),
            "chunked_only": (sub_ids, pooler_k, qrels_k, qids_k, keep),
        }.items():
            qsel = _unit(vec.queries) if label == "full_pool" else _unit(vec.queries)[keep_q]
            res: Dict[str, np.ndarray] = {}
            for arm, mat in (("rawspan", vec.sections), ("centroid", cstats.vectors)):
                sims = (qsel @ _unit(mat[rows]).T).astype(np.float32)
                pooled = pooler.pool(sims)
                pq, _ = ed.score_arm(pooler.units, pooled, qids, qrels)
                res[arm] = pq
            band_sub = {q: qband[q] for q in qids}
            variants[label] = {
                "n_queries": len(qids),
                "n_sections": len(set(ids)),
                "rawspan": float(res["rawspan"].mean()),
                "centroid": float(res["centroid"].mean()),
                "delta_overall": float((res["rawspan"] - res["centroid"]).mean()),
                "delta_banded": {
                    name: (
                        _band_means(res["rawspan"], qids, band_sub)[name]
                        - _band_means(res["centroid"], qids, band_sub)[name]
                    )
                    for name, _, _ in BANDS
                },
                "boot": ed.paired_bootstrap(res["rawspan"], res["centroid"]),
            }
        out["models"][mkey] = {  # type: ignore[index]
            "model": ed.MODEL_POOL[mkey].model,
            "chunkless_sections": int(cstats.empty.sum()),
            "variants": variants,
        }
        del vec
    return out


def print_confound(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("CONFOUND CHECK: does the crossover survive an identical candidate pool?")
    print("  full_pool    = every section (centroid arm silently loses chunkless ones)")
    print("  chunked_only = both arms restricted to sections owning >=1 chunk")
    print("=" * 78)
    for mkey, m in res["models"].items():
        print(f"\n--- {mkey} ({m['model']}) — {m['chunkless_sections']} chunkless sections ---")
        for label, v in m["variants"].items():
            line = "  %-13s raw %.4f  cent %.4f  delta %+.4f [%+.4f,%+.4f]" % (
                label,
                v["rawspan"],
                v["centroid"],
                v["boot"]["delta"],
                v["boot"]["ci_lo"],
                v["boot"]["ci_hi"],
            )
            print(line + "   (nq=%d, nsec=%d)" % (v["n_queries"], v["n_sections"]))
            bands = "                by band: " + "  ".join(
                "%s %+.4f" % (name, v["delta_banded"][name])
                for name, _, _ in BANDS
                if v["delta_banded"][name] == v["delta_banded"][name]
            )
            print(bands)


# ---------------------------------------------------------------------------
# chunks: where does shrinking the retrieval unit stop paying?
#
# Chunk size and MRL dimension are the two sides of one ratio -- tokens per
# stored vector -- so if the dilution account is right, the chunk-size curve and
# the section-LENGTH curve should be the SAME curve. That superposition is the
# falsifiable claim: a 2k-token chunk should retrieve about as well as a
# 2k-token section. If the two curves separate, span length is not the whole
# story and semantic boundaries carry independent value.
#
# Reported against a VECTOR BUDGET, because halving chunk size doubles the index
# and quality-per-vector is the actual deployment question ("diminishing
# returns" is only meaningful per unit of cost).
#
# This is the one analysis here that NEEDS embedding: a different chunking is
# different text, so the cache cannot serve it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkArm:
    """One chunker configuration, as ``method[:overlap][*scale]``.

    ``scale`` multiplies the nominal chunk size for THIS arm only. A chunker that
    cuts at the strongest boundary inside a budget systematically under-fills it,
    so equal ``max_tokens`` does not mean equal spans; the scale calibrates arms
    onto equal median tokens while the sweep still pairs them at the same nominal
    size. Without it a "win" could be nothing but the smaller-chunks effect.
    """

    label: str
    method: Optional[str]
    overlap: Optional[int]
    scale: float = 1.0

    def size_for(self, nominal: int) -> int:
        return max(1, int(round(nominal * self.scale)))


_CONTRACT_HEADINGS: Dict[int, List[int]] = {}


def contract_heading_finder(text: str) -> List[int]:
    """Clause-heading cut positions for plain-text contracts.

    Reuses `maud_data.detect_contract_sections` rather than widening
    `StructureChunker._HEADING`: a bare regex for `Section N.N` fires on every
    cross-reference and on the table of contents, which repeats every heading.
    That detector's gap test, TOC-run drop and LIS monotonicity filter are what
    make the positions usable, and they are corpus knowledge, not chunker logic.

    NOTE the circularity risk: these are the same boundaries MAUD's section
    qrels are built from, so chunks will align with gold sections *by
    construction*. Read any win here against the gold-span split test, not
    against section attribution alone.
    """
    from benchmarks.maud_data import detect_contract_sections

    key = hash(text)
    hit = _CONTRACT_HEADINGS.get(key)
    if hit is None:
        hit = [s.start_pos for s in detect_contract_sections(text) if s.start_pos > 0]
        _CONTRACT_HEADINGS[key] = hit
    return hit


HEADING_FINDERS = {"none": None, "contract": contract_heading_finder}


def parse_arms(spec: Optional[str]) -> List[ChunkArm]:
    """``"sentences:1,structure:0*1.2"`` -> arms; empty spec keeps the shipped default."""
    if not spec or not spec.strip():
        return [ChunkArm("default", None, None)]
    arms: List[ChunkArm] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        head, sep, scale = tok.partition("*")
        method, _, ov = head.partition(":")
        arms.append(ChunkArm(tok, method.strip(), int(ov) if ov.strip() else None, float(scale) if sep else 1.0))
    return arms


def run_chunks(
    corpus: Corpus,
    model_keys: List[str],
    sizes: List[int],
    arms: List[ChunkArm],
    allow_embed: bool,
    heading_finder: Optional[Callable[[str], Iterable[int]]] = None,
) -> Dict[str, object]:
    qband, band_counts = query_bands(corpus)
    target = "section" if "section" in ed.TARGETS else ed.TARGETS[0]
    detector = _section_detector(corpus)

    out: Dict[str, object] = {
        "query_band_counts": band_counts,
        "target": target,
        "sizes": sizes,
        "arms": [a.label for a in arms],
        "heading_finder": getattr(heading_finder, "__name__", None),
        "models": {},
    }
    for mkey in model_keys:
        spec = ed.MODEL_POOL[mkey]
        by_arm: Dict[str, object] = {}
        # Per-query scores kept so arms can be compared on the SAME queries at the
        # SAME size -- the only comparison that isolates where the cuts land.
        pq_by: Dict[Tuple[str, int], np.ndarray] = {}
        for arm in arms:
            rows: Dict[str, object] = {}
            for ck in sizes:
                actual = arm.size_for(ck)
                logger.info("=== %s @ chunk_tokens=%d (arm size %d) [%s] ===", mkey, ck, actual, arm.label)
                # The heading finder only means anything to the structure chunker.
                ckw = {"heading_finder": heading_finder} if (heading_finder and arm.method == "structure") else None
                units = ed.load_units(corpus.bench, actual, detector, arm.method, arm.overlap, ckw)
                vec = load_vectors(mkey, units, allow_embed)
                pooler = ed.Pooler(units.chunk_section if target == "section" else units.chunk_doc)
                sims = (_unit(vec.queries) @ _unit(vec.chunks).T).astype(np.float32)
                pooled = pooler.pool(sims)
                pq, rec = ed.score_arm(pooler.units, pooled, corpus.qids, corpus.qrels_by_target[target])
                pq_by[(arm.label, ck)] = pq
                toks = [_estimate_tokens(t) for t in units.chunk_texts]
                rows[str(ck)] = {
                    "arm_chunk_tokens": actual,
                    "ndcg": float(pq.mean()),
                    "recall10": float(rec.mean()),
                    "n_vectors": len(units.chunk_texts),
                    "mean_tokens": float(np.mean(toks)),
                    "median_tokens": float(np.median(toks)),
                    # Fraction of chunks straddling a section boundary: the §6.1
                    # attribution artifact, measured rather than argued about.
                    "crosses_rate": float(np.mean(units.chunk_crosses)) if units.chunk_crosses else float("nan"),
                    "banded": _band_means(pq, corpus.qids, qband),
                }
                del vec
            # Marginal return per doubling of the index, the "diminishing returns" read.
            ordered = sorted(sizes, reverse=True)
            marginal: Dict[str, object] = {}
            for prev, cur in zip(ordered, ordered[1:], strict=False):
                p, c = rows[str(prev)], rows[str(cur)]
                extra = c["n_vectors"] - p["n_vectors"]  # type: ignore[operator]
                marginal[f"{prev}->{cur}"] = {
                    "d_ndcg": c["ndcg"] - p["ndcg"],  # type: ignore[operator]
                    "extra_vectors": extra,
                    "ndcg_per_1k_extra_vectors": (
                        float((c["ndcg"] - p["ndcg"]) / (extra / 1000.0)) if extra else None  # type: ignore[operator]
                    ),
                }
            by_arm[arm.label] = {"by_size": rows, "marginal": marginal}
        # Every arm against the first, paired per query and matched on size.
        base = arms[0].label
        vs: Dict[str, object] = {}
        for arm in arms[1:]:
            vs[arm.label] = {str(ck): ed.paired_bootstrap(pq_by[(arm.label, ck)], pq_by[(base, ck)]) for ck in sizes}
        entry: Dict[str, object] = {"model": spec.model, "arms": by_arm, "vs_first_arm": {"base": base, "deltas": vs}}
        if len(arms) == 1:  # keep the single-arm JSON shape the earlier sweeps wrote
            entry["by_size"] = by_arm[arms[0].label]["by_size"]  # type: ignore[index]
            entry["marginal"] = by_arm[arms[0].label]["marginal"]  # type: ignore[index]
        out["models"][mkey] = entry  # type: ignore[index]
    return out


def print_chunks(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("CHUNK SIZE: quality vs vector budget, and where returns stop")
    print("=" * 88)
    for mkey, m in res["models"].items():
        # Sweeps written before the chunker-comparison arms have no "arms" key.
        arms = m.get("arms") or {"default": {"by_size": m["by_size"], "marginal": m["marginal"]}}
        for label, a in arms.items():
            print(f"\n--- {mkey} ({m['model']})  chunker={label} ---")
            hdr = "  %8s %8s %10s %9s %8s" % ("chunk", "nDCG@10", "vectors", "med_tok", "straddle")
            for name, _, _ in BANDS:
                hdr += " %9s" % name
            print(hdr)
            for ck, r in a["by_size"].items():
                cross = r.get("crosses_rate", float("nan"))
                actual = r.get("arm_chunk_tokens")
                shown = ck if actual in (None, int(ck)) else "%s(%s)" % (ck, actual)
                line = "  %8s %8.4f %10d %9.0f %8s" % (
                    shown,
                    r["ndcg"],
                    r["n_vectors"],
                    r["median_tokens"],
                    "--" if cross != cross else "%.1f%%" % (100 * cross),
                )
                for name, _, _ in BANDS:
                    v = r["banded"][name]
                    line += " %9s" % ("--" if v != v else "%.4f" % v)
                print(line)
            print("\n  marginal return per index growth:")
            for step, mm in a["marginal"].items():
                per = mm["ndcg_per_1k_extra_vectors"]
                print(
                    "    %-14s dnDCG %+.4f   +%d vectors   %s"
                    % (step, mm["d_ndcg"], mm["extra_vectors"], "--" if per is None else "%+.5f / 1k vectors" % per)
                )
        vs = m.get("vs_first_arm")
        if vs and vs["deltas"]:
            print(f"\n  paired vs {vs['base']} (same queries, matched chunk size):")
            print("    %-18s %8s %9s %18s %7s" % ("arm", "size", "dnDCG", "95% CI", "p_win"))
            for label, by_size in vs["deltas"].items():
                for ck, b in by_size.items():
                    print(
                        "    %-18s %8s %+9.4f  [%+.4f,%+.4f] %7.2f"
                        % (label, ck, b["delta"], b["ci_lo"], b["ci_hi"], b["p_win"])
                    )


# ---------------------------------------------------------------------------
# route: per-query weighting against the global-weight bar
# ---------------------------------------------------------------------------


def _topk_stats(m: np.ndarray, k: int = TOPK_SIGNAL) -> Dict[str, np.ndarray]:
    """Label-free per-query confidence signals from a scoped score matrix."""
    filled = np.where(np.isnan(m), -np.inf, m)
    part = np.sort(filled, axis=1)[:, ::-1][:, :k]
    top1 = part[:, 0]
    top2 = np.where(part.shape[1] > 1, part[:, min(1, part.shape[1] - 1)], part[:, 0])
    finite = np.where(np.isfinite(part), part, np.nan)
    ex = np.exp(finite - np.nanmax(finite, axis=1, keepdims=True))
    p = ex / np.nansum(ex, axis=1, keepdims=True)
    ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return {
        "top1": top1,
        "margin": top1 - top2,
        "entropy": ent,
        "spread": np.nanstd(finite, axis=1),
    }


def _topk_sets(m: np.ndarray, k: int = TOPK_SIGNAL) -> np.ndarray:
    filled = np.where(np.isnan(m), -np.inf, m)
    return np.argsort(-filled, axis=1)[:, :k]


def router_signals(ma: np.ndarray, mb: np.ndarray, corpus: Corpus, k: int = TOPK_SIGNAL) -> Dict[str, np.ndarray]:
    """Signals available at query time with no labels and no second retrieval pass."""
    sa, sb = _topk_stats(ma, k), _topk_stats(mb, k)
    ta, tb = _topk_sets(ma, k), _topk_sets(mb, k)
    agree = np.array([len(set(ta[i]) & set(tb[i])) / k for i in range(ta.shape[0])], dtype=np.float64)
    qlen = np.array([_estimate_tokens(t) for t in corpus.units.query_texts], dtype=np.float64)
    # Oriented so that LARGER = "prefer the section leg" (larger w), matching the
    # sign convention of the weight grid.
    return {
        "d_margin": sb["margin"] - sa["margin"],
        "d_top1": sb["top1"] - sa["top1"],
        "d_entropy": sa["entropy"] - sb["entropy"],
        "d_spread": sb["spread"] - sa["spread"],
        "agreement": agree,
        "qlen": qlen,
    }


def _pq_by_weight(
    a: ed.ArmScores, b: ed.ArmScores, corpus: Corpus, target: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Per-query nDCG at every grid weight, plus the two scoped min-max matrices.

    Returns ``(pq_by_w, mm_a, mm_b, union)`` where ``pq_by_w[i]`` is the
    per-query nDCG at ``ALPHAS[i]`` (the SECTION-leg weight), so row 0 is the
    chunk arm alone and row -1 is the section arm alone.
    """
    union, ma, mb = ed._union_fill(a.units, a.mat, b.units, b.mat)
    ma, mb = ed._scope_nan((ma, mb), corpus.qids, union)
    mm_a, mm_b = ed._minmax_rows(ma), ed._minmax_rows(mb)
    _, _, pqs = ed._sweep(mm_a, mm_b, union, corpus.qids, corpus.qrels_by_target[target])
    return np.vstack(pqs), ma, mb, union


def _fit_adaptive(pq_by_w: np.ndarray, sig: np.ndarray, train: np.ndarray) -> Tuple[float, float, float, float]:
    """Pick (w0, gain) maximising train mean nDCG; z-stats come from train only."""
    mu, sd = float(sig[train].mean()), float(sig[train].std())
    sd = sd if sd > 0 else 1.0
    z = (sig - mu) / sd
    best = (float(ed.ALPHAS[int(np.argmax(pq_by_w[:, train].mean(axis=1)))]), 0.0, -1.0)
    for w0 in ed.ALPHAS:
        for g in ROUTER_GAINS:
            idx = _quantise(w0 + g * z)
            score = float(pq_by_w[idx, np.arange(len(sig))][train].mean())
            if score > best[2]:
                best = (float(w0), float(g), score)
    return best[0], best[1], mu, sd


def _quantise(w: np.ndarray) -> np.ndarray:
    """Snap continuous weights onto the ALPHAS grid (indices into pq_by_w)."""
    clipped = np.clip(w, ed.ALPHAS[0], ed.ALPHAS[-1])
    return np.abs(clipped[:, None] - ed.ALPHAS[None, :]).argmin(axis=1)


def _folds(n: int, k: int = CV_FOLDS) -> List[np.ndarray]:
    rng = np.random.default_rng(ed.BOOTSTRAP_SEED)
    order = rng.permutation(n)
    return [np.sort(order[i::k]) for i in range(k)]


def run_route(corpus: Corpus, pairs: List[Tuple[str, str]], allow_embed: bool) -> Dict[str, object]:
    needed = sorted({m for pair in pairs for m in pair})
    vecs = {m: load_vectors(m, corpus.units, allow_embed) for m in needed}

    out: Dict[str, object] = {"folds": CV_FOLDS, "gains": list(ROUTER_GAINS), "targets": {}}
    for target in ed.TARGETS:
        singles = {
            m: {
                "chunk": _arm("chunk", target, v.chunks, v.queries, corpus),
                "section": _arm("section", target, v.sections, v.queries, corpus),
            }
            for m, v in vecs.items()
        }
        tgt: Dict[str, object] = {}
        for cm, sm in pairs:
            a, b = singles[cm]["chunk"], singles[sm]["section"]
            pq_by_w, ma, mb, _ = _pq_by_weight(a, b, corpus, target)
            nq = pq_by_w.shape[1]
            sigs = router_signals(ma, mb, corpus)

            means = pq_by_w.mean(axis=1)
            gi = int(np.argmax(means))
            ceilings = {
                "arm_chunk": float(means[0]),
                "arm_section": float(means[-1]),
                "global_best_w": float(ed.ALPHAS[gi]),
                "global_best": float(means[gi]),
                # Ceiling for arm SELECTION (the P0-A oracle) ...
                "oracle_arm": float(np.maximum(pq_by_w[0], pq_by_w[-1]).mean()),
                # ... and for per-query WEIGHTING, the ceiling this thread targets.
                "oracle_weight": float(pq_by_w.max(axis=0).mean()),
            }

            folds = _folds(nq)
            routers: Dict[str, object] = {}
            # CV'd global weight: the honest bar (fitted per fold, like the routers).
            g_held = np.zeros(nq)
            for f in folds:
                train = np.setdiff1d(np.arange(nq), f)
                g_held[f] = pq_by_w[int(np.argmax(pq_by_w[:, train].mean(axis=1))), f]
            routers["global_w_cv"] = {"held_out_ndcg": float(g_held.mean())}

            for name, sig in sigs.items():
                held = np.zeros(nq)
                params: List[Dict[str, float]] = []
                for f in folds:
                    train = np.setdiff1d(np.arange(nq), f)
                    w0, g, mu, sd = _fit_adaptive(pq_by_w, sig, train)
                    idx = _quantise(w0 + g * ((sig - mu) / sd))
                    held[f] = pq_by_w[idx[f], f]
                    params.append({"w0": w0, "gain": g})
                gap = ceilings["oracle_weight"] - g_held.mean()
                routers[name] = {
                    "held_out_ndcg": float(held.mean()),
                    "vs_global_cv": ed.paired_bootstrap(held, g_held),
                    "capture_of_weight_oracle": (float((held.mean() - g_held.mean()) / gap) if gap > 1e-9 else None),
                    "folds": params,
                    "picked_nonzero_gain": sum(1 for p in params if p["gain"] > 0),
                }
            tgt[f"{cm}/chunk+{sm}/section"] = {"ceilings": ceilings, "routers": routers}
        out["targets"][target] = tgt  # type: ignore[index]
    return out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def print_hier(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("SECTION REPRESENTATION: raw-span vs centroid, banded by gold-section length")
    print("=" * 78)
    print(f"\nsections per band: {res['section_band_counts']}")
    print(f"queries per band:  {res['query_band_counts']}")

    for mkey, m in res["models"].items():
        print(f"\n--- {mkey} ({m['model']}, ctx={m['num_ctx']}) ---")
        print(
            "  chunkless sections: %d (gold: %d/%d)"
            % (m["chunkless_sections"], m["chunkless_gold_sections"], m["gold_sections"])
        )
        for target, t in m["targets"].items():
            print(f"\n  target={target}   nDCG@10 overall / by band")
            hdr = "    %-18s %7s" % ("arm", "all")
            for name, _, _ in BANDS:
                hdr += " %8s" % name
            print(hdr)
            for arm, overall in t["overall"].items():
                line = "    %-18s %7.4f" % (arm, overall)
                for name, _, _ in BANDS:
                    v = t["banded"][arm][name]
                    line += " %8s" % ("--" if v != v else "%.4f" % v)
                print(line)
            b = t["boot"]["rawspan_vs_centroid"]
            print(
                "    F2 rawspan-centroid: %+.4f [%+.4f,%+.4f] p(win)=%.2f   (best w: raw %.2f / centroid %.2f)"
                % (
                    b["delta"],
                    b["ci_lo"],
                    b["ci_hi"],
                    b["p_win"],
                    t["best_weight"]["fused_rawspan"],
                    t["best_weight"]["fused_centroid"],
                )
            )
            for name, _, _ in BANDS:
                bb = t["boot_banded_rawspan_vs_centroid"].get(name)
                if bb:
                    print(
                        "      band %-6s %+.4f [%+.4f,%+.4f] p(win)=%.2f"
                        % (name, bb["delta"], bb["ci_lo"], bb["ci_hi"], bb["p_win"])
                    )

    if res["pooling_did"]:
        print("\n" + "-" * 78)
        print("CONTEXT-LENGTH NATURAL EXPERIMENT (difference-in-differences)")
        print("  short-ctx minus long-ctx model, in the band where only the short model pools")
        print("  vs the band where neither does. Negative DiD = window pooling costs quality.")
        print("-" * 78)
        for key, d in res["pooling_did"].items():
            print(
                "  %-40s gap<=2k %+.4f  gap2k-8k %+.4f  DiD %+.4f [%+.4f,%+.4f] p(neg)=%.2f  (n=%d/%d)"
                % (
                    key,
                    d["gap_inner"],
                    d["gap_outer"],
                    d["did"],
                    d["ci_lo"],
                    d["ci_hi"],
                    d["p_negative"],
                    d["n_inner"],
                    d["n_outer"],
                )
            )


def print_diag(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("WHY RAW-SPAN LOSES PAST ~2k  (T1 dilution / T2 collapse / T3 margin / T4 dynamic)")
    print("=" * 78)
    for mkey, m in res["models"].items():
        print(f"\n--- {mkey} ({m['model']}, ctx={m['num_ctx']}) ---")

        print("\n  T1  rawspan - centroid, by MRL output dimension")
        print("      (dilution predicts the advantage ERODES as dims shrink)")
        hdr = "      %6s %9s" % ("dim", "overall")
        for name, _, _ in BANDS:
            hdr += " %9s" % name
        print(hdr)
        for d, row in m["mrl_ladder"].items():
            line = "      %6s %+9.4f" % (d, row["delta_overall"])
            for name, _, _ in BANDS:
                v = row["delta_banded"][name]
                line += " %9s" % ("--" if v != v else "%+.4f" % v)
            print(line)

        print("\n  T2  within-scope geometry (mean pairwise cos / effective directions)")
        print("      %-10s %-24s %-24s" % ("band", "rawspan", "centroid"))
        for name, _, _ in BANDS:
            r = m["geometry"]["rawspan"][name]
            c = m["geometry"]["centroid"][name]
            print(
                "      %-10s cos %.4f  PR %6.1f   cos %.4f  PR %6.1f   (n=%d)"
                % (
                    name,
                    r["mean_pairwise_cos"],
                    r["participation_ratio"],
                    c["mean_pairwise_cos"],
                    c["participation_ratio"],
                    r["n_scopes"],
                )
            )

        print("\n  T3  gold section vs its own pool (z-score; higher = still findable)")
        print("      %-10s %-22s %-22s" % ("band", "rawspan", "centroid"))
        for name, _, _ in BANDS:
            r = m["gold_margin"]["rawspan"][name]
            c = m["gold_margin"]["centroid"][name]
            print(
                "      %-10s z %+6.3f  cos %.3f    z %+6.3f  cos %.3f    (n=%d)"
                % (name, r["gold_z_vs_pool"], r["gold_cos"], c["gold_z_vs_pool"], c["gold_cos"], r["n"])
            )

        print(
            "\n  coherence (‖mean of member unit vectors‖) by band: %s"
            % {k: round(v, 4) for k, v in m["coherence_by_band"].items()}
        )

        dyn = m["dynamic"]
        if "predictors" in dyn:
            print(f"\n  T4  does a signal predict rawspan-minus-centroid? (n={dyn['n']} queries)")
            print("      %-11s %8s %10s %s" % ("signal", "rho", "partial|len", "95% CI on rho"))
            for name, p in dyn["predictors"].items():
                part = p.get("partial_given_length")
                print(
                    "      %-11s %+8.3f %10s  [%+.2f,%+.2f]"
                    % (
                        name,
                        p["spearman"],
                        "--" if part is None else "%+.3f" % part,
                        p["boot"]["ci_lo"],
                        p["boot"]["ci_hi"],
                    )
                )
            print("\n      median split -> mean(rawspan - centroid) above / below:")
            for name, s in dyn["median_splits"].items():
                print(
                    "      %-11s med %8.1f   above %+.4f   below %+.4f   sep %+.4f"
                    % (name, s["median"], s["raw_minus_centroid_above"], s["raw_minus_centroid_below"], s["separation"])
                )


def print_route(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("PER-QUERY ROUTING: can a label-free signal beat one global fusion weight?")
    print(f"  {res['folds']}-fold cross-validated; every number below is held-out.")
    print("=" * 78)
    for target, pairs in res["targets"].items():
        for pname, p in pairs.items():
            c = p["ceilings"]
            print(f"\n--- target={target}  {pname} ---")
            print(
                "  arms: chunk %.4f  section %.4f | global best w=%.2f -> %.4f"
                % (c["arm_chunk"], c["arm_section"], c["global_best_w"], c["global_best"])
            )
            print(
                "  ceilings: oracle-arm-selection %.4f | oracle-per-query-weight %.4f (headroom over global: %+.4f)"
                % (c["oracle_arm"], c["oracle_weight"], c["oracle_weight"] - c["global_best"])
            )
            base = p["routers"]["global_w_cv"]["held_out_ndcg"]
            print("  %-12s %8.4f  (CV'd global-weight bar)" % ("global_w", base))
            rows = [(n, r) for n, r in p["routers"].items() if n != "global_w_cv"]
            for name, r in sorted(rows, key=lambda kv: -kv[1]["held_out_ndcg"]):
                cap = r["capture_of_weight_oracle"]
                print(
                    "  %-12s %8.4f  %+.4f [%+.4f,%+.4f] p(win)=%.2f  capture=%s  g>0 in %d/%d folds"
                    % (
                        name,
                        r["held_out_ndcg"],
                        r["vs_global_cv"]["delta"],
                        r["vs_global_cv"]["ci_lo"],
                        r["vs_global_cv"]["ci_hi"],
                        r["vs_global_cv"]["p_win"],
                        "--" if cap is None else "%.0f%%" % (100 * cap),
                        r["picked_nonzero_gain"],
                        len(r["folds"]),
                    )
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=("qasper", "maud"), default="qasper")
    common.add_argument("--split", default="dev")
    common.add_argument("--max-papers", type=int, default=None)
    common.add_argument("--chunk-tokens", type=int, default=None)
    common.add_argument("--allow-embed", action="store_true", help="Permit embedding on cache miss (default: refuse).")
    common.add_argument("--tag", default=None)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hier", parents=[common], help="Centroid vs raw-span sections, banded by span length.")
    h.add_argument("--models", default="openai,egemma", help="Comma-separated model keys.")
    r = sub.add_parser("route", parents=[common], help="Per-query routing/weighting vs a global weight.")
    r.add_argument("--pairs", default=ed.P2_DEFAULT_PAIRS, help="chunkmodel:sectionmodel pairs, comma-separated.")
    d = sub.add_parser("diag", parents=[common], help="Why raw-span loses past ~2k + dynamic-chunking criterion.")
    d.add_argument("--models", default="openai,egemma", help="Comma-separated model keys.")
    cf = sub.add_parser("confound", parents=[common], help="Is the crossover a chunkless-section artifact?")
    cf.add_argument("--models", default="openai,egemma", help="Comma-separated model keys.")
    c = sub.add_parser("chunks", parents=[common], help="Chunk-size sweep vs vector budget (REQUIRES embedding).")
    c.add_argument("--models", default="openai", help="Comma-separated model keys.")
    c.add_argument("--chunk-sizes", default="125,250,500,1000,2000", help="Comma-separated chunk token sizes.")
    c.add_argument(
        "--chunk-methods",
        default=None,
        help="Comma-separated method[:overlap] arms, e.g. 'sentences:1,sentences:0,structure:0'. "
        "The first is the baseline every other arm is paired against. Default: shipped config.",
    )
    c.add_argument(
        "--heading-finder",
        choices=tuple(HEADING_FINDERS),
        default="none",
        help="Extra heading detector for structure arms ('contract' = MAUD clause sectionizer).",
    )
    sp = sub.add_parser("split", parents=[common], help="Does a chunker win by not splitting gold evidence?")
    sp.add_argument("--models", default="openai", help="Single model key.")
    sp.add_argument("--chunk-methods", default="sentences:0,structure:0*1.2", help="Exactly two arms: base,candidate.")
    sp.add_argument("--heading-finder", choices=tuple(HEADING_FINDERS), default="none")
    sp.add_argument(
        "--pooling",
        choices=("max", "mean"),
        default="max",
        help="Chunk->section pooling. 'mean' tests mechanisms that rest on max-pooling redundancy.",
    )
    ra = sub.add_parser("ratio", parents=[common], help="Sweep chunk:section ratio; recoverability and benefit vs it.")
    ra.add_argument("--models", default="openai", help="Single model key (only used with --score).")
    ra.add_argument("--chunk-sizes", default="125,250,500,1000", help="Comma-separated chunk token sizes.")
    ra.add_argument("--chunk-methods", default="sentences:0,structure:0*1.2", help="Arms; first is the baseline.")
    ra.add_argument("--score", action="store_true", help="Also score each cell (needs cached vectors).")
    ra.add_argument("--heading-finder", choices=tuple(HEADING_FINDERS), default="none")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("localvectordb.chunking").setLevel(logging.ERROR)
    ed._load_experiments_env()
    args = parse_args(argv)
    corpus = setup(args)

    if args.cmd in ("hier", "diag", "chunks", "confound", "split", "ratio"):
        keys = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in keys if m not in ed.MODEL_POOL]
        if unknown:
            print(f"Unknown model keys: {unknown}", file=sys.stderr)
            return 1
        if args.cmd == "hier":
            res = run_hier(corpus, keys, args.allow_embed)
            print_hier(res)
        elif args.cmd == "diag":
            res = run_diag(corpus, keys, args.allow_embed)
            print_diag(res)
        elif args.cmd == "confound":
            res = run_confound(corpus, keys, args.allow_embed)
            print_confound(res)
        elif args.cmd == "split":
            res = run_split(
                corpus,
                keys[0],
                parse_arms(args.chunk_methods),
                args.chunk_tokens or 500,
                args.split,
                args.allow_embed,
                HEADING_FINDERS[args.heading_finder],
                args.pooling,
            )
            print_split(res)
        elif args.cmd == "ratio":
            res = run_ratio(
                corpus,
                parse_arms(args.chunk_methods),
                sorted({int(s) for s in args.chunk_sizes.split(",") if s.strip()}, reverse=True),
                args.split,
                keys[0],
                args.score,
                args.allow_embed,
                HEADING_FINDERS[args.heading_finder],
            )
            print_ratio(res)
        else:
            sizes = sorted({int(s) for s in args.chunk_sizes.split(",") if s.strip()}, reverse=True)
            arms = parse_arms(args.chunk_methods)
            from localvectordb.chunking import ChunkerFactory

            bad = [a.method for a in arms if a.method and a.method not in ChunkerFactory.CHUNKERS]
            if bad:
                print(f"Unknown chunking methods: {bad}", file=sys.stderr)
                return 1
            res = run_chunks(corpus, keys, sizes, arms, args.allow_embed, HEADING_FINDERS[args.heading_finder])
            print_chunks(res)
        payload: Dict[str, object] = {"analysis": args.cmd, "models": keys, "result": res}
    else:
        pairs = []
        for tok in args.pairs.split(","):
            tok = tok.strip()
            if not tok:
                continue
            cm, _, sm = tok.partition(":")
            if cm not in ed.MODEL_POOL or sm not in ed.MODEL_POOL:
                print(f"Unknown pair: {tok}", file=sys.stderr)
                return 1
            pairs.append((cm, sm))
        res = run_route(corpus, pairs, args.allow_embed)
        print_route(res)
        payload = {"analysis": "route", "pairs": [f"{a}:{b}" for a, b in pairs], "result": res}

    payload.update(
        {
            "schema": 1,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": {
                "name": args.dataset,
                "split": args.split,
                "docs": len(corpus.bench.corpus),
                "queries": len(corpus.qids),
                "chunks": len(corpus.units.chunk_texts),
                "sections": len(corpus.units.section_texts),
            },
            "chunk_tokens": args.chunk_tokens or 500,
        }
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    tag = args.tag or args.cmd
    out = RESULTS_DIR / f"levels_{tag}_{args.dataset}_{args.split}_{len(corpus.bench.corpus)}p_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
