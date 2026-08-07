"""DIAGNOSTIC: what would a `sections_fts` actually buy? Measure before building it.

THE QUESTION. `search_level="sections"` and `"fused"` have no keyword leg -- FTS5
indexes chunks only, and the `sections` table stores no text, just start/end
offsets into its parent document. Adding one is a schema change plus a migration,
so it deserves a measurement first.

WHY THE OBVIOUS ESTIMATE IS NOT GOOD ENOUGH. We measured BM25's worth to the
CHUNK arm: +0.084 to +0.131 nDCG@10 across six leg/encoder pairs. Quoting that as
the headroom for sections is an extrapolation wearing a measurement's clothes --
exactly the error that produced the hybrid-default confound (an effect credited to
the mechanism under study that came from the baseline it was measured against).

There is a specific reason to expect it to differ. Qasper sections average ~3.3k
chars against ~800 for chunks, and BM25 is not scale-free: the `b` length
normalisation and term-frequency saturation both bite differently at 4x the
document length, and a query term that dominates a chunk is diluted in a section.
The section-level gain could be most of +0.10, or a fraction of it, or negative.

HOW. Build a REAL in-memory FTS5 index over section text and query it with
SQLite's own `bm25()`. Not a Python re-implementation: FTS5's tokenizer, k1 and b
defaults are what a shipped `sections_fts` would inherit, and a hand-rolled BM25
would answer a question about my arithmetic instead of about the feature.
Fusion replicates `_relative_score_fusion` exactly -- min-max within the query's
own candidate pool, then `vector_weight` blend, missing leg = 0.0.

THE VALIDATION THAT MAKES IT TRUSTWORTHY. The same code path also scores the
CHUNK arm, where the answer is already known from `src/`. If this harness does not
reproduce the published chunk-level BM25 delta, its section number means nothing
and the run aborts on that basis rather than reporting a number that looks fine.

Zero embedding: every vector comes from the `hier_embed` disk cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metrics import ndcg_at_k, recall_at_k  # noqa: E402

logger = logging.getLogger("section_bm25")

K = 10
# Mirrors _hybrid_search: search_k = max(k, min(k*4, 100)); the keyword leg
# over-fetches 2x then truncates back to search_k. Pool size changes the min-max
# normalisation, so it is part of the fusion rule, not an efficiency knob.
SEARCH_K = max(K, min(K * 4, 100))
VECTOR_WEIGHT = 0.5


def _minmax(values: Sequence[float]) -> List[float]:
    """Same degenerate-case behaviour as src's _minmax_normalize."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def fuse(vector_scores: Dict[str, float], bm25_raw: Dict[str, float], vector_weight: float) -> Dict[str, float]:
    """Replica of _relative_score_fusion. BM25 arrives raw and negative-is-better."""
    nv = dict(zip(vector_scores, _minmax(list(vector_scores.values())), strict=True))
    nk = dict(zip(bm25_raw, _minmax([-r for r in bm25_raw.values()]), strict=True))
    fused: Dict[str, float] = {}
    for key in (*vector_scores, *bm25_raw):
        if key in fused:
            continue
        fused[key] = vector_weight * nv.get(key, 0.0) + (1.0 - vector_weight) * nk.get(key, 0.0)
    return fused


class FTS:
    """An in-memory FTS5 index over arbitrary units -- i.e. the sections_fts we might ship.

    Carries an UNINDEXED ``doc`` column so a scoped corpus (MAUD ranks each query
    only within its own contract) can filter *inside* the query. Filtering after
    ``LIMIT`` would be wrong: the pool would fill with out-of-scope hits and the
    in-scope ones would never be seen.
    """

    def __init__(self, ids: Sequence[str], texts: Sequence[str], docs: Sequence[str]) -> None:
        self.ids = list(ids)
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE VIRTUAL TABLE u USING fts5(body, doc UNINDEXED)")
        self.conn.executemany(
            "INSERT INTO u(rowid, body, doc) VALUES (?, ?, ?)",
            ((i, t, d) for i, (t, d) in enumerate(zip(texts, docs, strict=True))),
        )
        self.conn.commit()

    def search(self, query: str, limit: int, scope: Optional[str] = None) -> Dict[str, float]:
        """{unit_id: raw bm25} best-first, using src's own query sanitisation."""
        from localvectordb._filters import FTSQuerySanitization

        sanitized = FTSQuerySanitization.sanitize_fts_query(query)
        if not sanitized:
            return {}
        sql = "SELECT rowid, bm25(u) AS rank FROM u WHERE u MATCH ?"
        params: List[object] = [sanitized]
        if scope is not None:
            sql += " AND doc = ?"
            params.append(scope)
        sql += " ORDER BY rank ASC LIMIT ?"
        params.append(limit)
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:  # malformed MATCH after sanitisation
            return {}
        return {self.ids[r[0]]: float(r[1]) for r in rows}


def vector_top(sims: np.ndarray, ids: Sequence[str], k: int) -> Dict[str, float]:
    idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    idx = idx[np.argsort(-sims[idx])]
    return {ids[i]: float(sims[i]) for i in idx}


def rollup(scores: Dict[str, float], owner: Dict[str, Sequence[str]]) -> Dict[str, float]:
    """Max roll-up of unit scores to their parent(s).

    ``owner`` maps a unit to a *list* of parents so midpoint and overlap
    attribution share one code path: midpoint passes a one-element list, overlap
    passes every section the chunk touches. That is the only difference between
    the two arms -- see ``--rollup``.
    """
    out: Dict[str, float] = {}
    for uid, s in scores.items():
        for p in owner[uid]:
            if s > out.get(p, -np.inf):
                out[p] = s
    return out


def score(ranked_per_query: List[List[str]], qids: Sequence[str], qrels: Dict[str, Dict[str, int]]) -> np.ndarray:
    pairs = zip(qids, ranked_per_query, strict=True)
    return np.array([ndcg_at_k(ranked, qrels.get(q, {}), K) for q, ranked in pairs], dtype=np.float64)


def recall(ranked_per_query: List[List[str]], qids: Sequence[str], qrels: Dict[str, Dict[str, int]]) -> float:
    vals = [recall_at_k(ranked, qrels.get(q, {}), K) for q, ranked in zip(qids, ranked_per_query, strict=True)]
    return float(np.mean(vals)) if vals else float("nan")


def paired(a: np.ndarray, b: np.ndarray, resamples: int = 10_000) -> Dict[str, float]:
    rng = np.random.default_rng(0)
    d = a - b
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    means = d[idx].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, len(d)))
    p = float((np.abs((signs * d).mean(axis=1)) >= abs(d.mean())).sum() + 1) / (resamples + 1)
    return {
        "delta": float(d.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "p": p,
    }


def run_arm(
    unit_ids: Sequence[str],
    unit_vecs: np.ndarray,
    fts: Optional[FTS],
    qvecs: np.ndarray,
    qtexts: Sequence[str],
    owner: Optional[Dict[str, str]],
    vector_weight: float,
    unit_docs: Optional[Sequence[str]] = None,
    q_scope: Optional[Sequence[Optional[str]]] = None,
) -> List[Dict[str, float]]:
    """Per-query {target_id: score}. ``fts=None`` is the vector-only arm.

    ``q_scope`` restricts each query to units of one document -- MAUD asks "which
    section of *this contract*", so ranking against all 152 contracts would be a
    different and much easier-looking task.

    Scoping SLICES the unit matrix per document rather than masking the full one.
    Masking looked equivalent and was ~150x slower: `unit_docs == scope` over an
    object array is a Python-level loop across 34k strings, run once per query per
    arm. Slicing also shrinks the matmul itself to the ~228 units that can score.
    """
    by_doc: Dict[str, np.ndarray] = {}
    if q_scope is not None and unit_docs is not None:
        order: Dict[str, List[int]] = {}
        for i, d in enumerate(unit_docs):
            order.setdefault(d, []).append(i)
        by_doc = {d: np.asarray(ix, dtype=np.int64) for d, ix in order.items()}

    out: List[Dict[str, float]] = []
    for qi, qtext in enumerate(qtexts):
        scope = q_scope[qi] if q_scope is not None else None
        if scope is None:
            sims = unit_vecs @ qvecs[qi]
            ids: Sequence[str] = unit_ids
        else:
            idx = by_doc.get(scope)
            if idx is None or not len(idx):
                out.append({})
                continue
            sims = unit_vecs[idx] @ qvecs[qi]
            ids = [unit_ids[i] for i in idx]
        vec = vector_top(sims, ids, SEARCH_K)
        if fts is None:
            scores = vec
        else:
            kw = fts.search(qtext, SEARCH_K * 2, scope)
            kw = dict(list(kw.items())[:SEARCH_K])
            scores = fuse(vec, kw, vector_weight)
        out.append(rollup(scores, owner) if owner is not None else scores)
    return out


def fuse_levels(
    chunk_side: List[Dict[str, float]], section_side: List[Dict[str, float]], section_weight: float
) -> List[Dict[str, float]]:
    """Blend a chunk-derived and a section-derived pool -- the `fused` search level.

    Min-max within each query's own pool before blending, mirroring
    `_two_leg_minmax_fuse`: the two legs are on unrelated scales otherwise and
    ``section_weight`` stops being a blend.
    """
    out: List[Dict[str, float]] = []
    for a, b in zip(chunk_side, section_side, strict=True):
        na = dict(zip(a, _minmax(list(a.values())), strict=True))
        nb = dict(zip(b, _minmax(list(b.values())), strict=True))
        keys = {*a, *b}
        out.append({k: (1.0 - section_weight) * na.get(k, 0.0) + section_weight * nb.get(k, 0.0) for k in keys})
    return out


def to_ranked(scores_per_query: List[Dict[str, float]]) -> List[List[str]]:
    return [[u for u, _ in sorted(s.items(), key=lambda kv: -kv[1])[:K]] for s in scores_per_query]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-key", default="egemma")
    p.add_argument("--dataset", choices=("qasper", "maud", "mldr", "nq"), default="qasper")
    p.add_argument(
        "--coarse",
        choices=("sections", "documents"),
        default="sections",
        help="The coarse unit to pit against chunks. 'documents' exists because MLDR has NO detectable "
        "headings (0/800 docs) and returns section_qrels empty on purpose -- inventing sections there "
        "would be spans with fabricated relevance. It is also the cheaper fix to evaluate: "
        "`documents_fts` is ALREADY created, populated and trigger-maintained in every database, and "
        "no search path reads it, so wiring a document-level keyword leg needs no migration at all.",
    )
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--vector-weight", type=float, default=VECTOR_WEIGHT)
    p.add_argument(
        "--section-weight",
        type=float,
        default=0.65,
        help="Weight on the section leg of the fused arm. Default is the shipped 0.65, which is also "
        "the measured qasper argmax -- and wrong on long-section legs (0.10-0.40 there).",
    )
    p.add_argument(
        "--rollup",
        choices=("midpoint", "overlap"),
        default="midpoint",
        help="How a chunk hit is credited to sections. 'midpoint' mirrors what src/ stores in "
        "chunks.section_id (one section, the one holding the chunk's midpoint) and orphans 40.1%% of "
        "sections at the shipped chunk_size. 'overlap' credits every section the chunk touches -- "
        "what a chunk<->section join table would allow. Only affects the SECTION target.",
    )
    p.add_argument(
        "--reachable-only",
        action="store_true",
        help="Drop queries whose gold sections own no chunk. The chunks arm cannot rank those at any "
        "k (chunks.section_id is single-valued), so an unfiltered section-target win is partly a "
        "reachability artifact. Run BOTH and report the pair.",
    )
    p.add_argument("--allow-embed", action="store_true", help="permit cache misses to be embedded (costs money/time)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("localvectordb.chunking").setLevel(logging.ERROR)

    from benchmarks.eval_dual import MODEL_POOL, _load_experiments_env, load_units

    # experiments/.env holds OPENAI_API_KEY. eval_dual loads it in its own main();
    # this harness has to do it too or any cache miss dies on 'API key is required'.
    _load_experiments_env()

    if args.dataset == "maud":
        from benchmarks.maud_data import detect_contract_sections, load_maud

        bench = load_maud(max_contracts=args.max_papers)
        units = load_units(bench, None, detect_contract_sections)
    elif args.dataset == "mldr":
        from benchmarks.mldr_data import load_mldr

        bench = load_mldr(split="dev", max_queries=args.max_papers)
        units = load_units(bench, None)
    elif args.dataset == "nq":
        from benchmarks.nq_data import load_nq

        # NQ carries ~1.06 queries per article, so --max-papers caps both.
        bench = load_nq(max_queries=args.max_papers)
        units = load_units(bench, None)
    else:
        from benchmarks.qasper_data import load_qasper

        bench = load_qasper(split="dev", max_papers=args.max_papers)
        units = load_units(bench, None)

    if args.coarse == "sections" and not units.section_texts:
        raise SystemExit(
            f"{args.dataset} yielded 0 sections, so there is no section arm to measure. MLDR has no "
            "detectable headings and returns section_qrels empty ON PURPOSE -- fabricating sections "
            "would be spans with invented relevance. Use --coarse documents."
        )

    spec = MODEL_POOL[args.model_key]
    coarse_texts = list(units.section_texts) if args.coarse == "sections" else [bench.corpus[d] for d in bench.corpus]
    coarse_docs = list(units.section_doc) if args.coarse == "sections" else list(bench.corpus)
    coarse_uid = list(units.section_ids) if args.coarse == "sections" else list(bench.corpus)

    from dataclasses import replace

    from benchmarks.eval_dual import PrefixedEncoder

    doc_enc = PrefixedEncoder(spec, spec.doc_prefix)
    qry_enc = PrefixedEncoder(spec, spec.query_prefix)
    # Mirrors embed_model: a spec may re-window the coarse unit more finely than
    # chunks. Encoding here rather than via embed_model because that helper always
    # encodes sections, and --coarse documents needs none -- on MLDR that meant
    # trying to embed stray uncached section windows the run never uses.
    coarse_enc = doc_enc
    if args.coarse == "sections" and spec.section_window_chars is not None:
        coarse_enc = PrefixedEncoder(
            replace(spec, window_chars=spec.section_window_chars, window_tokens=None), spec.doc_prefix
        )

    if not args.allow_embed:
        miss = (
            doc_enc.count_misses(units.chunk_texts)[1]
            + coarse_enc.count_misses(coarse_texts)[1]
            + qry_enc.count_misses(units.query_texts)[1]
        )
        if miss:
            raise SystemExit(
                f"{miss} vectors are not cached for {spec.model}. This harness is zero-embedding "
                "by default; pass --allow-embed to encode them."
            )

    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1.0, n)

    cv = unit(doc_enc.encode(units.chunk_texts, normalize=False))
    qv = unit(qry_enc.encode(units.query_texts, normalize=False))
    # encode() windows and mean-pools anything over the model's window, so a
    # 36k-char MLDR document is represented in full rather than truncated.
    sv = unit(coarse_enc.encode(coarse_texts, normalize=False))
    qids = list(units.query_ids)
    qtexts = list(units.query_texts)

    # THE ROLL-UP HANDICAP CONTROL. `chunks.section_id` is single-valued, so a
    # section that owns no chunk cannot be ranked by the chunks arm at any k --
    # 26.3% of qasper's gold sections and 17.5% of NQ's. Any section-target win
    # is therefore partly a reachability artifact rather than a retrieval one.
    # This drops the queries whose gold is unreachable, so both arms compete over
    # the same attainable set. Report it BESIDE the unfiltered number, never
    # instead of it: the dropped queries are real queries the roll-up really fails.
    if args.reachable_only:
        if not bench.section_qrels:
            raise SystemExit("--reachable-only needs section qrels; MLDR has none by design.")
        # Reachability follows the SELECTED attribution: under `overlap` far more
        # sections own a chunk, which is exactly the effect being measured.
        if args.rollup == "overlap":
            owned = {s for ss in units.chunk_sections_all for s in ss if s is not None}
        else:
            owned = {s for s in units.chunk_section if s is not None}
        keep = [i for i, q in enumerate(qids) if any(s in owned for s in bench.section_qrels.get(q, {}))]
        dropped = len(qids) - len(keep)
        if not keep:
            raise SystemExit("--reachable-only removed every query; nothing to measure.")
        logger.info(
            "reachable-only: dropped %d/%d queries (%.1f%%) whose gold sections own no chunk",
            dropped,
            len(qids),
            100 * dropped / len(qids),
        )
        qv = qv[keep]
        qids = [qids[i] for i in keep]
        qtexts = [qtexts[i] for i in keep]

    logger.info("%d chunks, %d %s, %d queries", len(cv), len(sv), args.coarse, len(qids))

    # FTS over each unit type. The coarse index is the thing under test.
    chunk_uid = [f"c{i}" for i in range(len(units.chunk_texts))]
    sec_uid = coarse_uid
    fts_chunks = FTS(chunk_uid, units.chunk_texts, units.chunk_doc)
    fts_secs = FTS(sec_uid, coarse_texts, coarse_docs)
    chunk_docs = list(units.chunk_doc)
    section_docs = coarse_docs

    # MAUD is per-contract scoped retrieval; qasper is not. Mirrors eval_dual's SCOPE_QID.
    q_scope: Optional[List[Optional[str]]] = None
    if args.dataset == "maud":
        q_scope = [str(q).split("||", 1)[0] for q in qids]
        logger.info("scoped retrieval: %d distinct contracts", len({s for s in q_scope}))

    # Roll-up attribution: the thing under test. `midpoint` is what src/ stores in
    # chunks.section_id -- one section per chunk, the one holding the chunk's
    # midpoint -- and it orphans 40.1% of sections at the shipped chunk_size.
    # `overlap` credits every section the chunk touches, which is what a
    # chunk<->section join table would allow. Nothing else differs between them.
    if args.rollup == "overlap":
        chunk_to_sec = {u: list(v) for u, v in zip(chunk_uid, units.chunk_sections_all, strict=True)}
    else:
        chunk_to_sec = {u: [v] for u, v in zip(chunk_uid, units.chunk_section, strict=True)}
    chunk_to_doc = {u: [v] for u, v in zip(chunk_uid, units.chunk_doc, strict=True)}
    sec_to_doc = {u: [v] for u, v in zip(sec_uid, coarse_docs, strict=True)}

    if args.coarse == "sections":
        reached = {s for ss in chunk_to_sec.values() for s in ss}
        blind = sum(1 for s in sec_uid if s not in reached)
        logger.info(
            "rollup=%s: %d/%d sections own no chunk (%.1f%%) -- unrankable by the chunks arm at any k",
            args.rollup,
            blind,
            len(sec_uid),
            100 * blind / max(len(sec_uid), 1),
        )

    results: Dict[str, Dict[str, float]] = {}
    targets = [("section", bench.section_qrels), ("doc", bench.doc_qrels)]
    if args.dataset == "maud":
        # Every query is scoped to its own contract, so the doc target has one
        # candidate and is trivially perfect -- a number that means nothing.
        targets = [("section", bench.section_qrels)]

    for tname, qrels in targets:
        if not qrels:
            continue
        chunk_owner = chunk_to_sec if tname == "section" else chunk_to_doc
        sec_owner = None if tname == "section" else sec_to_doc
        CN = args.coarse  # "sections" or "documents" -- label the arm by what it is
        arms = {
            "chunks · vector": (chunk_uid, cv, None, chunk_owner, chunk_docs),
            "chunks · hybrid": (chunk_uid, cv, fts_chunks, chunk_owner, chunk_docs),
            f"{CN} · vector": (sec_uid, sv, None, sec_owner, section_docs),
            f"{CN} · hybrid": (sec_uid, sv, fts_secs, sec_owner, section_docs),
        }
        raw: Dict[str, List[Dict[str, float]]] = {}
        for label, (uids, uvecs, fts, owner, udocs) in arms.items():
            raw[label] = run_arm(uids, uvecs, fts, qv, qtexts, owner, args.vector_weight, udocs, q_scope)

        # `fused` blends the two levels. "vector" is what ships TODAY -- the fused
        # level ignores search_type entirely, so both its legs are vector-only.
        # "hybrid" is what it would become once sections had a keyword leg.
        raw["fused · vector"] = fuse_levels(raw["chunks · vector"], raw[f"{CN} · vector"], args.section_weight)
        raw["fused · hybrid"] = fuse_levels(raw["chunks · hybrid"], raw[f"{CN} · hybrid"], args.section_weight)

        pq: Dict[str, np.ndarray] = {}
        print(f"\n=== target: {tname} ===")
        print(f"  {'arm':<20} {'nDCG@10':>9} {'recall@10':>10}")
        for label, scores_per_q in raw.items():
            ranked = to_ranked(scores_per_q)
            pq[label] = score(ranked, qids, qrels)
            print(f"  {label:<20} {pq[label].mean():>9.4f} {recall(ranked, qids, qrels):>10.4f}")
            results.setdefault(tname, {})[label] = float(pq[label].mean())

        def _report(label: str, a: str, b: str, _pq=pq, _t=tname) -> None:
            st = paired(_pq[a], _pq[b])
            results[_t][label] = st["delta"]
            star = "*" if (st["ci_lo"] > 0 or st["ci_hi"] < 0) else " "
            print(f"  {label:<34} {st['delta']:+.4f} [{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}] p={st['p']:.4f}{star}")

        # What BM25 is worth at each level -- the ratio between these two is the
        # quantity a `sections_fts` decision actually turns on.
        for level in ("chunks", CN):
            _report(f"BM25 contribution to {level}", f"{level} · hybrid", f"{level} · vector")
        c = results[tname].get("BM25 contribution to chunks") or float("nan")
        s = results[tname].get(f"BM25 contribution to {CN}") or float("nan")
        results[tname]["coarse_share_of_chunk_gain"] = s / c if c else float("nan")
        # Quote the ABSOLUTE gain, not this: the share is 63-72% on qasper and 477%
        # on MAUD purely because the two baselines have different headroom.
        print(f"  {f'-> {CN} get this share of it':<34} {100 * s / c:.0f}%" if c else "")

        # The decision question: with BOTH levels given a keyword leg, which wins?
        # Today only the left side of this comparison exists.
        _report(f"{CN} vs chunks (both HYBRID)", f"{CN} · hybrid", "chunks · hybrid")
        _report(f"{CN} vs chunks (both vector)", f"{CN} · vector", "chunks · vector")
        _report("BM25 contribution to fused", "fused · hybrid", "fused · vector")
        # The real product question. LHS is what a user would get after the fix;
        # RHS is the best arm available today (chunks is the only level with BM25).
        _report("fused-hybrid vs chunks-hybrid", "fused · hybrid", "chunks · hybrid")
        _report("fused-vector vs chunks-hybrid (TODAY)", "fused · vector", "chunks · hybrid")

    print("\n" + "=" * 72)
    print("VALIDATION: the chunk-level BM25 delta must reproduce what src/ already gives.")
    print("If it does not, the section-level number above is measuring this harness, not the feature.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "model": spec.model,
                    "dataset": args.dataset,
                    "coarse": args.coarse,
                    "rollup": args.rollup,
                    "queries": len(qids),
                    # A reachable-only run is a CONTROL, not a headline. Recording
                    # the flag stops the two files being compared as if they were
                    # the same measurement over the same query set.
                    "reachable_only": bool(args.reachable_only),
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] if len(sys.argv) > 1 else None))
