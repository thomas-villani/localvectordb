"""Sweep BM25's `k1` -- reachable for free through FTS5's column weight.

THE SETUP. Every keyword number in this study came from stock FTS5 at its
defaults, and `k1` (term-frequency saturation) is one of them: 1.2, hardcoded in
`fts5_aux.c`, never chosen by us. `KEYWORD-STRATEGIES.md` §1.2 lists a "`k1`/`b`
sweep" as query-time and free. Half of that is true, and the half that works is
not the obvious one.

WHY THIS IS REACHABLE AT ALL. `bm25()` exposes no k1 argument. Its optional
arguments are per-column weights -- and FTS5 multiplies the column weight into
TERM FREQUENCY, not into the finished score. Written out, with w the body
column's weight:

    IDF * (w*f)*(k1+1) / (w*f + k1*(1 - b + b*D/avgdl))

Divide numerator and denominator by w:

    IDF * f*(k1+1) / (f + (k1/w)*L)          where L = 1 - b + b*D/avgdl

which is BM25 at **k1_eff = k1/w = 1.2/w**, times the constant (k1+1)/(k1_eff+1).
That factor is the same for every document and every query term, so it cannot
reorder anything, and min-max normalisation in fusion is invariant to a positive
scale -- the sweep is exact end to end, not an approximation.

Verified against a hand-computed BM25 on a 22-document corpus, matching FTS5's
output to 4 decimal places, before this harness was written.

WHAT IS NOT REACHABLE. `b` (length normalisation). On a single indexed column the
second weight argument is silently ignored -- `bm25(u, 1.0, 0.0)` and
`bm25(u, 1.0, 1.0)` return identical scores -- so anything that looks like a `b`
sweep is a no-op. Reaching `b` means scoring outside `bm25()`, which is the same
work as BM25+, and it belongs with the expensive items rather than the free ones.
§1.2 has been corrected accordingly.

TWO CONTROLS, because a null result here is easy to fake:

  1. PLUMBING. The explicit `bm25(u, 1.0)` form must return per-query scores
     EXACTLY equal to the shipped `bm25(u)` form. If it does not, the weight
     argument is not doing what this file claims and no cell below is readable.
  2. CHURN and OVERLAP. Every cell reports both the share of queries whose
     keyword top-10 changed at all (order included) and the mean share of the
     baseline's top-10 that survived. "No nDCG effect" and "no effect on
     retrieval" are different findings, and the pair separates them: churn alone
     counts a rank-9/10 swap as a full replacement, overlap alone hides pure
     reordering. A flat row at 0% churn would mean the grid was too narrow, not
     that k1 does not matter.

Zero embedding: vectors come from the `hier_embed` disk cache.

    python benchmarks/eval_bm25_k1.py --dataset qasper
    python benchmarks/eval_bm25_k1.py --dataset nq --coarse documents
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.eval_section_bm25 import (  # noqa: E402
    FTS,
    SEARCH_K,
    VECTOR_WEIGHT,
    blend_arm,
    capture_arm,
    paired,
    recall,
    score,
    to_ranked,
)

logger = logging.getLogger("bm25_k1")

# FTS5's hardcoded k1. The sweep is expressed in k1 and converted to the weight
# that produces it, so the table reads as BM25 rather than as SQLite trivia.
K1_SHIPPED = 1.2

# Spans the range the IR literature actually uses (roughly 0.9-2.0) plus a hard
# saturating end and a near-linear end, so a flat middle can be told apart from a
# grid that never left the neighbourhood of the default.
#
# EXTENDED DOWNWARD after the first MAUD run put its argmax ON the boundary at
# k1=0.3, monotone and significant at every step -- a grid that does not bracket
# its optimum reports the edge it stopped at, not the optimum. The low end is
# the interesting direction anyway: as k1 -> 0 the tf factor f*(k1+1)/(f + k1*L)
# tends to 1 for any f > 0, so BM25 degenerates to a pure sum of IDF over
# matched terms -- binary "did the term appear", with term frequency and length
# normalisation both switched off. If MAUD's optimum is down there, the finding
# is that BM25's term-frequency model is actively harmful on templated text,
# which is a stronger and more falsifiable claim than "tune k1 lower".
K1_GRID: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0)


def weight_for(k1: float) -> float:
    """The bm25() column weight that makes FTS5 behave as BM25 at ``k1``."""
    return K1_SHIPPED / k1


def capture_keyword(
    fts: FTS, qtexts: Sequence[str], q_scope: Optional[Sequence[Optional[str]]]
) -> List[Dict[str, float]]:
    """Per-query raw BM25 hits, fetched exactly as ``capture_arm`` fetches them.

    Split out so the sweep pays for the vector matmul ONCE. Vector scores do not
    depend on k1; re-running ``capture_arm`` per grid point would spend all its
    time recomputing an identical matrix.
    """
    out: List[Dict[str, float]] = []
    for qi, qtext in enumerate(qtexts):
        scope = q_scope[qi] if q_scope is not None else None
        kw = fts.search(qtext, SEARCH_K * 2, scope)
        out.append(dict(list(kw.items())[:SEARCH_K]))
    return out


def top_ids(kw: Dict[str, float], k: int = 10) -> Tuple[str, ...]:
    return tuple(list(kw)[:k])


def churn(a: Sequence[Dict[str, float]], b: Sequence[Dict[str, float]]) -> float:
    """Share of queries whose keyword top-10 differs -- ORDER included.

    Deliberately the most sensitive form: one swap of ranks 9 and 10 counts the
    same as a complete replacement. Read it only next to ``overlap`` below, which
    is the one that says how MUCH moved. Quoting this alone would turn "the pool
    jiggled" into "the pool was replaced".
    """
    diff = sum(1 for x, y in zip(a, b, strict=True) if top_ids(x) != top_ids(y))
    return diff / max(len(a), 1)


def overlap(a: Sequence[Dict[str, float]], b: Sequence[Dict[str, float]]) -> float:
    """Mean share of the baseline's top-10 SET still present -- membership only.

    The pair (churn, overlap) separates three outcomes a single number confuses:
    nothing moved (low churn, overlap 1.0), the same documents got reordered
    (high churn, overlap ~1.0), and a genuinely different pool was retrieved
    (high churn, low overlap). Only the third can move nDCG much.
    """
    vals = []
    for x, y in zip(a, b, strict=True):
        base = set(top_ids(y))
        if not base:
            continue
        vals.append(len(base & set(top_ids(x))) / len(base))
    return float(np.mean(vals)) if vals else float("nan")


def verify_algebra(texts: Sequence[str], sample: int = 60, probes: int = 300) -> bool:
    """CONTROL 0: is the column weight really a k1 sweep, on this corpus's own text?

    The whole file rests on one algebraic claim. This checks it end to end against
    a hand-computed BM25, taking term frequencies and document lengths from
    ``fts5vocab`` -- FTS5's OWN tokenizer -- so a disagreement means the claim is
    wrong rather than that my tokenizer is.

    Runs on a SAMPLE. The identity is corpus-independent; sampling keeps the
    O(terms x weights x docs) Python loop cheap, and the full-corpus sweep above
    is the thing we actually want the CPU for.

    THE ONE PLACE FTS5 IS NOT BM25. ``fts5_aux.c`` clamps a negative IDF to
    +1e-6, so a term appearing in more than about half the corpus gets a tiny
    POSITIVE weight where textbook BM25 gives it a negative one -- flipping the
    sign of its whole contribution. It lands on stopwords, which our OR-joined
    queries carry by design: 29.3% of qasper query terms, 88.7% of queries.

    That reach sounds alarming and the magnitude says otherwise -- clamped terms
    supply 0.435% of the top-10 score, since 1e-6 against a real idf of 2-8 is
    nothing. It is free stopword removal rather than a bias, and Lucene keeps
    idf positive too. The failure mode to watch is a query whose terms are ALL
    clamped, where ranking falls to tf and length alone; that set is empty on
    qasper.

    The deviation is present at w=1.0, i.e. in shipped behaviour. It is NOT
    caused by the weight, so it is reported separately rather than being allowed
    to look like a failure of the sweep.
    """
    import math
    import random
    import sqlite3
    from collections import defaultdict

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE u USING fts5(body)")
    sub = list(texts)[:sample]
    conn.executemany("INSERT INTO u(rowid, body) VALUES (?, ?)", enumerate(sub))
    conn.execute("CREATE VIRTUAL TABLE v USING fts5vocab(u, instance)")
    conn.commit()

    tf: Dict[Tuple[str, int], int] = defaultdict(int)
    dl: Dict[int, int] = defaultdict(int)
    for term, doc, _col, _off in conn.execute("SELECT term, doc, col, offset FROM v"):
        tf[(term, doc)] += 1
        dl[doc] += 1
    n_docs = len(sub)
    if n_docs < 5 or not dl:
        print("  ALGEBRA CONTROL: corpus too small to verify; skipped")
        return True
    avgdl = sum(dl.values()) / n_docs
    df: Dict[str, int] = defaultdict(int)
    for term, _doc in tf:
        df[term] += 1

    b = 0.75

    def hand(term: str, k1: float) -> Dict[int, float]:
        idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5))
        out = {}
        for doc in range(n_docs):
            f = tf.get((term, doc), 0)
            if f:
                out[doc] = idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[doc] / avgdl))
        return out

    def idf_of(term: str) -> float:
        return math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5))

    rng = random.Random(0)
    pos = [t for t, n in df.items() if n >= 2 and t.isalpha() and idf_of(t) > 0]
    clamped = [t for t, n in df.items() if n >= 2 and t.isalpha() and idf_of(t) <= 0]
    if not pos:
        print("  ALGEBRA CONTROL: no positive-idf terms in sample; skipped")
        return True

    ok = True
    for label, pool in (("positive idf", pos), ("clamped idf", clamped)):
        if not pool:
            continue
        picks = rng.sample(pool, min(probes, len(pool)))
        bad = checked = 0
        worst = 0.0
        for k1 in K1_GRID:
            w = weight_for(k1)
            const = (K1_SHIPPED + 1) / (k1 + 1)
            for term in picks:
                rows = conn.execute(
                    "SELECT rowid, bm25(u, ?) FROM u WHERE u MATCH ? ORDER BY 2 ASC", (w, term)
                ).fetchall()
                if len(rows) < 3:
                    continue
                checked += 1
                mine = hand(term, k1)
                if [r[0] for r in rows] != [d for d, _ in sorted(mine.items(), key=lambda kv: (-kv[1], kv[0]))]:
                    bad += 1
                ratios = [(-sc) / (mine[rid] * const) for rid, sc in rows if mine.get(rid)]
                if len(ratios) > 1:
                    worst = max(worst, (max(ratios) - min(ratios)) / abs(np.mean(ratios)))
        note = ""
        if label == "clamped idf" and bad:
            note = "  (expected: FTS5 clamps negative idf to +1e-6, so it is not BM25 here)"
        else:
            ok = ok and bad == 0
        print(
            f"  ALGEBRA CONTROL [{label}]: {checked} probes, ranking != BM25(k1_eff): {bad}, "
            f"worst relative score spread {worst:.1e}{note}"
        )
    print(f"  ({len(pos)} positive-idf vs {len(clamped)} clamped-idf terms in a {n_docs}-doc sample)")
    return ok


def unit_profile(texts: Sequence[str], qtexts: Sequence[str], sample: int = 3000) -> Dict[str, float]:
    """Unit length and query-term repetition -- the quantities k1 acts on.

    WHY THIS IS HERE. MAUD's optimum is far below the shipped k1 at SECTION
    granularity (+0.0276) and barely moves at CHUNK granularity (+0.0029);
    MLDR's long documents lean the same way; qasper's short units are flat. The
    hypothesis is that k1 only matters when term frequencies are large enough for
    saturation to be the operative part of BM25 -- i.e. in long, repetitive
    units. The saturation term is f/(k1*L), so this reports mean tokens per unit
    and the mean tf of query terms where they occur, which is what that ratio is
    built from.

    Computed on a sample and on the raw query string's terms: this is a
    descriptive profile for interpreting the sweep, not an input to any score.
    """
    import sqlite3
    from collections import defaultdict

    sub = list(texts)[:sample]
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE u USING fts5(body)")
    conn.executemany("INSERT INTO u(rowid, body) VALUES (?,?)", enumerate(sub))
    conn.execute("CREATE VIRTUAL TABLE vi USING fts5vocab(u, instance)")
    conn.commit()

    tf: Dict[Tuple[str, int], int] = defaultdict(int)
    dl: Dict[int, int] = defaultdict(int)
    for term, doc, _col, _off in conn.execute("SELECT term, doc, col, offset FROM vi"):
        tf[(term, doc)] += 1
        dl[doc] += 1
    if not dl:
        return {"avgdl": float("nan"), "mean_tf": float("nan"), "p90_tf": float("nan")}

    qterms = {t for q in qtexts[:2000] for t in str(q).lower().split() if t.isalpha()}
    vals = [f for (term, _doc), f in tf.items() if term in qterms]
    return {
        "avgdl": float(np.mean(list(dl.values()))),
        "mean_tf": float(np.mean(vals)) if vals else float("nan"),
        "p90_tf": float(np.percentile(vals, 90)) if vals else float("nan"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("qasper", "maud", "mldr", "nq"), default="qasper")
    p.add_argument("--model-key", default="egemma")
    p.add_argument("--coarse", choices=("sections", "documents"), default="sections")
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--vector-weight", type=float, default=VECTOR_WEIGHT)
    p.add_argument("--allow-embed", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("localvectordb.chunking").setLevel(logging.ERROR)

    from benchmarks.eval_dual import MODEL_POOL, PrefixedEncoder, _load_experiments_env, load_units

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

        bench = load_nq(max_queries=args.max_papers)
        units = load_units(bench, None)
    else:
        from benchmarks.qasper_data import load_qasper

        bench = load_qasper(split="dev", max_papers=args.max_papers)
        units = load_units(bench, None)

    if args.coarse == "sections" and not units.section_texts:
        raise SystemExit(f"{args.dataset} yielded 0 sections; use --coarse documents.")

    spec = MODEL_POOL[args.model_key]
    coarse_texts = list(units.section_texts) if args.coarse == "sections" else [bench.corpus[d] for d in bench.corpus]
    coarse_docs = list(units.section_doc) if args.coarse == "sections" else list(bench.corpus)
    coarse_uid = list(units.section_ids) if args.coarse == "sections" else list(bench.corpus)

    from dataclasses import replace

    doc_enc = PrefixedEncoder(spec, spec.doc_prefix)
    qry_enc = PrefixedEncoder(spec, spec.query_prefix)
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
    sv = unit(coarse_enc.encode(coarse_texts, normalize=False))
    qv = unit(qry_enc.encode(units.query_texts, normalize=False))
    qids, qtexts = list(units.query_ids), list(units.query_texts)

    q_scope: Optional[List[Optional[str]]] = None
    if args.dataset == "maud":
        q_scope = [str(q).split("||", 1)[0] for q in qids]

    chunk_uid = [f"c{i}" for i in range(len(units.chunk_texts))]
    chunk_docs = list(units.chunk_doc)
    chunk_to_sec = {u: [v] for u, v in zip(chunk_uid, units.chunk_section, strict=True)}
    chunk_to_doc = {u: [v] for u, v in zip(chunk_uid, chunk_docs, strict=True)}
    sec_to_doc = {u: [v] for u, v in zip(coarse_uid, coarse_docs, strict=True)}

    logger.info("%d chunks, %d %s, %d queries", len(cv), len(sv), args.coarse, len(qids))

    levels = {
        "chunks": (chunk_uid, cv, list(units.chunk_texts), chunk_docs),
        args.coarse: (coarse_uid, sv, coarse_texts, coarse_docs),
    }
    targets = [("section", bench.section_qrels), ("doc", bench.doc_qrels)]
    if args.dataset == "maud":
        targets = [("section", bench.section_qrels)]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_query_out: Dict[str, List[float]] = {}

    for lname, (uids, uvecs, texts, udocs) in levels.items():
        print(f"\n  --- {lname} ---")
        if not verify_algebra(texts):
            print("  the weight is NOT a k1 sweep on this corpus; refusing to report the table below")
            return 1
        prof = unit_profile(texts, qtexts)
        print(
            f"  UNIT PROFILE: {prof['avgdl']:.0f} tokens/unit, query-term tf mean "
            f"{prof['mean_tf']:.2f}, p90 {prof['p90_tf']:.0f}  (what k1 saturates)"
        )
        results[f"{args.dataset}/{lname}|profile"] = {
            "avgdl": prof["avgdl"],
            "mean_tf": prof["mean_tf"],
            "p90_tf": prof["p90_tf"],
        }
        fts = FTS(uids, texts, udocs)
        # CONTROL 1. The shipped SQL, then the same thing written with an explicit
        # neutral weight. These must agree exactly or the sweep is measuring the
        # weight argument's side effects rather than k1.
        base_kw = capture_keyword(fts, qtexts, q_scope)
        fts.weight = 1.0
        ctl_kw = capture_keyword(fts, qtexts, q_scope)
        exact = all(a == b for a, b in zip(base_kw, ctl_kw, strict=True))
        print(f"\n  PLUMBING CONTROL [{lname}]: bm25(u) == bm25(u, 1.0) exactly -> {'OK' if exact else 'VIOLATED'}")
        if not exact:
            print("  refusing to report a sweep whose neutral point is not neutral")
            return 1

        # Vector side is k1-invariant: capture it once with fts=None.
        vec_cap = [v for v, _ in capture_arm(uids, uvecs, None, qv, qtexts, udocs, q_scope)]

        kw_by_k1: Dict[float, List[Dict[str, float]]] = {}
        for k1 in K1_GRID:
            fts.weight = weight_for(k1)
            kw_by_k1[k1] = capture_keyword(fts, qtexts, q_scope)

        for tname, qrels in targets:
            if not qrels:
                continue
            if lname == "chunks":
                owner = chunk_to_sec if tname == "section" else chunk_to_doc
            else:
                owner = None if tname == "section" else sec_to_doc
            if lname == "documents" and tname == "section":
                continue

            for arm_name, vw in (("keyword", 0.0), ("hybrid", args.vector_weight)):
                cells: Dict[str, Dict[str, float]] = {}
                pq: Dict[float, np.ndarray] = {}
                for k1 in K1_GRID:
                    captured = list(zip(vec_cap, kw_by_k1[k1], strict=True))
                    ranked = to_ranked(blend_arm(captured, owner, vw, use_keyword=True))
                    pq[k1] = score(ranked, qids, qrels)
                    cells[f"k1={k1:g}"] = {
                        "ndcg@10": float(pq[k1].mean()),
                        "recall@10": recall(ranked, qids, qrels),
                        "churn": churn(kw_by_k1[k1], kw_by_k1[K1_SHIPPED]),
                        "overlap": overlap(kw_by_k1[k1], kw_by_k1[K1_SHIPPED]),
                    }

                tag = f"{args.dataset}/{lname}/{tname}/{arm_name}"
                print(f"\n=== {tag} · vw={vw:.2f} ===")
                print(
                    f"  {'k1':>6} {'weight':>8} {'nDCG@10':>9} {'delta':>9} {'95% CI':>20} "
                    f"{'churn':>7} {'overlap':>8}"
                )
                for k1 in K1_GRID:
                    st = paired(pq[k1], pq[K1_SHIPPED])
                    cell = cells[f"k1={k1:g}"]
                    ci = f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
                    star = "*" if (st["ci_lo"] > 0 or st["ci_hi"] < 0) else " "
                    mark = "  <- shipped" if k1 == K1_SHIPPED else ""
                    pct = 100 * cell["churn"]
                    print(
                        # :g, not :.1f -- the extended grid reaches 0.05, which
                        # .1f renders as "0.1", giving two rows the same label.
                        f"  {k1:>6g} {weight_for(k1):>8.3f} {pq[k1].mean():>9.4f} "
                        f"{st['delta']:>+9.4f} {ci:>20}{star} {pct:>6.1f}% {cell['overlap']:>8.3f}{mark}"
                    )
                    cell["delta"] = st["delta"]
                    cell["ci_lo"] = st["ci_lo"]
                    cell["ci_hi"] = st["ci_hi"]
                    per_query_out[f"{tag}|k1={k1:g}"] = [float(v) for v in pq[k1]]
                best = max(K1_GRID, key=lambda k: pq[k].mean())
                print(
                    f"  argmax k1={best:g} ({pq[best].mean():.4f}, {pq[best].mean() - pq[K1_SHIPPED].mean():+.4f} "
                    "vs shipped) -- picked on the queries it is scored on, so it is an upper bound"
                )
                results[tag] = cells

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "model": spec.model,
                    "coarse": args.coarse,
                    "vector_weight": args.vector_weight,
                    "queries": len(qids),
                    "qids": [str(q) for q in qids],
                    "grid": list(K1_GRID),
                    "results": results,
                    "per_query": per_query_out,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] if len(sys.argv) > 1 else None))
