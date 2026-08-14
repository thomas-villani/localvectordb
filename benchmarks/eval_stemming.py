"""Porter stemming: the one-line default nobody chose (KEYWORD-STRATEGIES §1.1).

`grep` finds NO `tokenize=` clause on any FTS5 table in src/, so every keyword
number in this study was produced on the default `unicode61`: no stemming, no
stopword handling. "training", "trained" and "train" are three unrelated terms,
and a qasper query asking "what datasets were used" cannot match "our dataset".

    CREATE VIRTUAL TABLE chunks_fts USING fts5(content, tokenize='porter unicode61')

Shipping that needs an index rebuild and is therefore a migration question. MEASURING
it does not: this harness builds its own in-memory index, so the rebuild is a
shipping cost rather than an evaluation one. Zero embedding either way.

WHY THIS REPORTS FIVE THINGS AND NOT ONE. Stemming is usually quoted as "worth a
point or three on English collections", but it trades two effects against each
other and a single nDCG number hides which one we got:

  * RECALL UP -- morphological variants now match. Visible as the dead-leg count
    (queries whose keyword leg returns nothing) and as recall@10.
  * PRECISION DOWN -- it conflates words that should stay distinct. Classic
    damage: "university"/"universe" both stem to "univers".
  * IDF SHIFTS -- merging variants RAISES df, which pushes terms across FTS5's
    negative-idf boundary into the clamped regime (see eval_bm25_k1.py). This
    was predicted in KEYWORD-STRATEGIES §1.2a before it was measured here, so
    it is reported whether or not it flatters the result.
  * VOCABULARY SHRINKS -- the mechanical size of the change, which says whether
    a null result means "stemming did nothing" or "the tokenizer barely fired".
  * CHURN/OVERLAP -- did the retrieved set actually move? A flat nDCG with an
    unchanged pool is a different finding from a flat nDCG with a replaced one.

    python benchmarks/eval_stemming.py --dataset qasper
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.eval_bm25_k1 import capture_keyword, churn, overlap  # noqa: E402
from benchmarks.eval_section_bm25 import (  # noqa: E402
    FTS,
    VECTOR_WEIGHT,
    blend_arm,
    capture_arm,
    paired,
    recall,
    score,
    to_ranked,
)

logger = logging.getLogger("stemming")

# None is the shipped tokenizer (no clause -> unicode61). "porter unicode61"
# wraps that in the Porter stemmer, which is the exact one-line change §1.1
# proposes.
TOKENIZERS: Tuple[Tuple[str, Optional[str]], ...] = (
    ("unicode61", None),
    ("porter", "porter unicode61"),
)


def sanitize_and(query: str):
    """The shipped sanitiser with its OR-joins turned into AND-joins.

    THE CONTROL FOR THE MECHANISM, not a proposal. Stemming lost on qasper and I
    explained it by saying its upside was already spent: `sanitize_fts_query`
    OR-joins every term, so no query ever returns nothing, so there is nothing
    for stemming to rescue (dead legs 0/882 both ways). That explanation is a
    claim about an INTERACTION, and it predicts something falsifiable -- under an
    AND-join, where dead legs are common, stemming should HELP.

    Run as a 2x2 (tokenizer x join) the interaction term tests the story. If
    porter loses under both joins, my explanation is wrong and the honest reading
    is simply that stemming hurts this corpus.

    Only the OR-joins are rewritten; the phrase branch already AND-joins, so
    those queries are identical in both arms and contribute no interaction.
    """
    from localvectordb._filters import FTSQuerySanitization as S

    produced = S.sanitize_fts_query(query)
    if not isinstance(produced, str):
        return produced
    return produced.replace(" OR ", " AND ")


JOINS: Tuple[Tuple[str, Optional[object]], ...] = (
    ("or", None),  # None -> src's own sanitiser, i.e. what ships
    ("and", sanitize_and),
)
BASELINE = "unicode61/or"


def stem(terms: Sequence[str], tokenize: Optional[str]) -> List[str]:
    """Push terms through FTS5's OWN tokenizer and read back what it stored.

    Each term goes in as its own row, so ``fts5vocab(instance)`` maps row -> term.
    Guessing the stem in Python would test my Porter implementation rather than
    SQLite's, and the query side is tokenized by the table's tokenizer at MATCH
    time, so this is what the index actually looks up.
    """
    conn = sqlite3.connect(":memory:")
    clause = "" if tokenize is None else f", tokenize='{tokenize}'"
    conn.execute(f"CREATE VIRTUAL TABLE t USING fts5(body{clause})")
    conn.executemany("INSERT INTO t(rowid, body) VALUES (?,?)", enumerate(terms))
    conn.execute("CREATE VIRTUAL TABLE ti USING fts5vocab(t, instance)")
    conn.commit()
    out = dict(conn.execute("SELECT doc, term FROM ti"))
    return [out.get(i, terms[i]) for i in range(len(terms))]


def vocab_stats(texts: Sequence[str], tokenize: Optional[str], qterms: Sequence[str]) -> Dict[str, float]:
    """Vocabulary size and clamped-idf reach under a given tokenizer.

    Clamp reach must be measured at the unit under test: N sets the idf sign
    boundary, so the clamped vocabulary is a different set at every granularity.
    Stemming merges variants, which RAISES df and should push terms across that
    boundary -- the §1.2a prediction this function exists to check.
    """
    conn = sqlite3.connect(":memory:")
    clause = "" if tokenize is None else f", tokenize='{tokenize}'"
    conn.execute(f"CREATE VIRTUAL TABLE u USING fts5(body{clause})")
    conn.executemany("INSERT INTO u(rowid, body) VALUES (?,?)", enumerate(texts))
    conn.execute("CREATE VIRTUAL TABLE vr USING fts5vocab(u, row)")
    conn.commit()
    n = len(texts)
    df = {t: d for t, d, _c in conn.execute("SELECT term, doc, cnt FROM vr")}

    def clamped(d: int) -> bool:
        return math.log((n - d + 0.5) / (d + 0.5)) <= 0

    hit = tot = 0
    for term in stem(list(qterms), tokenize):
        if term in df:
            tot += 1
            hit += clamped(df[term])
    return {
        "vocab": float(len(df)),
        "clamped_vocab": float(sum(1 for d in df.values() if clamped(d))),
        "clamped_query_terms": (hit / tot) if tot else float("nan"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("qasper", "maud", "mldr", "nq"), default="qasper")
    p.add_argument("--model-key", default="egemma")
    p.add_argument("--coarse", choices=("sections", "documents"), default="sections")
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--vector-weight", type=float, default=VECTOR_WEIGHT)
    p.add_argument("--allow-embed", action="store_true")
    p.add_argument(
        "--join-control",
        action="store_true",
        help="Run the 2x2 (tokenizer x OR/AND join) and report the interaction. This TESTS the "
        "explanation for the OR-only result rather than assuming it: if stemming's upside is "
        "already spent by the OR-join, stemming should do better under AND, where dead legs are "
        "common. AND-joining is NOT a proposal -- §19.6 measured it as the wrong default.",
    )
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

    results: Dict[str, Dict[str, float]] = {}
    per_query_out: Dict[str, List[float]] = {}

    for lname, (uids, uvecs, texts, udocs) in levels.items():
        print(f"\n{'=' * 78}\n{args.dataset} / {lname} / {len(texts)} units\n{'=' * 78}")

        # Mechanical size of the change, before any nDCG: if the vocabulary
        # barely moves, a null result means the tokenizer did not fire.
        raw_terms = [t for q in qtexts for t in str(q).lower().split() if t.isalpha()]
        for label, tok in TOKENIZERS:
            st = vocab_stats(texts, tok, raw_terms)
            print(
                f"  {label:<20} vocabulary {int(st['vocab']):>7}   "
                f"clamped-idf terms {int(st['clamped_vocab']):>4}   "
                f"clamped query terms {100 * st['clamped_query_terms']:>5.1f}%"
            )
            results[f"{lname}|{label}|vocab"] = st["vocab"]
            results[f"{lname}|{label}|clamped_vocab"] = st["clamped_vocab"]
            results[f"{lname}|{label}|clamped_query_terms"] = st["clamped_query_terms"]

        arms: List[str] = []
        kw: Dict[str, List[Dict[str, float]]] = {}
        dead: Dict[str, int] = {}
        for tlabel, tok in TOKENIZERS:
            for jlabel, san in JOINS:
                if jlabel != "or" and not args.join_control:
                    continue
                label = f"{tlabel}/{jlabel}"
                arms.append(label)
                fts = FTS(uids, texts, udocs, sanitize=san, tokenize=tok)
                kw[label] = capture_keyword(fts, qtexts, q_scope)
                dead[label] = sum(1 for h in kw[label] if not h)
                print(
                    f"  {label:<20} dead keyword legs {dead[label]:>5}/{len(qtexts)} "
                    f"({100 * dead[label] / max(len(qtexts), 1):.1f}%)"
                )
                results[f"{lname}|{label}|dead"] = float(dead[label])

        # IS THE INTERACTION EVEN MEASURABLE HERE? When the AND-join kills every
        # leg, both tokenizer arms fall back to the identical vector ranking, so
        # "stemming under AND" is exactly 0.0000 BY CONSTRUCTION and the
        # interaction is mechanically -(the OR effect). MAUD does this: its
        # queries average 29 terms, and conjoining 29 terms matches nothing
        # (2752/2752 dead at chunks, 99.6% at sections). That produced a
        # significant-looking +0.0147 that is arithmetic, not evidence.
        degenerate = False
        if args.join_control:
            live = len(qtexts) - dead.get("unicode61/and", 0)
            live_p = len(qtexts) - dead.get("porter/and", 0)
            degenerate = min(live, live_p) < 0.05 * max(len(qtexts), 1)
            if degenerate:
                print(
                    f"  !! AND arm is DEGENERATE ({live} and {live_p} live legs of {len(qtexts)}): "
                    "both tokenizers collapse to the same vector-only ranking, so the interaction "
                    "below is -(the OR effect) by construction and is NOT evidence."
                )

        vec_cap = [v for v, _ in capture_arm(uids, uvecs, None, qv, qtexts, udocs, q_scope)]

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
                pq: Dict[str, np.ndarray] = {}
                rec: Dict[str, float] = {}
                for label in arms:
                    captured = list(zip(vec_cap, kw[label], strict=True))
                    ranked = to_ranked(blend_arm(captured, owner, vw, use_keyword=True))
                    pq[label] = score(ranked, qids, qrels)
                    rec[label] = recall(ranked, qids, qrels)

                tag = f"{args.dataset}/{lname}/{tname}/{arm_name}"
                print(f"\n=== {tag} · vw={vw:.2f} ===")
                print(f"  {'arm':<22}{'nDCG@10':>9}{'recall@10':>11}{'delta':>10}{'95% CI':>21}{'overlap':>9}")
                for label in arms:
                    st = paired(pq[label], pq[BASELINE])
                    ci = f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
                    star = "*" if (st["ci_lo"] > 0 or st["ci_hi"] < 0) else " "
                    ov = overlap(kw[label], kw[BASELINE])
                    print(
                        f"  {label:<22}{pq[label].mean():>9.4f}{rec[label]:>11.4f}"
                        f"{st['delta']:>+10.4f}{ci:>21}{star}{ov:>9.3f}"
                    )
                    results[f"{tag}|{label}"] = float(pq[label].mean())
                    results[f"{tag}|{label}|delta"] = st["delta"]
                    per_query_out[f"{tag}|{label}"] = [float(v) for v in pq[label]]
                print(f"  keyword top-10 changed on {100 * churn(kw['porter/or'], kw[BASELINE]):.1f}% of queries")

                # THE INTERACTION. Stemming's effect under each join; if the
                # "OR-join already spent the upside" story holds, the AND column
                # is the more positive of the two.
                if args.join_control:
                    d_or = pq["porter/or"].mean() - pq["unicode61/or"].mean()
                    d_and = pq["porter/and"].mean() - pq["unicode61/and"].mean()
                    inter = paired(pq["porter/and"] - pq["unicode61/and"], pq["porter/or"] - pq["unicode61/or"])
                    star = "*" if (inter["ci_lo"] > 0 or inter["ci_hi"] < 0) else " "
                    note = "  [DEGENERATE -- arithmetic, not evidence]" if degenerate else ""
                    print(
                        f"  stemming effect: {d_or:+.4f} under OR, {d_and:+.4f} under AND; "
                        f"interaction {inter['delta']:+.4f} "
                        f"[{inter['ci_lo']:+.4f},{inter['ci_hi']:+.4f}]{star}{note}"
                    )
                    results[f"{tag}|interaction"] = inter["delta"]
                    results[f"{tag}|interaction_degenerate"] = float(degenerate)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "model": spec.model,
                    "coarse": args.coarse,
                    "queries": len(qids),
                    "qids": [str(q) for q in qids],
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
