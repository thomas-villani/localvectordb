"""DIAGNOSTIC: what does the phrase-query AND-join cost, and which fix is right?

THE BUG. `FTSQuerySanitization.handle_phrase_query` conjoins every part of a
query containing a quoted phrase -- including stopwords:

    'How is a "chunk of posts" defined in this work?'
     -> '"How" AND "is" AND "a" AND "chunk of posts" AND "defined" AND ...'

The plain-text branch OR-joins for an explicitly stated reason: "requiring every
term (including stopwords) to appear makes almost any real-world sentence match
nothing at all." The phrase branch does the condemned thing.

WHAT IT COSTS, MEASURED FIRST. At section granularity, queries taking this branch
return ZERO keyword hits 99.8% of the time on MAUD (565/566) and 100% on qasper
(5/5), against 0% dead and ~70-80 mean hits for the plain-OR branch. On MAUD that
is 20.6% of the query set, so MAUD's published +0.0582 BM25 contribution was
earned by 79.4% of its queries with the rest contributing exactly nothing. That
number is a FLOOR, and the corrected value is unknown until this is fixed --
scaling it by 1/0.794 would be an extrapolation wearing a measurement's clothes.

WHY TWO CANDIDATES AND NOT ONE. Quoting means different things and the choice is
not obvious from taste:

  * `phrase_required` -- the phrase is a hard constraint, the loose terms only
    rank: `"chunk of posts" AND ("How" OR "is" OR ...)`. Honours what a quote
    conventionally means. Risk: if the phrase appears nowhere, the leg is still
    dead (though it then degrades to the vector leg, which is the designed
    fallback).
  * `all_or` -- everything OR-joined with the phrase kept as a phrase unit.
    Cannot ever return nothing, and is consistent with the plain-text branch's
    stated philosophy. Risk: the quote stops constraining anything at all.

Both are swept against `current` on the same pools with paired CIs, so the choice
is made on evidence. Zero embedding: vectors come from the `hier_embed` cache.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

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
from localvectordb._filters import FTSQuerySanitization as S  # noqa: E402

logger = logging.getLogger("fts_phrase")


def _split_phrases(query: str) -> tuple[List[str], List[str]]:
    """Split a query into (quoted phrases, loose terms), both cleaned and quoted.

    Scans exactly as ``handle_phrase_query`` does, including its treatment of an
    unclosed trailing quote as a phrase, so the ONLY difference between the
    candidates below and the shipped function is how the pieces are joined.
    """
    phrases: List[str] = []
    loose: List[str] = []
    current, in_quote = "", False
    for char in query:
        if char == '"':
            if in_quote:
                cleaned = S.clean_term(current)
                if cleaned:
                    phrases.append(f'"{cleaned}"')
            else:
                loose.extend(S.quote_terms(current))
            current, in_quote = "", not in_quote
        else:
            current += char
    if current.strip():
        if in_quote:
            cleaned = S.clean_term(current)
            if cleaned:
                phrases.append(f'"{cleaned}"')
        else:
            loose.extend(S.quote_terms(current))
    return phrases, loose


def sanitize_current(query: str) -> str:
    return S.sanitize_fts_query(query)


def _wrap(query: str, join: Callable[[List[str], List[str]], str]) -> str:
    """Apply ``join`` to the phrase branch only; every other branch is untouched.

    Leaving the plain-text and boolean branches alone is what makes this a
    measurement of the phrase rule rather than of a whole new sanitiser.
    """
    text = (query or "").strip()
    if not text or '"' not in text:
        return S.sanitize_fts_query(query)
    if text.startswith('"') and text.endswith('"') and text.count('"') == 2:
        return S.sanitize_fts_query(query)  # a fully-quoted query IS the phrase
    phrases, loose = _split_phrases(text)
    if not phrases:
        return S.sanitize_fts_query(query)
    return join(phrases, loose)


def sanitize_phrase_required(query: str) -> str:
    """Phrases mandatory and AND-ed; loose terms OR-ed into one ranking group.

    The loose group is parenthesised so FTS5's precedence (NOT, then AND, then
    OR) cannot regroup it into ``phrase AND t1`` OR ``t2``.
    """

    def join(phrases: List[str], loose: List[str]) -> str:
        head = " AND ".join(phrases)
        if not loose:
            return head
        return f"{head} AND (" + " OR ".join(loose) + ")"

    return _wrap(query, join)


def sanitize_all_or(query: str) -> str:
    """Everything OR-ed, the phrase surviving as a single phrase token."""

    def join(phrases: List[str], loose: List[str]) -> str:
        return " OR ".join([*phrases, *loose])

    return _wrap(query, join)


VARIANTS: Dict[str, Callable[[str], str]] = {
    "current": sanitize_current,
    "phrase_required": sanitize_phrase_required,
    "all_or": sanitize_all_or,
}
BASELINE = "current"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["qasper", "maud", "mldr"], default="maud")
    ap.add_argument("--model-key", default="egemma")
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--vector-weight", type=float, default=VECTOR_WEIGHT)
    ap.add_argument("--allow-embed", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
    else:
        from benchmarks.qasper_data import load_qasper

        bench = load_qasper(split="dev", max_papers=args.max_papers)
        units = load_units(bench, None)

    spec = MODEL_POOL[args.model_key]
    doc_enc = PrefixedEncoder(spec, spec.doc_prefix)
    qry_enc = PrefixedEncoder(spec, spec.query_prefix)
    if not args.allow_embed:
        miss = doc_enc.count_misses(units.chunk_texts)[1] + qry_enc.count_misses(units.query_texts)[1]
        if miss:
            raise SystemExit(f"{miss} vectors are not cached; pass --allow-embed to encode them.")

    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1.0, n)

    cv = unit(doc_enc.encode(units.chunk_texts, normalize=False))
    qv = unit(qry_enc.encode(units.query_texts, normalize=False))
    qids, qtexts = list(units.query_ids), list(units.query_texts)
    chunk_uid = [f"c{i}" for i in range(len(units.chunk_texts))]
    chunk_docs = list(units.chunk_doc)

    q_scope: Sequence[str] | None = None
    if args.dataset == "maud":
        q_scope = [str(q).split("||", 1)[0] for q in qids]

    # How many queries each variant actually rescues, before any nDCG is quoted:
    # a retrieval delta with no change in dead-leg count would mean the fix is
    # doing something other than what it claims.
    affected = [q for q in qtexts if '"' in (q or "").strip()]
    print(f"\n{len(affected)}/{len(qtexts)} queries contain a quote ({100*len(affected)/max(1,len(qtexts)):.1f}%)")

    chunk_to_sec = {u: [v] for u, v in zip(chunk_uid, units.chunk_section, strict=True)}
    chunk_to_doc = {u: [v] for u, v in zip(chunk_uid, chunk_docs, strict=True)}

    targets = [("section", bench.section_qrels)]
    if args.dataset != "maud" and bench.doc_qrels:
        targets.append(("doc", bench.doc_qrels))

    per_variant: Dict[str, Dict[str, np.ndarray]] = {}
    for name, fn in VARIANTS.items():
        fts = FTS(chunk_uid, units.chunk_texts, chunk_docs, sanitize=fn)
        scopes = list(q_scope) if q_scope is not None else [None] * len(qtexts)
        dead = sum(1 for qt, sc in zip(qtexts, scopes, strict=True) if not fts.search(qt, SEARCH_K, sc))
        cap = capture_arm(chunk_uid, cv, fts, qv, qtexts, chunk_docs, q_scope)
        per_variant[name] = {}
        for tname, qrels in targets:
            if not qrels:
                continue
            owner = chunk_to_sec if tname == "section" else chunk_to_doc
            ranked = to_ranked(blend_arm(cap, owner, args.vector_weight, use_keyword=True))
            per_variant[name][tname] = score(ranked, qids, qrels)
            per_variant[name][f"{tname}_recall"] = np.array([recall(ranked, qids, qrels)])
        print(f"  {name:<16} dead keyword legs: {dead}/{len(qtexts)} ({100*dead/len(qtexts):.1f}%)")

    for tname, _ in targets:
        print(f"\n=== {args.dataset} · target {tname} · hybrid vw={args.vector_weight} ===")
        base = per_variant[BASELINE][tname]
        print(f"  {BASELINE:<16} nDCG@10 = {base.mean():.4f}")
        for name in VARIANTS:
            if name == BASELINE:
                continue
            arm = per_variant[name][tname]
            stat = paired(arm, base)
            if stat["ci_lo"] > 0:
                verdict = "SIGNIFICANT WIN"
            elif stat["ci_hi"] < 0:
                verdict = "SIGNIFICANT LOSS"
            else:
                verdict = "indistinguishable"
            print(
                f"  {name:<16} nDCG@10 = {arm.mean():.4f}  delta={stat['delta']:+.4f} "
                f"95% CI [{stat['ci_lo']:+.4f}, {stat['ci_hi']:+.4f}] p={stat['p']:.3f}  {verdict}"
            )


if __name__ == "__main__":
    main()
