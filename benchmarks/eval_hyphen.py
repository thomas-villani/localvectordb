"""`clean_term` has TWO separator bugs, not one, and they pull in opposite directions.

`FTSQuerySanitization.clean_term` runs `re.sub(r"[^\\w\\s'-]", "", term)` and then
`quote_terms` wraps each whitespace token in double quotes. That single regex
treats separators in two different ways, and neither matches what FTS5 does.

DEFECT 1 -- HYPHENS AND APOSTROPHES ARE KEPT, becoming ADJACENCY PHRASES. FTS5's
unicode61 tokenizer splits on them, so a quoted token containing one is not a
term at all:

    state-of-the-art   ->   "state-of-the-art"   ->   state THEN of THEN the THEN art
    don't              ->   "don't"              ->   don THEN t

That is a conjunction plus a word-order constraint, smuggled into what the rest
of `sanitize_fts_query` treats as one OR-ed term, and it silently breaks that
method's own promise that "terms are OR-joined so BM25 can rank partial matches".

DEFECT 2 -- EVERY OTHER SEPARATOR IS DELETED, WELDING THE FRAGMENTS TOGETHER:

    Aeroflot/Polar -> aeroflotpolar      J.K.  -> jk
    best/worst     -> bestworst          (c.f. -> cf

Those are single tokens occurring in NO document, so the term does not rank
badly -- it vanishes. But the same deletion is sometimes exactly right
(`dataset(s)` -> `datasets`, `8,640` -> `8640`), which is why the repair has to
be conditional rather than "also split on these".

FOUR ARMS, because "bag of words" changes several things at once:

    shipped   sanitize_fts_query as it stands.
    split     DEFECT 1 only: a whitespace token FTS5 would split becomes its
              tokens, OR-ed. Every branch -- user-quoted phrases, booleans, the
              fully-quoted fast path -- untouched, so a deliberate phrase stays a
              phrase and only accidental ones are split.
    vocab     DEFECT 2 only: a welded term that matches NOTHING falls back to its
              raw tokens. Fires only where shipped already scores zero, so it
              cannot lose a working match -- see `vocab_sanitizer`.
    bow       every FTS5 token of the raw query, quoted separately and OR-ed. The
              crude version, and it changes BOTH defects plus user quotes and
              punctuation handling -- so `bow` is not attributable on its own.

Reporting `split` alone would leave "maybe the gain is really about quotes" open;
reporting `bow` alone answers a different question than the one asked. The set
separates them, and that matters here: the original §1.4 lead was a `bow`-shaped
+0.0125 on MLDR that was assumed to be hyphens.

EACH ARM IS SCORED ON ITS OWN ELIGIBLE SUBSET, since the two defects fire at very
different rates (11.5% vs ~1% of qasper queries) and a shared mask would dilute
the rarer one. Each also gets a PLACEBO: an arm's delta on the queries it cannot
touch must be exactly zero, or its subset number means nothing.

THE SUBSET IS THE EFFECT SIZE, and the corpus mean is the shipping impact. If 4%
of queries contain a hyphenated token, a real +0.10 on those queries shows up as
+0.004 overall. Both are printed, and neither is the headline on its own: the
subset delta says whether the mechanism is real, the corpus delta says whether
changing shipped code is worth it. Quoting only one of them would mislead in
opposite directions.

    python benchmarks/eval_hyphen.py --dataset mldr --model-key openai

--- SECOND MODE: --ocr-sweep -------------------------------------------------

The query side is only half of it. OCR'd and PDF-extracted text carries LINE-BREAK
HYPHENATION -- "informa-\ntion" -- which after extraction is a hyphen inside what
should be one word. FTS5 splits it, so the corpus term `information` simply does
not exist in that document and no query sanitisation can recover it. That is an
INDEX-side loss and it is invisible to every corpus in this study, all of which
are clean.

`--ocr-sweep` injects synthetic line-break hyphenation at a rate and measures the
damage, then measures a rejoin repair. Note what it does NOT model: real OCR also
substitutes characters and drops words, so this isolates hyphenation alone and is
an UNDERSTATEMENT of wild-PDF damage, not an estimate of it.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sqlite3
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.eval_prf import bow_sanitizer, build_cell, tokenize_queries  # noqa: E402
from benchmarks.eval_section_bm25 import (  # noqa: E402
    FTS,
    SEARCH_K,
    VECTOR_WEIGHT,
    blend_arm,
    capture_arm,
    paired,
    score,
    to_ranked,
)
from localvectordb._filters import _HAS_ALPHANUMERIC, FTSQuerySanitization  # noqa: E402

logger = logging.getLogger("hyphen")

# Only these survive clean_term's character filter, so only these can produce a
# multi-token quoted string. Asserted against FTS5 rather than assumed -- see
# `audit_tokens`.
SPLITTERS = re.compile(r"[-']")


def fts5_tokens(words: Sequence[str]) -> List[List[str]]:
    """FTS5's own tokenisation of each word, via a throwaway index."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE w USING fts5(body)")
    conn.executemany("INSERT INTO w(rowid, body) VALUES (?, ?)", enumerate(words))
    conn.execute("CREATE VIRTUAL TABLE wv USING fts5vocab(w, instance)")
    conn.commit()
    out: List[List[str]] = [[] for _ in words]
    for term, doc, _col, _off in conn.execute("SELECT term, doc, col, offset FROM wv ORDER BY doc, offset"):
        out[doc].append(term)
    conn.close()
    return out


def split_sanitizer(qtexts: Sequence[str], chars: str = "-'") -> Callable[[str], str]:
    """Shipped sanitisation with ONLY `quote_terms` changed.

    Patching the one method rather than reimplementing `sanitize_fts_query` is
    deliberate: the branch structure (fully-quoted fast path, `handle_phrase_query`,
    `handle_boolean_query`) is exactly what this arm must NOT change, and a
    reimplementation would silently drift from it. A user-written phrase still
    stays a phrase -- only tokens that became phrases BY ACCIDENT are split.

    `chars` restricts which separator triggers the split, which turns "the effect
    is corpus-dependent" into a testable mechanism. The two separators are not
    the same kind of thing:

      HYPHEN     `state-of-the-art` -- the phrase is MEANINGFUL. Its tokens in
                 that order are the compound; OR-ing them adds `of` and `the`.
      APOSTROPHE `don't` -> don THEN t -- the phrase is a tokenisation artefact
                 carrying no information, and both forms match the same text.

    So the prediction is that splitting hyphens costs and splitting apostrophes
    is a wash, and the corpus ordering follows which separator dominates -- not
    the corpus. Passing `chars` separately is what tests that.
    """
    words = sorted({w for t in qtexts for w in t.split()})
    cleaned = [FTSQuerySanitization.clean_term(w) for w in words]
    memo = dict(zip(cleaned, fts5_tokens(cleaned), strict=True))

    original = FTSQuerySanitization.quote_terms

    def patched(text: str) -> List[str]:
        quoted: List[str] = []
        for word in text.split():
            term = FTSQuerySanitization.clean_term(word)
            # Shipped's own guard, replicated EXACTLY. Dropping it would make
            # this arm differ from `shipped` for a second reason, and the placebo
            # cannot catch that -- an all-punctuation term is not a "multi-token"
            # term, so such a query sits in the ineligible set where the arm is
            # asserted to be inert.
            if not term or not _HAS_ALPHANUMERIC.search(term):
                continue
            if not any(c in term for c in chars):
                quoted.append(f'"{term}"')
                continue
            toks = memo.get(term)
            if toks is None:
                toks = fts5_tokens([term])[0]
                memo[term] = toks
            quoted.extend(f'"{t}"' for t in toks)
        return quoted

    def _san(query: str) -> str:
        FTSQuerySanitization.quote_terms = staticmethod(patched)  # type: ignore[method-assign]
        try:
            return FTSQuerySanitization.sanitize_fts_query(query)
        finally:
            FTSQuerySanitization.quote_terms = original  # type: ignore[method-assign]

    return _san


def corpus_vocab(texts: Sequence[str]) -> set:
    """Every term FTS5 actually indexes. Cheap: `fts5vocab(row)` is one row per term."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE c USING fts5(body)")
    conn.executemany("INSERT INTO c(rowid, body) VALUES (?, ?)", enumerate(texts))
    conn.execute("CREATE VIRTUAL TABLE cv USING fts5vocab(c, row)")
    conn.commit()
    out = {t for (t,) in conn.execute("SELECT term FROM cv")}
    conn.close()
    return out


def vocab_sanitizer(qtexts: Sequence[str], vocab: set) -> Tuple[Callable[[str], str], set]:
    """Shipped, except a term that matches NOTHING falls back to its raw tokens.

    THE SECOND DEFECT IN `clean_term`, and it is not the hyphen one. The regex
    `[^\\w\\s'-]` DELETES every other separator rather than splitting on it, so
    the surrounding fragments are welded together:

        Aeroflot/Polar -> aeroflotpolar      J.K.  -> jk
        best/worst     -> bestworst          (c.f. -> cf

    Those are single FTS5 tokens that occur in no document, so the term
    contributes exactly zero -- it does not rank badly, it disappears. Note the
    deletion is sometimes RIGHT: `dataset(s)` -> `datasets` and `8,640` -> `8640`
    both recover the real term, which is why blanket splitting is the wrong fix
    and why `split` is not this arm.

    THE RULE IS SAFE BY CONSTRUCTION: it fires only when the shipped term is
    absent from the index, i.e. only where shipped already scores nothing, and
    only if at least one raw token IS present. It cannot take away a match that
    was working, so unlike `split` it has no downside to trade off -- the only
    open question is how often it fires.

    Hyphenated tokens are deliberately untouched. They are multi-token PHRASES,
    not welded single tokens, and `split` measures what happens to those.
    """
    words = sorted({w for t in qtexts for w in t.split()})
    cleaned = [FTSQuerySanitization.clean_term(w) for w in words]
    clean_toks = fts5_tokens(cleaned)
    raw_toks = fts5_tokens(words)
    repair: Dict[str, List[str]] = {}
    fires: set = set()
    dead = 0
    for w, c, ct, rt in zip(words, cleaned, clean_toks, raw_toks, strict=True):
        if len(ct) != 1 or ct[0] in vocab:
            continue
        dead += 1
        # A dead term is only REPAIRABLE if the raw form splits into something
        # the index actually contains. Most dead terms are simply rare words or
        # typos, and no sanitisation can rescue those -- conflating the two would
        # make the arm's eligible subset far larger than the set of queries it
        # can move, and its placebo would then pass trivially.
        if len(rt) > 1 and any(t in vocab for t in rt):
            repair[c] = [t for t in rt if t in vocab]
            fires.add(w)
    logger.info(
        "query words: %d distinct, %d match nothing, %d of those are WELDED and repairable",
        len(words),
        dead,
        len(repair),
    )

    original = FTSQuerySanitization.quote_terms

    def patched(text: str) -> List[str]:
        quoted: List[str] = []
        for word in text.split():
            term = FTSQuerySanitization.clean_term(word)
            if not term or not _HAS_ALPHANUMERIC.search(term):  # shipped's guard, replicated
                continue
            alt = repair.get(term)
            quoted.extend(f'"{t}"' for t in alt) if alt else quoted.append(f'"{term}"')
        return quoted

    def _san(query: str) -> str:
        FTSQuerySanitization.quote_terms = staticmethod(patched)  # type: ignore[method-assign]
        try:
            return FTSQuerySanitization.sanitize_fts_query(query)
        finally:
            FTSQuerySanitization.quote_terms = original  # type: ignore[method-assign]

    return _san, fires


def audit_tokens(qtexts: Sequence[str]) -> Tuple[List[bool], Dict[str, int]]:
    """Which queries contain a whitespace token FTS5 splits, and on what.

    Returns the per-query affected flag and a count of splitting characters, so
    "it is hyphens" is measured rather than assumed -- if apostrophes or digits
    dominate on some corpus, the fix is the same but the story is not.
    """
    vocab = sorted({w for t in qtexts for w in t.split()})
    cleaned = [FTSQuerySanitization.clean_term(w) for w in vocab]
    toks = fts5_tokens(cleaned)
    multi = {v for v, c, tk in zip(vocab, cleaned, toks, strict=True) if len(tk) > 1}
    chars: Dict[str, int] = {}
    for v in multi:
        for ch in set(FTSQuerySanitization.clean_term(v)):
            if SPLITTERS.match(ch):
                chars[ch] = chars.get(ch, 0) + 1
    affected = [any(w in multi for w in t.split()) for t in qtexts]
    return affected, chars


def hyphenate(text: str, rate: float, rng: random.Random) -> str:
    """Inject line-break hyphenation: `information` -> `informa- tion`.

    FTS5 tokenises `informa- tion` and `informa-tion` identically (two tokens),
    so which side of the break the whitespace lands on does not matter here; the
    space form is used because it is what most extractors emit and because it is
    the form a rejoin repair can key on.
    """
    out = []
    for word in text.split(" "):
        if len(word) >= 6 and rng.random() < rate:
            cut = rng.randint(3, len(word) - 3)
            out.append(f"{word[:cut]}- {word[cut:]}")
        else:
            out.append(word)
    return " ".join(out)


def dehyphenate(text: str) -> str:
    """Rejoin `X- Y` across whitespace. Deliberately does NOT touch `X-Y`.

    The whitespace is the line-break signal. A repair that also joined `X-Y`
    would destroy every legitimate compound ("cross-encoder" -> "crossencoder"),
    which is why hyphenated compounds are left exactly as they are. That makes
    this repair conservative: it cannot fix a hyphenation whose whitespace the
    extractor already ate.
    """
    return re.sub(r"(\w)-\s+(\w)", r"\1\2", text)


def evaluate(
    kw_caps: Sequence[Dict[str, float]],
    vec_cap: Sequence[Dict[str, float]],
    owner: Optional[Dict[str, List[str]]],
    vw: float,
    qids: Sequence[str],
    qrels: Dict[str, Dict[str, float]],
) -> np.ndarray:
    captured = list(zip(vec_cap, kw_caps, strict=True))
    return score(to_ranked(blend_arm(captured, owner, vw, use_keyword=True)), qids, qrels)


def report(
    tag: str,
    pq: Dict[str, np.ndarray],
    order: Sequence[str],
    base_key: str,
    subsets: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    base = pq[base_key]
    print(f"\n=== {tag} · n={len(base)} ===")
    print(f"  {'arm':>10} {'nDCG@10':>9} {'delta':>9} {'95% CI':>20}")
    for k in order:
        st = paired(pq[k], base)
        ci = f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
        star = "*" if (st["ci_lo"] > 0 or st["ci_hi"] < 0) else " "
        print(f"  {k:>10} {pq[k].mean():>9.4f} {st['delta']:>+9.4f} {ci:>20}{star}")
        out[k] = {"ndcg@10": float(pq[k].mean()), **st}
    if not subsets:
        return out
    # EACH ARM GETS ITS OWN SUBSET. `split` can only move a query containing a
    # token FTS5 splits; `vocab` can only move one containing a welded term that
    # matches nothing. Scoring both against a single shared mask would dilute
    # whichever arm fires more rarely, and the rates differ by an order of
    # magnitude here (11.5% vs ~1%).
    for k, mask in subsets.items():
        if k not in pq:
            continue
        n_aff = int(mask.sum())
        if n_aff == 0:
            print(f"  {k.upper()} SUBSET: 0 queries eligible -- this corpus cannot answer for this arm")
            continue
        st = paired(pq[k][mask], base[mask])
        ci = f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
        star = "*" if (st["ci_lo"] > 0 or st["ci_hi"] < 0) else " "
        print(
            f"  {k.upper()} SUBSET ({n_aff}/{len(base)} = {100 * n_aff / len(base):.1f}%): "
            f"{pq[k][mask].mean():.4f} {st['delta']:+.4f} {ci}{star}"
        )
        out[f"{k}|affected"] = {"n": n_aff, "ndcg@10": float(pq[k][mask].mean()), **st}
        # PLACEBO. An arm cannot move a query it is not eligible for, so a
        # non-zero delta on the complement is a bug in the arm -- and it would
        # invalidate the subset number just printed.
        rest = ~mask
        if rest.sum():
            ps = paired(pq[k][rest], base[rest])
            flag = "" if abs(ps["delta"]) < 1e-9 else f"   !! {k} changed a query it cannot touch"
            print(f"    placebo (ineligible, n={int(rest.sum())}): {ps['delta']:+.6f}{flag}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("qasper", "maud", "mldr", "nq"), default="qasper")
    p.add_argument("--model-key", default="egemma")
    p.add_argument("--coarse", choices=("sections", "documents"), default="sections")
    p.add_argument("--level", choices=("chunks", "coarse"), default="chunks")
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--vector-weight", type=float, default=VECTOR_WEIGHT)
    p.add_argument("--ocr-sweep", action="store_true", help="inject line-break hyphenation into the CORPUS")
    p.add_argument("--ocr-rates", type=float, nargs="*", default=[0.02, 0.05, 0.10, 0.20])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-embed", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("localvectordb.chunking").setLevel(logging.ERROR)

    cell = build_cell(args)
    uids, texts, udocs = cell.uids, cell.texts, cell.udocs
    qids, qtexts, q_scope = cell.qids, cell.qtexts, cell.q_scope
    scopes = [q_scope[i] if q_scope else None for i in range(len(qtexts))]

    vec_cap = [v for v, _ in capture_arm(uids, cell.uv, None, cell.qv, qtexts, udocs, q_scope)]

    affected_list, split_chars = audit_tokens(qtexts)
    affected = np.array(affected_list)
    logger.info(
        "%d/%d queries contain a token FTS5 splits; splitting chars %s",
        int(affected.sum()),
        len(qtexts),
        split_chars or "{}",
    )

    caps: Dict[str, List[Dict[str, float]]] = {}
    if args.ocr_sweep:
        # The QUERY side is held at shipped throughout: this mode is about what
        # the extractor did to the corpus, and changing both at once would make
        # the rates unreadable.
        for rate in [0.0] + list(args.ocr_rates):
            rng = random.Random(args.seed)
            damaged = [hyphenate(t, rate, rng) for t in texts] if rate else list(texts)
            for repair, fn in (("raw", None), ("rejoin", dehyphenate)):
                if rate == 0.0 and repair == "rejoin":
                    continue
                body = [fn(t) for t in damaged] if fn else damaged
                idx = FTS(uids, body, udocs)
                label = "clean" if rate == 0.0 else f"{rate:g}/{repair}"
                caps[label] = [idx.search(t, SEARCH_K, scopes[i]) for i, t in enumerate(qtexts)]
                logger.info("built OCR arm %s", label)
        order = list(caps)
        base_key = "clean"
        subsets = None
    else:
        qtok = dict(zip(qtexts, tokenize_queries(qtexts), strict=True))
        vocab_san, fires = vocab_sanitizer(qtexts, corpus_vocab(texts))
        for label, san in (
            ("shipped", None),
            ("split", split_sanitizer(qtexts)),
            ("split-hy", split_sanitizer(qtexts, chars="-")),
            ("split-ap", split_sanitizer(qtexts, chars="'")),
            ("vocab", vocab_san),
            ("bow", bow_sanitizer(qtok)),
        ):
            idx = FTS(uids, texts, udocs, sanitize=san)
            caps[label] = [idx.search(t, SEARCH_K, scopes[i]) for i, t in enumerate(qtexts)]
        order = ["shipped", "split", "split-hy", "split-ap", "vocab", "bow"]
        base_key = "shipped"

        def eligible(chars: str) -> np.ndarray:
            return np.array(
                [
                    any(
                        any(c in FTSQuerySanitization.clean_term(w) for c in chars)
                        and len(fts5_tokens([FTSQuerySanitization.clean_term(w)])[0]) > 1
                        for w in t.split()
                    )
                    for t in qtexts
                ]
            )

        subsets = {
            "split": affected,
            "split-hy": eligible("-"),
            "split-ap": eligible("'"),
            "vocab": np.array([any(w in fires for w in t.split()) for t in qtexts]),
        }

    bench = cell.bench
    targets = [("section", bench.section_qrels), ("doc", bench.doc_qrels)]
    if args.dataset == "maud":
        targets = [("section", bench.section_qrels)]

    results: Dict[str, dict] = {}
    for tname, qrels in targets:
        if not qrels:
            continue
        if args.level == "chunks":
            owner = cell.owner_sec if tname == "section" else cell.owner_doc
        else:
            owner = None if tname == "section" else cell.owner_doc
            if args.coarse == "documents" and tname == "section":
                continue
        for arm_name, vw in (("keyword", 0.0), ("hybrid", args.vector_weight)):
            pq = {k: evaluate(v, vec_cap, owner, vw, qids, qrels) for k, v in caps.items()}
            tag = f"{args.dataset}/{args.level}/{tname}/{arm_name}"
            got = report(tag, pq, order, base_key, subsets)
            for k, v in got.items():
                results[f"{tag}|{k}"] = v

    if args.out:
        payload = {
            "config": vars(args) | {"out": str(args.out)},
            "n_affected": int(affected.sum()),
            "n_queries": len(qtexts),
            "split_chars": split_chars,
            "results": results,
        }
        args.out.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
