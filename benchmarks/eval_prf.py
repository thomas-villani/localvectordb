"""Pseudo-relevance feedback, and whether a DECORRELATED feedback set drifts less.

THE QUESTION. RM3 is one of the most durable results in classical IR: run the
query, treat the top-k as if it were relevant, harvest distinctive terms, re-run.
Its named failure mode is QUERY DRIFT -- when the top-k is off-topic the
expansion amplifies the error. `KEYWORD-STRATEGIES.md` §1.4 proposes a variant:
seed the feedback set from the VECTOR or HYBRID list instead of from BM25's own,
so a retriever that fails on different queries damps the drift.

Dense PRF is well studied (Vector-PRF, ColBERT-PRF, ANCE-PRF) but all of it is
same-modality -- dense results feeding dense queries. Using a decorrelated
retriever's output as the feedback set for the OTHER retriever is the variant
this file measures.

WHY IT IS A REAL PREDICTION AND NOT A HUNCH. `dual-embedding-findings.md` §3
measured per-query rank-quality correlation between arms: same-model 0.64-0.80,
cross-model 0.47-0.61. Different retrievers fail on different queries, and drift
is precisely a correlated-failure problem. So the arms below are ordered by how
decorrelated the feedback source is from the leg being expanded:

    rm3      feedback = the keyword leg's own top-k   (fully correlated)
    xprf-h   feedback = the hybrid top-k              (partly -- it CONTAINS bm25)
    xprf-v   feedback = the pure vector top-k         (fully decorrelated)

THREE SEPARABLE CLAIMS LIVE IN THAT LADDER, and an earlier version of this
header collapsed them into one -- "the gradient runs rm3 < xprf-h < xprf-v" --
which then misreported the qasper result by crediting the proposal with a
prediction it never made. Keep them apart:

  A. cross-modal beats same-modal:      xprf-h  >  rm3      (the actual proposal)
  B. hybrid is the right cross seed:    xprf-h  >  xprf-v   (its parenthetical)
  C. decorrelation is monotonic:        xprf-v  >  xprf-h   (an extrapolation)

B and C point in OPPOSITE directions, so they cannot both hold and no single
"gradient" summarises the table. On qasper: A fails narrowly, B holds 4/4, C is
falsified. Report the three separately.

If instead the gain just tracks how good the feedback set was, it will track
feedback precision and nothing else -- which is why that is measured per arm
rather than assumed.

WHY THIS NEEDS A HAND-ROLLED BM25. RM3 produces a WEIGHTED term vector, and
FTS5's MATCH has no per-term weight -- `bm25()`'s only weights are per column.
So the scoring moves out of `bm25()` into `LexIndex` below, which reads term
frequencies and document lengths from `fts5vocab` -- FTS5's OWN tokenizer -- so
the only thing that changes is who multiplies the numbers. That also puts `b`,
BM25+ and SDM in reach for the first time (§1.2, §1.5, §1.6); they are the
"needs scoring outside bm25()" tier, and this is that scorer.

FOUR CONTROLS, because a hand-rolled scorer can quietly not be the shipped one:

  1. CALIBRATION. `LexIndex` at uniform weights must reproduce FTS5's own
     `bm25()` top-10. Reported against TWO references: against the same bag of
     words (which gates, and runs at 100%), and against shipped sanitisation,
     which is lower purely because src quotes each whitespace token, so a
     hyphenated word becomes an adjacency-constrained PHRASE that a positionless
     scorer cannot express. Naming the gap keeps it a known difference in the
     query model rather than an unexplained discrepancy.
  1b. THE SAME COMPARISON END TO END. `none` (this file's BM25) and `shipped`
     (FTS5's) are carried as two ARMS through the same fusion, roll-up and
     scoring path. Control 1 compares two orderings in isolation and is blind to
     how they are consumed downstream -- it sat at 100% while a sign-convention
     mismatch inverted every ranking that reached `fuse`. A component verified
     against a reference can still be wrong about its own interface, and only an
     end-to-end control sees that.
  2. FEEDBACK PRECISION. Share of each arm's feedback set that is genuinely
     relevant, plus the per-query deltas split by whether the feedback set
     contained ANY relevant unit. Separates "decorrelated" from "just better".
  3. ORACLE. An arm whose feedback set IS the gold-relevant units. The ceiling
     on what expansion can buy here. Without it a null result cannot distinguish
     "PRF does not work on this corpus" from "our feedback sets are too weak",
     and those want opposite follow-ups. It has already earned its place twice:
     it caught the sign bug (a perfect feedback set scoring WORST is not a
     finding, it is a defect) and it caught the term-selection defect before
     that (see ``rm3_weights``).

Zero embedding: vectors come from the `hier_embed` disk cache.

    python benchmarks/eval_prf.py --dataset qasper
    python benchmarks/eval_prf.py --dataset maud --model-key openai
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from array import array
from collections import defaultdict
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
    fuse,
    paired,
    recall,
    score,
    to_ranked,
)

logger = logging.getLogger("prf")

# FTS5's hardcoded constants, so the scorer's neutral point IS shipped behaviour.
K1_SHIPPED = 1.2
B_SHIPPED = 0.75

# RM3's three knobs, at the values the literature converged on. They are NOT
# swept by default: every parameter in this study has turned out to be a
# per-corpus argmax, and the point of this file is the ARM comparison. A sweep
# picked on the queries it is scored on would flatter whichever arm has the most
# room, which is exactly the arm the hypothesis predicts wins.
FB_K = 10  # feedback documents
FB_TERMS = 10  # expansion terms kept
ALPHA = 0.5  # weight on the original query


class LexIndex:
    """BM25 over FTS5's tokenizer, with per-term query weights.

    The postings come from ``fts5vocab(instance)``, so term boundaries, case
    folding and diacritic handling are FTS5's rather than mine -- a disagreement
    with ``bm25()`` then means the SCORING differs, which is a real finding,
    rather than that my tokenizer differs, which would be a bug.

    ``clamp_idf`` reproduces the one place FTS5 is not textbook BM25: it floors a
    negative IDF at +1e-6 rather than letting a term in more than half the corpus
    push scores DOWN. Default True because the question here is PRF, not the
    clamp, and every baseline number in this study was produced with it.
    """

    def __init__(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        docs: Sequence[str],
        k1: float = K1_SHIPPED,
        b: float = B_SHIPPED,
        clamp_idf: bool = True,
        tokenize: Optional[str] = None,
    ) -> None:
        self.ids = list(ids)
        self.docs = list(docs)
        self.k1 = k1
        self.b = b
        self.clamp_idf = clamp_idf

        conn = sqlite3.connect(":memory:")
        clause = "" if tokenize is None else f", tokenize='{tokenize}'"
        conn.execute(f"CREATE VIRTUAL TABLE u USING fts5(body{clause})")
        conn.executemany("INSERT INTO u(rowid, body) VALUES (?, ?)", enumerate(texts))
        conn.execute("CREATE VIRTUAL TABLE v USING fts5vocab(u, instance)")
        conn.commit()

        # Aggregate in SQLite rather than Python: the instance table has one row
        # per TOKEN, and materialising ~10^7 of those as Python objects is how
        # this box runs out of memory. GROUP BY collapses them to one row per
        # (term, unit) before they ever cross the boundary.
        self.n_docs = len(self.ids)
        term_ids: Dict[str, int] = {}
        post_doc: List[array] = []
        post_tf: List[array] = []
        dl = np.zeros(self.n_docs, dtype=np.int64)
        for term, doc, tf in conn.execute("SELECT term, doc, COUNT(*) FROM v GROUP BY term, doc"):
            ti = term_ids.get(term)
            if ti is None:
                ti = len(term_ids)
                term_ids[term] = ti
                post_doc.append(array("i"))
                post_tf.append(array("i"))
            post_doc[ti].append(doc)
            post_tf[ti].append(tf)
            dl[doc] += tf
        conn.close()

        self.term_ids = term_ids
        self.post_doc = [np.frombuffer(a, dtype=np.int32) for a in post_doc]
        self.post_tf = [np.frombuffer(a, dtype=np.int32).astype(np.float64) for a in post_tf]
        self.dl = dl.astype(np.float64)
        self.avgdl = float(self.dl.mean()) if self.n_docs else 1.0
        # 1 - b + b*D/avgdl, precomputed once: it is per-unit and k1-free.
        self.norm = 1.0 - self.b + self.b * self.dl / max(self.avgdl, 1e-9)
        self._idf = np.array(
            [
                math.log((self.n_docs - len(self.post_doc[i]) + 0.5) / (len(self.post_doc[i]) + 0.5))
                for i in range(len(term_ids))
            ],
            dtype=np.float64,
        )
        if self.clamp_idf:
            self._idf = np.maximum(self._idf, 1e-6)

    def idf(self, term: str) -> float:
        ti = self.term_ids.get(term)
        return 0.0 if ti is None else float(self._idf[ti])

    def doc_terms(self, unit_row: int) -> Dict[str, int]:
        """{term: tf} for one unit. Used to harvest expansion candidates."""
        out: Dict[str, int] = {}
        for term, ti in self.term_ids.items():
            hit = np.searchsorted(self.post_doc[ti], unit_row)
            if hit < len(self.post_doc[ti]) and self.post_doc[ti][hit] == unit_row:
                out[term] = int(self.post_tf[ti][hit])
        return out

    def score(
        self,
        weights: Dict[str, float],
        limit: int,
        scope_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Weighted-sum BM25. ``weights`` is {term: query weight}, 1.0 == plain BM25.

        RETURNS FTS5'S SIGN CONVENTION: negative, most-negative-is-best, so this
        is a drop-in for ``FTS.search``. Ordering is still best-first.

        THIS COST A DEBUGGING CYCLE AND IT IS WORTH THE COMMENT. ``fuse`` takes
        BM25 raw and negates it (`_minmax([-r for r in ...])`) because that is
        what FTS5 hands it. Returning a natural positive score here therefore
        INVERTED the entire keyword ranking downstream -- silently, because every
        arm was inverted equally and the table still looked like a table. The
        oracle arm is what exposed it: a feedback set of nothing but relevant
        units produced the WORST nDCG in every cell, since the better the
        retrieval, the worse the flipped result.

        Note that the calibration control passed at 100% throughout. It compared
        each scorer's own best-first ordering and so could not see a sign
        convention that only exists at the boundary. A component checked against
        a reference can still be wrong about how it is consumed.
        """
        acc = np.zeros(self.n_docs, dtype=np.float64)
        touched = False
        for term, w in weights.items():
            ti = self.term_ids.get(term)
            if ti is None or w == 0.0:
                continue
            f = self.post_tf[ti]
            d = self.post_doc[ti]
            acc[d] += w * self._idf[ti] * f * (self.k1 + 1) / (f + self.k1 * self.norm[d])
            touched = True
        if not touched:
            return {}
        if scope_mask is not None:
            acc = np.where(scope_mask, acc, 0.0)
        nz = np.flatnonzero(acc)
        if not len(nz):
            return {}
        take = nz[np.argsort(-acc[nz])[:limit]]
        return {self.ids[i]: -float(acc[i]) for i in take}


def build_doc_term_matrix(index: LexIndex) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Invert the postings once: per-unit (term_id, tf) arrays.

    ``doc_terms`` above is O(vocab) per call, which is fine for a handful of
    probes and hopeless for FB_K feedback units on every query. Feedback harvest
    is the inner loop of this whole file, so it gets the transpose.
    """
    rows: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for ti in range(len(index.term_ids)):
        for d, f in zip(index.post_doc[ti], index.post_tf[ti], strict=True):
            rows[int(d)].append((ti, float(f)))
    terms: List[np.ndarray] = []
    tfs: List[np.ndarray] = []
    for r in range(index.n_docs):
        pairs = rows.get(r, [])
        terms.append(np.array([p[0] for p in pairs], dtype=np.int32))
        tfs.append(np.array([p[1] for p in pairs], dtype=np.float64))
    return terms, tfs


def tokenize_queries(qtexts: Sequence[str], tokenize: Optional[str] = None) -> List[List[str]]:
    """Query terms via FTS5's OWN tokenizer, so query and corpus agree.

    Tokenises the RAW query text, not the sanitised MATCH string: sanitisation
    inserts `OR` and quotes, which FTS5 would happily tokenise as the term "or".
    The cost is that a quoted PHRASE becomes a bag of words here -- which is
    exactly what the calibration control counts, rather than assuming it is rare.
    """
    conn = sqlite3.connect(":memory:")
    clause = "" if tokenize is None else f", tokenize='{tokenize}'"
    conn.execute(f"CREATE VIRTUAL TABLE q USING fts5(body{clause})")
    conn.executemany("INSERT INTO q(rowid, body) VALUES (?, ?)", enumerate(qtexts))
    conn.execute("CREATE VIRTUAL TABLE qv USING fts5vocab(q, instance)")
    conn.commit()
    out: List[List[str]] = [[] for _ in qtexts]
    for term, doc, _col, off in conn.execute("SELECT term, doc, col, offset FROM qv ORDER BY doc, offset"):
        out[doc].append(term)
        del off
    conn.close()
    return out


def query_model(qterms: Sequence[str]) -> Dict[str, float]:
    """P(t|q) = tf(t,q) / |q| -- the original query model, COUNTS not a term set.

    Deduplicating here was a bug with two heads. RM3's own estimator is defined
    over term frequencies in the query, and separately FTS5 emits one phrase per
    OCCURRENCE, so `bm25()` already counts a repeated term twice: a query
    containing "the common belief" twice scored 24.84 in FTS5 against 19.77 from
    the deduplicated form. Counting fixes the estimator and the calibration in
    the same line.
    """
    counts: Dict[str, float] = defaultdict(float)
    for t in qterms:
        counts[t] += 1.0
    n = sum(counts.values()) or 1.0
    return {t: c / n for t, c in counts.items()}


def bow_sanitizer(qtokens: Dict[str, List[str]]):
    """Sanitiser emitting FTS5's own tokens as single-token quoted OR clauses.

    THE CONTROL THIS EXISTS FOR. `sanitize_fts_query` quotes each whitespace
    token, so a hyphenated word like "state-of-the-art" becomes a QUOTED PHRASE
    that FTS5 scores with an adjacency requirement, while this scorer -- which has
    no positions -- scores its four tokens independently. That is a genuine
    difference in the query model, not an arithmetic error, and it is not
    something a positionless index can be talked out of.

    So calibration is reported twice: against shipped, which measures the phrase
    gap, and against this, which quotes every token SEPARATELY and therefore asks
    the strictly answerable question -- given the same bag-of-words query, does my
    BM25 equal FTS5's? Only the second one gates.
    """

    def _san(query: str):
        toks = qtokens.get(query, [])
        return " OR ".join(f'"{t}"' for t in toks)

    return _san


def rm3_weights(
    index: LexIndex,
    qterms: Sequence[str],
    feedback_rows: Sequence[int],
    fb_scores: Sequence[float],
    doc_terms: List[np.ndarray],
    doc_tfs: List[np.ndarray],
    fb_terms: int = FB_TERMS,
    alpha: float = ALPHA,
    discount: bool = True,
) -> Dict[str, float]:
    """Classical RM3, with the feedback SET left as an argument.

    That argument is the whole experiment: RM3 fixes it to the keyword leg's own
    top-k, and §1.4's proposal is that any other retriever's list may be
    substituted. Everything else here is the textbook estimator.

        P(t|R) = sum_d P(t|d) * P(d|q)        P(t|d) = tf(t,d) / |d|

    then blend with the original query model at ``alpha``.

    THREE CHOICES WORTH NAMING.

    Feedback weights P(d|q) are min-max normalised within the feedback set before
    being normalised to sum 1, because the arms supply scores on incomparable
    scales -- raw BM25 for rm3, cosine for the vector arm -- and an unnormalised
    mixture would make the comparison a comparison of score ranges.

    Candidate terms with idf <= 0 are dropped. FTS5 clamps those to +1e-6 anyway,
    so they cannot help, and they are the stopwords drift is made of.

    ``discount`` scores candidates by P(t|R)*idf(t) rather than by P(t|R) alone.
    THIS IS NOT COSMETIC. P(t|R) is raw frequency, so the textbook estimator
    selects the commonest words in the feedback set, and the ORACLE control
    caught it doing exactly that: given a feedback set of nothing but relevant
    units it picked `features` (df 84/233, idf 0.57) at a larger weight than any
    query term, plus `use` and `all` (idf 0.01), and nDCG fell in all four cells.
    Published RM3 pairs this estimator with a stopword list; we have none, and
    the idf clamp only removes terms above 50% df. The discount is that missing
    filter, expressed as the quantity we already have. ``discount=False`` keeps
    the textbook form so the choice stays a measured one -- see --rm3-select.
    """
    if not feedback_rows:
        return query_model(qterms)

    s = np.asarray(fb_scores, dtype=np.float64)
    lo, hi = float(s.min()), float(s.max())
    w = np.ones_like(s) if hi - lo < 1e-12 else (s - lo) / (hi - lo)
    if w.sum() <= 0:
        w = np.ones_like(s)
    w = w / w.sum()

    rel: Dict[int, float] = defaultdict(float)
    for wd, row in zip(w, feedback_rows, strict=True):
        length = max(float(index.dl[row]), 1.0)
        for ti, f in zip(doc_terms[row], doc_tfs[row], strict=True):
            rel[int(ti)] += wd * (f / length)

    scored = [(ti, m * float(index._idf[ti]) if discount else m) for ti, m in rel.items() if index._idf[ti] > 1e-6]
    scored.sort(key=lambda kv: -kv[1])
    picked = scored[:fb_terms]
    if not picked:
        return query_model(qterms)

    inv = {ti: t for t, ti in index.term_ids.items()}
    total = sum(m for _, m in picked)
    exp_w = {inv[ti]: (m / total) for ti, m in picked}

    out = {t: alpha * p for t, p in query_model(qterms).items()}
    for t, m in exp_w.items():
        out[t] = out.get(t, 0.0) + (1.0 - alpha) * m
    return out


def feedback_precision(rows: Sequence[int], rel_rows: set) -> float:
    """Share of the feedback set that is genuinely relevant.

    CONTROL 2's numerator. The arms differ in two ways at once -- where the
    feedback came from, and how good it was -- and this is the only thing that
    tells those apart afterwards.
    """
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r in rel_rows) / len(rows)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, without pulling in scipy (not a dependency here)."""
    if len(a) < 3:
        return float("nan")

    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ties, or the correlation is distorted wherever values repeat
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    """rho(x,y | z) -- the rank correlation of x and y with z partialled out.

    THE CONTROL THE VOCABULARY-GAP TEST CANNOT DO WITHOUT. Coverage and baseline
    nDCG are strongly correlated: a query whose terms all appear in the gold unit
    is one BM25 already answers, so it starts high and has little headroom. On NQ
    the lowest-coverage quartile starts at 0.176 and the highest at 0.732. A pure
    CEILING effect would therefore reproduce the entire quartile pattern with no
    vocabulary mechanism at all, and reporting the raw correlation would have
    presented that artifact as a discovery.

    Partialling baseline out asks the question that survives: among queries with
    the SAME headroom, does lower coverage still gain more from cross-modal
    feedback? Returns (raw rho, rho controlling for baseline, rho(coverage,
    baseline)) so the strength of the confound is visible rather than assumed.
    """
    r_xy, r_xz, r_yz = spearman(x, y), spearman(x, z), spearman(y, z)
    denom = math.sqrt(max(0.0, (1 - r_xz**2) * (1 - r_yz**2)))
    partial = (r_xy - r_xz * r_yz) / denom if denom > 1e-12 else float("nan")
    return r_xy, partial, r_xz


def query_gold_coverage(
    index: LexIndex,
    qterms: Sequence[Sequence[str]],
    qids: Sequence[str],
    rel_rows_for: Dict[str, set],
    doc_terms: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-query share of the query's own terms that appear in its GOLD units.

    THE QUANTITY THE FOURTH MECHANISM RESTS ON. NQ reversed all three other
    corpora, and the proposed reason is that NQ alone has a large
    query-document VOCABULARY GAP -- its queries are real searches by people who
    had not read the documents, while qasper/MAUD/MLDR queries were written from
    or against theirs. If that is right, cross-modal feedback wins precisely
    where the query lacks the document's words, because the vector arm supplies
    terms BM25 could never reach.

    Measured against the GOLD units, never the retrieved ones. Retrieved units
    were found BY these query terms, so overlap with them is guaranteed high and
    the whole test would be circular.

    Returns (raw coverage, idf-weighted coverage). The idf-weighted form is the
    one to trust: matching "the" is not evidence that the query speaks the
    document's language, and raw coverage counts it the same as a rare term.
    """
    raw = np.full(len(qids), np.nan)
    weighted = np.full(len(qids), np.nan)
    for qi, q in enumerate(qids):
        rows = rel_rows_for[str(q)]
        if not rows:
            continue
        terms = list(dict.fromkeys(qterms[qi]))
        tids = [index.term_ids[t] for t in terms if t in index.term_ids]
        if not tids:
            continue
        gold_tids: set = set()
        for r in rows:
            gold_tids.update(int(t) for t in doc_terms[r])
        hits = [t for t in tids if t in gold_tids]
        raw[qi] = len(hits) / len(tids)
        wt = sum(float(index._idf[t]) for t in tids)
        weighted[qi] = (sum(float(index._idf[t]) for t in hits) / wt) if wt > 0 else np.nan
    return raw, weighted


PRECISION_GRID: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def precision_sweep(
    args,
    index: LexIndex,
    doc_terms: List[np.ndarray],
    doc_tfs: List[np.ndarray],
    qterms: List[List[str]],
    qids: Sequence[str],
    qtexts: Sequence[str],
    vec_cap: List[Dict[str, float]],
    hyb_cap: List[Dict[str, float]],
    kw_cap: List[Dict[str, float]],
    masks: List[Optional[np.ndarray]],
    rel_rows_for: Dict[str, set],
    owner_sec,
    owner_doc,
    bench,
    row_of: Dict[str, int],
) -> int:
    """Measure nDCG as a function of FEEDBACK PRECISION, holding everything else fixed.

    WHY THIS EXISTS. The arm tables give exactly two precision points per corpus
    -- whatever the retrievers happened to deliver (qasper 0.043, MAUD 0.408) and
    the oracle's 1.000 -- and I used them to assert that PRF's response to
    precision is steeply nonlinear, i.e. that only near-perfect feedback pays.
    Two points do not determine a curve, and the two corpora sit at very
    different baselines (0.18 vs 0.81), so a ceiling effect alone could produce
    the same appearance. Three hypotheses have already died in this study by
    being asserted from the shape of a table instead of measured; this one gets
    measured before it is believed.

    THE CONSTRUCTION. For a target precision p, each query's feedback set is
    round(p*fb_k) genuinely relevant units plus enough non-relevant ones to fill
    fb_k. Everything else -- fb_k, term count, alpha, the scorer -- is held
    fixed, so precision is the only thing moving.

    THE DESIGN POINT THAT MATTERS: where the filler comes from. Random corpus
    units are off-topic in a way that RETRIEVED-but-wrong units are not, and real
    PRF only ever sees the latter. So the filler is drawn from the top of the
    query's own HYBRID list with the relevant units removed -- the actual
    distractors PRF has to survive. ``--sweep-filler random`` is the contrast.

    MEASURED, AND IT INVERTED WHAT THIS PARAGRAPH ORIGINALLY CLAIMED. I predicted
    random filler would FLATTER the curve, on the reasoning that
    retrieved-but-wrong units are topically close and therefore more confusable.
    The opposite holds. At precision 0.0 on NQ, random filler costs -0.2144* on
    section/keyword and -0.3704* on doc/keyword, against -0.0759 and -0.0035 for
    retrieved filler.

    **Drift from near-misses is far cheaper than drift from noise.** Random units
    inject vocabulary unrelated to anything and pull the expanded query in
    arbitrary directions; retrieved-but-wrong units stay topically adjacent, so
    their terms remain partly on target. The two curves converge at precision
    1.0, where there is no filler left to differ over. This is the opposite of
    the usual intuition that query drift is worst when the feedback is
    PLAUSIBLY wrong.

    Consequence for the break-even reading: realistic (retrieved) distractors
    give the LOWER break-even -- 0.1-0.2 vs 0.3 on doc/keyword -- so quoting the
    random-filler figure would understate what PRF can do in deployment.

    RESTRICTED TO QUERIES THAT CAN REACH EVERY GRID POINT (>= fb_k relevant units
    and >= fb_k distractors). Otherwise a query capped at p=0.3 would silently
    drop out of the high cells and the curve would compare different query sets
    at different points -- which is the same confound as an unbracketed argmax,
    wearing a different hat. The qualifying count is printed; if it is small, the
    curve is not readable and says so.
    """
    import random

    rng = random.Random(0)
    fb_k = args.fb_k
    n_units = index.n_docs

    eligible: List[int] = []
    gold_pool: Dict[int, List[int]] = {}
    fill_pool: Dict[int, List[int]] = {}
    for qi in range(len(qtexts)):
        rel = rel_rows_for[str(qids[qi])]
        if len(rel) < fb_k:
            continue
        retrieved = [row_of[u] for u in hyb_cap[qi] if u in row_of]
        fill = [r for r in retrieved if r not in rel]
        if args.sweep_filler == "random":
            pool = [r for r in rng.sample(range(n_units), min(n_units, 8 * fb_k)) if r not in rel]
            fill = pool
        if len(fill) < fb_k:
            continue
        eligible.append(qi)
        gold_pool[qi] = sorted(rel)
        fill_pool[qi] = fill

    print(
        f"\n  PRECISION SWEEP [{args.dataset}/{args.level}] filler={args.sweep_filler}: "
        f"{len(eligible)}/{len(qtexts)} queries have >= {fb_k} relevant units AND "
        f">= {fb_k} distractors, so every grid point is measured on the SAME queries"
    )
    if len(eligible) < 30:
        print("  too few eligible queries to read a curve; not reporting")
        return 1

    sub_qids = [qids[i] for i in eligible]
    kw_by_p: Dict[float, List[Dict[str, float]]] = {}
    for p in PRECISION_GRID:
        n_gold = int(round(p * fb_k))
        caps: List[Dict[str, float]] = []
        for qi in eligible:
            rows = gold_pool[qi][:n_gold] + fill_pool[qi][: fb_k - n_gold]
            w = rm3_weights(
                index,
                list(qterms[qi]),
                rows,
                [1.0] * len(rows),
                doc_terms,
                doc_tfs,
                args.fb_terms,
                args.alpha,
                discount=args.rm3_select == "idf",
            )
            caps.append(index.score(w, SEARCH_K, masks[qi]))
        kw_by_p[p] = caps

    sub_vec = [vec_cap[i] for i in eligible]
    sub_kw = [kw_cap[i] for i in eligible]
    targets = [("section", bench.section_qrels), ("doc", bench.doc_qrels)]
    if args.dataset == "maud":
        targets = [("section", bench.section_qrels)]

    out: Dict[str, Dict[str, float]] = {}
    for tname, qrels in targets:
        if not qrels:
            continue
        if args.level == "chunks":
            owner = owner_sec if tname == "section" else owner_doc
        else:
            owner = None if tname == "section" else owner_doc
            if args.coarse == "documents" and tname == "section":
                continue
        for arm_name, vw in (("keyword", 0.0), ("hybrid", args.vector_weight)):
            base = score(
                to_ranked(blend_arm(list(zip(sub_vec, sub_kw, strict=True)), owner, vw, use_keyword=True)),
                sub_qids,
                qrels,
            )
            pq = {}
            for p in PRECISION_GRID:
                ranked = to_ranked(blend_arm(list(zip(sub_vec, kw_by_p[p], strict=True)), owner, vw, use_keyword=True))
                pq[p] = score(ranked, sub_qids, qrels)
            tag = f"{args.dataset}/{args.level}/{tname}/{arm_name}"
            print(f"\n=== {tag} · PRECISION SWEEP · n={len(eligible)} · no-PRF={base.mean():.4f} ===")
            print(f"  {'fb prec':>8} {'gold/k':>7} {'nDCG@10':>9} {'vs no-PRF':>10} {'95% CI':>20}")
            for p in PRECISION_GRID:
                st = paired(pq[p], base)
                ci = f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
                star = "*" if (st["ci_lo"] > 0 or st["ci_hi"] < 0) else " "
                print(
                    f"  {p:>8.2f} {int(round(p * fb_k)):>4}/{fb_k:<2} {pq[p].mean():>9.4f} "
                    f"{st['delta']:>+10.4f} {ci:>20}{star}"
                )
                out[f"{tag}|p={p:g}"] = {
                    "ndcg@10": float(pq[p].mean()),
                    "delta": st["delta"],
                    "ci_lo": st["ci_lo"],
                    "ci_hi": st["ci_hi"],
                }
            # WHERE DOES IT CROSS ZERO? That break-even is the whole practical
            # question: it says how good a feedback set must be before expansion
            # is worth running at all, and therefore whether a reranked seed
            # (precision ~0.6) could ever pay.
            crossing = next((p for p in PRECISION_GRID if pq[p].mean() > base.mean()), None)
            print(
                f"  break-even precision: {crossing if crossing is not None else 'NEVER on this grid'}"
                "  <- a reranked feedback set plausibly reaches ~0.6; compare against this"
            )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "level": args.level,
                    "mode": "precision_sweep",
                    "filler": args.sweep_filler,
                    "fb_k": fb_k,
                    "eligible": len(eligible),
                    "grid": list(PRECISION_GRID),
                    "results": out,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.out}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("qasper", "maud", "mldr", "nq"), default="qasper")
    p.add_argument("--model-key", default="egemma")
    p.add_argument("--coarse", choices=("sections", "documents"), default="sections")
    p.add_argument("--level", choices=("chunks", "coarse"), default="chunks")
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--vector-weight", type=float, default=VECTOR_WEIGHT)
    p.add_argument("--fb-k", type=int, default=FB_K)
    p.add_argument("--fb-terms", type=int, default=FB_TERMS)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument(
        "--vocab-gap",
        action="store_true",
        help="stratify the cross-modal advantage by query-term coverage of the gold units",
    )
    p.add_argument(
        "--precision-sweep",
        action="store_true",
        help="measure nDCG vs feedback PRECISION on a fixed query set, instead of the arm table",
    )
    p.add_argument(
        "--sweep-filler",
        choices=("retrieved", "random"),
        default="retrieved",
        help="non-relevant feedback drawn from the query's own hybrid list (realistic) or at random",
    )
    p.add_argument(
        "--rm3-select",
        choices=("idf", "freq"),
        default="idf",
        help="expansion-term selection: P(t|R)*idf (default) or textbook P(t|R) alone",
    )
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

    spec = MODEL_POOL[args.model_key]
    from dataclasses import replace

    doc_enc = PrefixedEncoder(spec, spec.doc_prefix)
    qry_enc = PrefixedEncoder(spec, spec.query_prefix)

    if args.level == "chunks":
        texts = list(units.chunk_texts)
        uids = [f"c{i}" for i in range(len(texts))]
        udocs = list(units.chunk_doc)
        enc = doc_enc
        owner_sec = {u: [v] for u, v in zip(uids, units.chunk_section, strict=True)}
        owner_doc = {u: [v] for u, v in zip(uids, udocs, strict=True)}
        fb_owner = owner_sec
    else:
        if args.coarse == "sections":
            texts = list(units.section_texts)
            uids = list(units.section_ids)
            udocs = list(units.section_doc)
        else:
            texts = [bench.corpus[d] for d in bench.corpus]
            uids = list(bench.corpus)
            udocs = list(bench.corpus)
        enc = doc_enc
        if args.coarse == "sections" and spec.section_window_chars is not None:
            enc = PrefixedEncoder(
                replace(spec, window_chars=spec.section_window_chars, window_tokens=None), spec.doc_prefix
            )
        owner_sec = None  # type: ignore[assignment]
        owner_doc = {u: [v] for u, v in zip(uids, udocs, strict=True)}
        # At a coarse level the unit IS the judged thing, so it owns itself.
        fb_owner = {u: [u] for u in uids} if args.coarse == "sections" else owner_doc

    if not args.allow_embed:
        miss = enc.count_misses(texts)[1] + qry_enc.count_misses(units.query_texts)[1]
        if miss:
            raise SystemExit(
                f"{miss} vectors are not cached for {spec.model}. This harness is zero-embedding "
                "by default; pass --allow-embed to encode them."
            )

    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1.0, n)

    uv = unit(enc.encode(texts, normalize=False))
    qv = unit(qry_enc.encode(units.query_texts, normalize=False))
    qids, qtexts = list(units.query_ids), list(units.query_texts)

    q_scope: Optional[List[Optional[str]]] = None
    if args.dataset == "maud":
        q_scope = [str(q).split("||", 1)[0] for q in qids]

    logger.info("%d units (%s), %d queries", len(uids), args.level, len(qids))

    index = LexIndex(uids, texts, udocs)
    logger.info("lexical index: %d terms, avgdl %.0f", len(index.term_ids), index.avgdl)
    doc_terms, doc_tfs = build_doc_term_matrix(index)
    qterms = tokenize_queries(qtexts)
    row_of = {u: i for i, u in enumerate(uids)}
    doc_of_row = np.array(udocs, dtype=object)

    # ---- CONTROL 1: does the hand-rolled scorer reproduce FTS5's bm25()? ----
    # Reported against TWO references. Against shipped it measures the phrase gap
    # -- src quotes each whitespace token, so a hyphenated word becomes an
    # adjacency-constrained phrase this positionless scorer cannot express.
    # Against the same bag of words it asks the question that IS answerable, and
    # that is the one allowed to gate.
    # TIE-AWARE, and it has to be. Comparing ordered id lists conflates "ranked
    # wrongly" with "tied and ordered differently", and on NQ that false-rejected
    # the whole corpus: 179/2000 queries "disagreed" and ALL 179 were pure
    # tie-breaks with identical score multisets -- zero genuine differences. NQ
    # has 44,619 units and short real-user queries, so exact ties are common.
    #
    # This is the same flaw as the sign bug in ``LexIndex.score``, in the other
    # direction: the check looked at ORDER when the thing that matters is SCORE.
    # It gave a false pass there and a false fail here. Both rates are printed --
    # strict for visibility, tie-aware to gate.
    fts = FTS(uids, texts, udocs)
    fts_bow = FTS(uids, texts, udocs, sanitize=bow_sanitizer(dict(zip(qtexts, qterms, strict=True))))
    agree_ship = agree_bow = agree_bow_strict = multi_tok = tie_only = 0
    for qi, qtext in enumerate(qtexts):
        mask = None if q_scope is None else (doc_of_row == q_scope[qi])
        scope = q_scope[qi] if q_scope else None
        qm = query_model(qterms[qi])
        mine = list(index.score(qm, 10, mask))
        theirs_bow = list(fts_bow.search(qtext, 10, scope))
        if mine == list(fts.search(qtext, 10, scope)):
            agree_ship += 1
        if mine == theirs_bow:
            agree_bow_strict += 1
            agree_bow += 1
        else:
            # Same scores, different tie-break => both orderings are equally
            # correct BM25. Score under MY index so the comparison is of the
            # ranking function, not of two different score scales.
            full = index.score(qm, 200, mask)
            key = [round(full.get(u, 0.0), 9) for u in mine]
            if key == [round(full.get(u, 0.0), 9) for u in theirs_bow]:
                agree_bow += 1
                tie_only += 1
        # Any whitespace token FTS5 splits further becomes a phrase in shipped.
        if any(len(tokenize_queries([w])[0]) > 1 for w in qtext.split()):
            multi_tok += 1
    n_q = max(len(qtexts), 1)
    rate = agree_bow / n_q
    rate_ship = agree_ship / n_q
    print(
        f"\n  CALIBRATION [{args.dataset}/{args.level}]:\n"
        f"    vs FTS5 on the SAME bag of words : {agree_bow}/{n_q} ({100 * rate:.1f}%)  <- gates\n"
        f"       (exact id match {agree_bow_strict}/{n_q}; +{tie_only} identical-score tie-breaks, "
        "which are equally-correct BM25 orderings, not errors)\n"
        f"    vs shipped sanitisation          : {agree_ship}/{n_q} ({100 * rate_ship:.1f}%)  "
        f"-- {multi_tok} queries contain a token FTS5 splits (hyphens, apostrophes), which shipped\n"
        f"       scores as an adjacent PHRASE and a positionless scorer cannot"
    )
    if rate < 0.98:
        print(
            "  refusing to report PRF arms: the no-PRF baseline is not BM25 over the query it was "
            "given, so every delta below would be confounded with the scorer itself"
        )
        return 1

    # ---- feedback sources ----
    # The keyword feedback set is THIS scorer's own top-k, not FTS5's. RM3 is
    # defined as feeding back the ranking you are about to expand, and using the
    # shipped leg instead would hand the rm3 arm a feedback set drawn from a
    # different ranker -- quietly making the control a fourth cross-arm.
    vec_cap = [v for v, _ in capture_arm(uids, uv, None, qv, qtexts, udocs, q_scope)]
    masks = [None if q_scope is None else (doc_of_row == q_scope[i]) for i in range(len(qtexts))]
    kw_cap = [index.score(query_model(qterms[i]), SEARCH_K, masks[i]) for i in range(len(qtexts))]
    hyb_cap = [fuse(v, k, args.vector_weight) for v, k in zip(vec_cap, kw_cap, strict=True)]

    # RELEVANCE IS NOT DEFINED AT THE RETRIEVAL UNIT. qrels judge sections or
    # documents; the units being retrieved here are chunks. Comparing a chunk id
    # against a section id would make feedback precision identically 0 and hand
    # the oracle arm an EMPTY feedback set -- i.e. it would silently become a
    # second copy of `none`, and the ceiling control would report nothing while
    # looking like it worked. Map through ownership instead.
    target_qrels = bench.section_qrels if args.level == "chunks" else bench.doc_qrels
    if not target_qrels:
        target_qrels = bench.doc_qrels
        fb_owner = owner_doc
    rows_of_parent: Dict[str, List[int]] = defaultdict(list)
    for r, u in enumerate(uids):
        for parent in fb_owner[u]:
            rows_of_parent[parent].append(r)
    rel_rows_for: Dict[str, set] = {}
    for q in qids:
        parents = [d for d, g in (target_qrels.get(str(q), {}) or {}).items() if g > 0]
        rel_rows_for[str(q)] = {r for p in parents for r in rows_of_parent.get(p, ())}

    def rows_from(cap: Dict[str, float], k: int, sign: float = 1.0) -> Tuple[List[int], List[float]]:
        """Feedback rows + HIGHER-IS-BETTER scores.

        ``sign=-1`` for the keyword leg, which carries FTS5's negative
        convention. rm3_weights min-max normalises these, so feeding it raw BM25
        would hand the WORST feedback document the highest weight -- the same
        sign trap as ``fuse``, one layer up.
        """
        rows, scs = [], []
        for uid, sc in list(cap.items())[:k]:
            r = row_of.get(uid)
            if r is not None:
                rows.append(r)
                scs.append(sign * sc)
        return rows, scs

    if args.precision_sweep:
        return precision_sweep(
            args,
            index,
            doc_terms,
            doc_tfs,
            qterms,
            qids,
            qtexts,
            vec_cap,
            hyb_cap,
            kw_cap,
            masks,
            rel_rows_for,
            owner_sec,
            owner_doc,
            bench,
            row_of,
        )

    # CONTROL 1b, the one that actually catches interface bugs. `none` is this
    # file's own BM25 and `shipped` is FTS5's, pushed through the SAME fusion,
    # roll-up and scoring path. The isolated calibration above compares two
    # orderings and is blind to how they are CONSUMED -- it sat at 100% while a
    # sign-convention mismatch inverted every ranking downstream. These two rows
    # must agree to within the hyphen-phrase queries; if they do not, nothing
    # below is readable.
    ship_kw = [fts.search(t, SEARCH_K, q_scope[i] if q_scope else None) for i, t in enumerate(qtexts)]

    cov_raw, cov_w = (
        query_gold_coverage(index, qterms, qids, rel_rows_for, doc_terms)
        if args.vocab_gap
        else (np.full(len(qids), np.nan), np.full(len(qids), np.nan))
    )
    if args.vocab_gap:
        good = ~np.isnan(cov_w)
        logger.info(
            "query-gold coverage: raw %.3f, idf-weighted %.3f (n=%d)",
            float(np.nanmean(cov_raw)) if good.any() else float("nan"),
            float(np.nanmean(cov_w)) if good.any() else float("nan"),
            int(good.sum()),
        )

    ARMS = ("none", "shipped", "rm3", "xprf-h", "xprf-v", "oracle")
    kw_by_arm: Dict[str, List[Dict[str, float]]] = {a: [] for a in ARMS}
    fbp: Dict[str, List[float]] = {a: [] for a in ARMS}
    hit: Dict[str, List[bool]] = {a: [] for a in ARMS}

    for qi in range(len(qtexts)):
        mask = masks[qi]
        terms_i = list(qterms[qi])
        rel_rows = rel_rows_for[str(qids[qi])]
        for arm in ARMS:
            if arm in ("none", "shipped"):
                kw_by_arm[arm].append(kw_cap[qi] if arm == "none" else ship_kw[qi])
                fbp[arm].append(float("nan"))
                hit[arm].append(False)
                continue
            if arm == "rm3":
                rows, scs = rows_from(kw_cap[qi], args.fb_k, sign=-1.0)
            elif arm == "xprf-h":
                rows, scs = rows_from(hyb_cap[qi], args.fb_k)
            elif arm == "xprf-v":
                rows, scs = rows_from(vec_cap[qi], args.fb_k)
            else:
                # ORACLE: the feedback set IS the relevant units. Ranked by the
                # vector arm's own ordering only so the pick is deterministic;
                # the ceiling does not depend on which relevant units it gets.
                rows = sorted(rel_rows)[: args.fb_k]
                scs = [1.0] * len(rows)
            w = rm3_weights(
                index,
                terms_i,
                rows,
                scs,
                doc_terms,
                doc_tfs,
                args.fb_terms,
                args.alpha,
                discount=args.rm3_select == "idf",
            )
            kw_by_arm[arm].append(index.score(w, SEARCH_K, mask))
            fbp[arm].append(feedback_precision(rows, rel_rows))
            hit[arm].append(any(r in rel_rows for r in rows))

    results: Dict[str, Dict[str, float]] = {}
    targets = [("section", bench.section_qrels), ("doc", bench.doc_qrels)]
    if args.dataset == "maud":
        targets = [("section", bench.section_qrels)]

    for tname, qrels in targets:
        if not qrels:
            continue
        if args.level == "chunks":
            owner = owner_sec if tname == "section" else owner_doc
        else:
            owner = None if tname == "section" else owner_doc
            if args.coarse == "documents" and tname == "section":
                continue
        for arm_name, vw in (("keyword", 0.0), ("hybrid", args.vector_weight)):
            pq: Dict[str, np.ndarray] = {}
            for a in ARMS:
                captured = list(zip(vec_cap, kw_by_arm[a], strict=True))
                ranked = to_ranked(blend_arm(captured, owner, vw, use_keyword=True))
                pq[a] = score(ranked, qids, qrels)
                results[f"{args.dataset}/{args.level}/{tname}/{arm_name}|{a}"] = {
                    "ndcg@10": float(pq[a].mean()),
                    "recall@10": recall(ranked, qids, qrels),
                    "fb_precision": float(np.nanmean(fbp[a])) if a not in ("none", "shipped") else float("nan"),
                    "fb_hit_rate": float(np.mean(hit[a])) if a not in ("none", "shipped") else float("nan"),
                }
            tag = f"{args.dataset}/{args.level}/{tname}/{arm_name}"
            print(f"\n=== {tag} · vw={vw:.2f} · fb_k={args.fb_k} terms={args.fb_terms} a={args.alpha} ===")
            print(f"  {'arm':>8} {'nDCG@10':>9} {'delta':>9} {'95% CI':>20} {'fb prec':>8} {'fb hit':>7}")
            for a in ARMS:
                st = paired(pq[a], pq["none"])
                ci = f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
                star = "*" if (st["ci_lo"] > 0 or st["ci_hi"] < 0) else " "
                fp = float(np.nanmean(fbp[a])) if a not in ("none", "shipped") else float("nan")
                fh = float(np.mean(hit[a])) if a not in ("none", "shipped") else float("nan")
                print(f"  {a:>8} {pq[a].mean():>9.4f} {st['delta']:>+9.4f} {ci:>20}{star} " f"{fp:>8.3f} {fh:>7.3f}")
            gap = float(pq["none"].mean() - pq["shipped"].mean())
            # BE PRECISE ABOUT WHAT A GAP INVALIDATES. Every PRF delta is measured
            # against `none`, which is this file's own scorer, so the arm-vs-arm
            # and arm-vs-none comparisons (Claims A/B/C) stay internally valid
            # whatever this reads. What a gap breaks is TRANSFER: `none` is then
            # not shipped behaviour, so "PRF would do X to our system" does not
            # follow. An earlier version printed "read nothing below", which
            # overstated it and would have thrown away sound comparisons.
            flag = (
                ""
                if abs(gap) < 0.01
                else "   !! exceeds 0.01: arm-vs-arm still valid, but does NOT transfer to shipped"
            )
            print(f"  CONTROL none-vs-shipped BM25 through the same path: {gap:+.4f}{flag}")
            # CONTROL 2: is the gradient decorrelation, or just feedback quality?
            # Restricted to queries where BOTH arms' feedback sets contained a
            # relevant unit, the quality difference is largely removed and what
            # is left is where the terms came from.
            both = np.array([h1 and h2 for h1, h2 in zip(hit["rm3"], hit["xprf-v"], strict=True)])
            if both.sum() >= 20:
                d = paired(pq["xprf-v"][both], pq["rm3"][both])
                print(
                    f"  MATCHED STRATA (n={int(both.sum())}, both feedback sets contain a relevant unit): "
                    f"xprf-v - rm3 = {d['delta']:+.4f} [{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]"
                )
            else:
                print(f"  MATCHED STRATA: only {int(both.sum())} queries qualify -- not reported")

            if args.vocab_gap:
                # THE WITHIN-CORPUS TEST of the vocabulary-gap mechanism. Four
                # corpus-level points cannot establish it -- that is the shape of
                # reasoning that killed "unit length" in section 1.2. If the
                # mechanism is real it must ALSO hold inside a single corpus:
                # low-coverage queries should gain most from cross-modal
                # feedback, high-coverage queries least. Same queries, same
                # scorer, so corpus-level confounds cannot produce it.
                adv = pq["xprf-v"] - pq["rm3"]
                ok = ~np.isnan(cov_w)
                if ok.sum() >= 40:
                    c, a_ = cov_w[ok], adv[ok]
                    b_ = pq["none"][ok]
                    rho, rho_p, rho_cb = partial_spearman(c, a_, b_)
                    qs = np.quantile(c, [0.25, 0.5, 0.75])
                    buckets = np.digitize(c, qs)
                    print(
                        f"  VOCAB GAP (idf-weighted query-term coverage of GOLD, n={int(ok.sum())}): "
                        f"spearman(coverage, xprf-v minus rm3) = {rho:+.3f}"
                    )
                    print(
                        f"    CEILING CONTROL: rho(coverage, no-PRF baseline) = {rho_cb:+.3f}; "
                        f"partial rho controlling for baseline = {rho_p:+.3f}"
                        + ("" if abs(rho_p) >= 0.05 else "   <- mechanism does NOT survive the control")
                    )
                    # BASELINE PER QUARTILE, because without it a CEILING EFFECT
                    # is indistinguishable from the mechanism. High-coverage
                    # queries are the ones BM25 already answers well; if their
                    # no-PRF nDCG sits near 1.0, expansion has nowhere to go but
                    # down and the quartile pattern appears whatever the
                    # vocabulary story. Read the advantage against this column.
                    base_q = pq["none"][ok]
                    print(f"    {'quartile':>10} {'coverage':>9} {'n':>5} {'no-PRF':>8} {'xprf-v - rm3':>13}")
                    for bi, lbl in enumerate(("Q1 lowest", "Q2", "Q3", "Q4 highest")):
                        sel = buckets == bi
                        if not sel.any():
                            continue
                        print(
                            f"    {lbl:>10} {c[sel].mean():>9.3f} {int(sel.sum()):>5} "
                            f"{base_q[sel].mean():>8.4f} {a_[sel].mean():>+13.4f}"
                        )
                    lo, hi = buckets == 0, buckets == 3
                    if lo.sum() >= 15 and hi.sum() >= 15:
                        # Unpaired: different queries, so bootstrap the difference
                        # of means rather than reusing `paired`, which assumes
                        # per-query alignment and would be simply wrong here.
                        rng = np.random.default_rng(0)
                        d = np.array(
                            [
                                rng.choice(a_[lo], lo.sum(), replace=True).mean()
                                - rng.choice(a_[hi], hi.sum(), replace=True).mean()
                                for _ in range(10_000)
                            ]
                        )
                        ci = np.percentile(d, [2.5, 97.5])
                        star = "*" if (ci[0] > 0 or ci[1] < 0) else " "
                        print(
                            f"    Q1 minus Q4 = {a_[lo].mean() - a_[hi].mean():+.4f} "
                            f"[{ci[0]:+.4f},{ci[1]:+.4f}]{star}   "
                            "(mechanism predicts POSITIVE: low coverage gains more)"
                        )
                    # BASELINE-MATCHED STRATA -- the control that does not assume
                    # linearity, unlike the partial correlation above. Within each
                    # baseline quintile, headroom is roughly constant, so a
                    # surviving low-minus-high coverage gap cannot be a ceiling
                    # effect. Sign consistency ACROSS strata is the thing to read;
                    # individual strata are small.
                    bq = np.quantile(b_, [0.2, 0.4, 0.6, 0.8])
                    bstrat = np.digitize(b_, bq)
                    print(f"    {'baseline stratum':>17} {'n':>5} {'base':>6} {'lowcov':>8} {'highcov':>8} {'diff':>8}")
                    diffs = []
                    for si in range(5):
                        sel = bstrat == si
                        if sel.sum() < 20:
                            continue
                        cs, as_ = c[sel], a_[sel]
                        med = np.median(cs)
                        low, high = as_[cs <= med], as_[cs > med]
                        if len(low) < 8 or len(high) < 8:
                            continue
                        d = float(low.mean() - high.mean())
                        diffs.append(d)
                        print(
                            f"    {'B' + str(si + 1):>17} {int(sel.sum()):>5} {b_[sel].mean():>6.3f} "
                            f"{low.mean():>+8.4f} {high.mean():>+8.4f} {d:>+8.4f}"
                        )
                    if diffs:
                        pos = sum(1 for d in diffs if d > 0)
                        mean_d = float(np.mean(diffs))
                        # STATE THE VERDICT THIS DATA SUPPORTS, not the one the
                        # test was built hoping for. The earlier version printed
                        # "consistent sign is what the ceiling explanation cannot
                        # produce" unconditionally -- including on cells where the
                        # sign was 1/3 and the mean NEGATIVE, i.e. where it was
                        # arguing for a conclusion its own numbers refuted.
                        if pos == len(diffs) and mean_d > 0:
                            verdict = "SURVIVES: sign consistent across strata, which a ceiling artifact cannot produce"
                        elif pos > len(diffs) / 2 and mean_d > 0:
                            verdict = "PARTIAL: majority sign and positive mean, but not consistent"
                        else:
                            verdict = "FAILS: sign inconsistent or mean <= 0 -- the raw gradient was ceiling"
                        print(
                            f"    -> low-coverage gains more in {pos}/{len(diffs)} baseline strata "
                            f"(mean {mean_d:+.4f}) -- {verdict}"
                        )
                else:
                    print(f"  VOCAB GAP: only {int(ok.sum())} queries have gold coverage -- skipped")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "model": spec.model,
                    "level": args.level,
                    "fb_k": args.fb_k,
                    "fb_terms": args.fb_terms,
                    "alpha": args.alpha,
                    "calibration_rate": rate,
                    "queries": len(qids),
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] if len(sys.argv) > 1 else None))
