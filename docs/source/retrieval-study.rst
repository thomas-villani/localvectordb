The Retrieval Study
===================

Nearly every retrieval default in LocalVectorDB was chosen by measurement, not
convention. Over an intensive study we measured the retrieval stack more or
less exhaustively: **four real corpora** (Qasper research papers, Natural
Questions, MLDR long documents, MAUD legal contracts — plus BEIR SciFact and
NFCorpus for gating), **three encoders**, and every knob the library ships,
with paired-bootstrap confidence intervals on everything. This page presents
what we found, organised by conclusion. The companion page,
:doc:`retrieval-lab-notebook`, tells the story of *how* these numbers were
established — including the wrong conclusions we caught along the way and the
controls that caught them.

Throughout, the score is **nDCG@10** (0–1, "how good are the top ten
results"). On these corpora a *real* improvement is +0.01; anything under
+0.005 needs a confidence interval before you believe it, and our measured
build-to-build noise floor is 0.011.

.. contents::
   :local:
   :depth: 2

The headline: most knobs have no defensible global default
----------------------------------------------------------

Six retrieval parameters turned out to have **no value that is right across
corpora** — their optima disagree by more than the effects most tuning work
chases:

.. list-table::
   :header-rows: 1
   :widths: 28 24 44

   * - Parameter
     - Measured optima
     - What it depends on
   * - ``chunk_size``
     - 128–500+
     - Encoder context window; no interior optimum at all on pure vector
       search (smaller is monotonically better)
   * - ``search_level``
     - chunks / sections / fused
     - Corpus structure — and it resisted every rule we proposed
   * - ``section_weight``
     - 0.35 / 0.65 / 0.80
     - Corpus (MAUD / Qasper / NQ)
   * - ``section_vector_strategy``
     - rawspan / centroid
     - Span length vs. encoder window (rawspan loses 0.25–0.36 on >8k-token
       spans)
   * - ``vector_weight``
     - 0.5 / 0.9
     - Whether BM25 can discriminate on the corpus (NQ's optimum is 0.9)
   * - ``frequency_bias``
     - varies
     - Score scale and fanout

The product consequence is not better constants — it is a **diagnostic**.
Everything a user needs to know about which regime their corpus is in is
derivable from a built index in seconds: measured encoder coverage, section
lengths against the encoder window, fanout, keyword-index health. That is what
``db.diagnose()`` / ``lvdb db <name> doctor`` computes (see :doc:`cli`), and
why it exists instead of another round of default-tweaking.

Exactly one knob generalised across every corpus we measured — the aggregation
rule below.

The hierarchy of leverage
-------------------------

Not all tuning is equal. Measured effect sizes, largest first:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Lever
     - Worth (nDCG@10)
     - Note
   * - **Cross-encoder reranker — model choice**
     - 0.067 spread
     - Larger than every first-stage effect combined; two of nine models
       tested are indistinguishable from not reranking
   * - **The keyword (BM25) leg**
     - +0.084 to +0.131
     - Per retrieval level, across six corpus/encoder pairs
   * - Aggregation method
     - up to +0.023
     - The ``auto`` default captures it
   * - Candidate pool width
     - ~+0.008
     - Regime-dependent; the default stays 40
   * - BM25 parameters (``k1``, ``b``, BM25+)
     - ~0 net
     - Stock FTS5 is already near-optimal — see below

Two practical readings. First, **if BM25 is missing anywhere in your pipeline,
nothing else you tune matters as much** — which is why every retrieval level
in LocalVectorDB (chunks, sections, documents, fused) now carries a keyword
leg. Second, reranking is not a technique with a value; it is a *model
choice* with a spread. Picking the wrong model silently costs more than all
first-stage tuning combined.

Choosing a reranker
-------------------

Nine cross-encoders through one code path produced the study's largest single
effect — and its least intuitive rules:

- **The spread between models (0.067) is the finding.** The best model
  measured +0.0648 over first-stage retrieval; MS MARCO-trained cross-encoders
  measured indistinguishable from not reranking at all — including on MS
  MARCO's own home domain, and at matched capacity. The failure is the
  training recipe, not model size.
- **Price carries no information.** Cost per nDCG point spanned ~96× across
  vendors (Spearman ρ = +0.157, n.s.). A free model won.
- **Rankings do not transfer between corpora, but the floor does.** The model
  ranked 5th of 6 on one corpus ranked 1st on another. A leaderboard can tell
  you what to *rule out*; it cannot tell you what to *pick*.
- **Never truncate reranker input below 512 tokens.** A good reranker loses
  two thirds of its gain at 256. ``max_length`` is not a free latency knob.

LocalVectorDB's local default is ``BAAI/bge-reranker-base`` for exactly these
reasons. Full guidance in :doc:`querybuilder`.

Chunk size: a plateau with a cliff — or a slope, depending on your legs
-----------------------------------------------------------------------

On **hybrid** search (the default), chunk size is forgiving: an 8× range of
sizes inside the encoder's context varies total nDCG by only ~0.06. On pure
**vector** search there is no plateau and no interior optimum — smaller chunks
are monotonically better across a 14× range, with a total span of ~0.30:

.. list-table::
   :header-rows: 1
   :widths: 22 13 13 13 13 13 13

   * - MiniLM (256-token ctx)
     - 64
     - 128
     - 219
     - 256
     - 500
     - 1000
   * - vector
     - 0.8013
     - **0.8401**
     - 0.8318
     - 0.8188
     - 0.6727
     - 0.5453
   * - hybrid
     - 0.7445
     - 0.7732
     - 0.7839
     - 0.7809
     - 0.7570
     - 0.7274

Read the two rows together: **hybrid search is a safety net for chunk-size
misconfiguration**. A vector-only deployment is 4–5× more sensitive to
``chunk_size`` than a hybrid one.

Two mechanisms hide in that curve, and separating them mattered:

- **Granularity, not truncation, is the dominant term.** Between sizes 219 and
  1000, encoder coverage is 100% at every rung and vector nDCG still falls
  0.162. Nothing is being cut off — larger chunks simply produce more diluted
  vectors.
- **Truncation is real but second-order, and it switches on only past the
  context window.** In a controlled pair of builds, 46% of chunks truncated
  moved nDCG by 0.0000; damage appears only when the *vectors* measurably
  rotate. (The doctor's coverage warning is calibrated from exactly this
  measurement: it stays silent while truncation is free and fires at the
  point it starts costing.)

Aggregation: the one rule that generalised
------------------------------------------

When chunks roll up to a coarser unit, how should their scores combine? Across
twenty measured cells (three corpora, two legs, two pool widths), sum-like
aggregators **never lost on a document target and never won on a section
target**. The sign never tracked the corpus; it tracked the unit being ranked:

  A **document** is a bag of topics — a second strongly-matching passage is
  genuinely additional evidence. A **section** is one argument — a second
  matching chunk inside it is mostly the first chunk again. Summing prices
  redundancy as evidence, which is right for one unit and wrong for the
  other.

This shipped as ``document_scoring_method="auto"``: ``best`` for pure vector
search, ``frequency_boost`` for hybrid and keyword (worth +0.0226 / +0.0150 /
+0.0084 on the three corpora measured), while section roll-up keeps a plain
max and exposes no knob. See :doc:`document-scoring`.

One mechanism, five results: the compressed vector scale
--------------------------------------------------------

Vector similarities arrive in a **narrow band of high values** (an
already-high cosine, mapped to ``(d+1)/2``), while hybrid and keyword scores
are min-max normalised across the whole of [0, 1]. That single difference in
*scale* — not corpus, not encoder — explains five results we first recorded as
unrelated findings:

1. The clamp in ``frequency_boost`` is load-bearing on vector legs (worth up
   to +0.285) and irrelevant on hybrid — a compressed base score cannot
   survive an unbounded multiplier.
2. Unclamped, the same aggregator slides toward a pure length prior.
3. Length penalties collapse: when the max score is nearly constant,
   ``max/nᵖ`` ranks by ``n`` alone for every ``p`` — landing *below* a control
   that ignores scores entirely, because it inverts a genuinely positive
   signal rather than discarding it.
4. Widening the hybrid pool helps because a chunk one leg found and the other
   missed is zero-filled — and on a narrow band, the chunk just outside the
   cutoff was nearly as good as the one inside.
5. Weighting a section roll-up by *score × overlap-share* destroys vector
   rankings (−0.053 to −0.147) while being nearly free on hybrid: the 0–1
   geometry factor dominates the narrow band and the ranking degenerates into
   an overlap prior. The shipped form is a **tie-break** — equal scores order
   by overlap share — which cannot reorder across different scores and
   measures as pure gain.

The transferable lesson: **two legs on different scales are not comparable,
and every multiplicative or ratio-shaped scoring rule is secretly a statement
about scale.** Before reasoning about a scoring formula, find out what range
its inputs actually occupy.

Sections are a return unit, not a retrieval level
-------------------------------------------------

The study's most robust negative result: **which retrieval level wins is
corpus-dependent, and none of the three rules we proposed to predict it
survived measurement.** "Sections win on real documents" — withdrawn (MAUD is
real; its section arm loses by 0.209). "Fusion never beats chunks" —
withdrawn (an artifact, see the lab notebook). "Fanout gates it" — refuted
twice, from opposite directions.

What survived is sharper: after controlling for reachability, a section
*vector* is not intrinsically better than rolled-up chunks. What sections buy
is **reach and a natural unit to hand back to a user** — which is why
LocalVectorDB invests in making section *return* correct (every section
reachable by roll-up, ``k`` honoured) rather than pushing a section index as
the default way to search.

One control worth running before any chunks-vs-sections claim of your own:
**how section-shaped is your gold?** On 18% of Qasper's query/answer pairs the
answer span *is* the section, so a section index wins those for free; on NQ
the figure is 3.6%. The comparison means different things on the two corpora.

The keyword leg: powerful, and already well-tuned
-------------------------------------------------

The BM25 leg is the biggest lever in the system — and its *parameters* are
not. Sweeps of ``k1``, ``b``, and BM25+ over four corpora found stock FTS5
defaults at or near the optimum on three of four, with one interaction worth
carrying to any tuning work anywhere:

  On the one corpus where each of three knobs helped (~+0.027 alone), all
  three together measured *worse than shipped* (−0.0254). They were one
  mechanism wearing three hats. **Never add two 1-D argmaxes.**

The known exception to "hybrid everywhere": on Wikipedia-like corpora that
repeat title terms in every section, BM25 cannot discriminate within a
document and the keyword leg turns significantly negative. The escape hatch is
``vector_weight`` (NQ's optimum is 0.9), which the library documents per
level rather than papering over with a heuristic.

How the study shaped the library
--------------------------------

Every one of these findings is now a shipped behaviour rather than a document:

- **Keyword + hybrid at every retrieval level** (the largest effect in the
  study).
- **``lvdb doctor`` / ``db.diagnose()``** — the regime diagnostic that
  replaces the six defaults that cannot exist, plus a calibrated ingest-time
  coverage warning.
- **``document_scoring_method="auto"``** — the one rule that generalised.
- **``BAAI/bge-reranker-base`` as the local reranker default**, with model
  guidance in the docs.
- **Retrieval prefixes** applied automatically for asymmetric encoders.
- **Section roll-up that reaches every section**, with a scale-safe
  overlap tie-break.
- **Three regression gates** (two retrieval corpora with derived quoted-query
  arms, a hierarchical gate, and an extraction-fingerprint gate) so future
  changes are measured against committed baselines rather than intuition.

If you want to verify the headline comparison on your own corpus rather than
trust ours, ``examples/section_vs_chunk_retrieval.py`` runs it end-to-end
against a real embedding backend and reports nDCG/recall per mode.

.. seealso::

   :doc:`retrieval-lab-notebook`
      How these numbers were established: the instruments, the controls, and
      the six wrong conclusions they caught.
