The Lab Notebook
================

:doc:`retrieval-study` presents what the retrieval study found. This page is
about **how it was found** — the instruments, the disciplines, and above all
the wrong conclusions we drew first and the controls that caught them. We
publish it because the failure modes turned out to be worth more than the
findings: the findings are each worth perhaps half an nDCG point, while the
failure modes recur in every measurement effort we have seen, including ours.

Everything here happened while measuring one system (this library) on public
corpora — Qasper, Natural Questions, MLDR, MAUD, BEIR SciFact and NFCorpus —
with the tracked harnesses under ``benchmarks/``. Every number was re-read
from a result file while writing, not quoted from notes; that rule exists
because we broke it once and published two figures from memory that were wrong
in the second decimal.

.. contents::
   :local:
   :depth: 2

The shape of every error
------------------------

Several conclusions were formally withdrawn during the study, and they share
one shape:

  **A difference attributed to the mechanism under study that actually came
  from the baseline it was measured against.**

Not one of them was caught by re-reading the analysis. Every one was caught by
**re-running the measurement with one thing changed**. That is the whole
lesson, and it is why the episodes below are organised by *what control
cracked the error* rather than by what the error was.

A second regularity is worth stating up front: **every error flattered the
hypothesis of whoever made it.** Assume the next one will too.

Episode 1: the hybrid-default confound
--------------------------------------

``db.query()`` defaults to ``search_type="hybrid"``. For most of the study,
FTS5 indexed chunks only — so every comparison between retrieval *levels*
silently gave the chunks arm a free BM25 leg worth +0.084 to +0.131, which the
section and fused arms structurally could not receive. Three headline
conclusions reversed when everything was re-run on ``search_type="vector"``.

**How it was caught is the useful part.** Not by suspicion — by a violated
bound. In a fusion-weight sweep, the ``w=0`` arm is *definitionally* the chunk
path with the section leg switched off, so ``w=0`` and the plain chunks arm
had to agree. They didn't. When a provable bound is violated, the right
question is not "where is my bug" but **"are these two things actually the
same system?"**

The diagnostic signature, should you ever meet it: switching hybrid→vector
moves *only* one arm, and every other number is identical to four decimals —
meaning those arms were already vector-only.

The fix was not more careful comparisons; it was making the asymmetry
impossible (a keyword leg for every level) and teaching the gate to refuse
cross-``search_type`` comparisons outright.

Episode 2: the reachability confound
------------------------------------

Chunk→section attribution credited each chunk to the single section holding
its midpoint. At the default chunk size — larger than the median section on
both sectioned corpora — one chunk swallows about two sections and credits
one. Consequence, identical on two unrelated corpora: **40.1% of sections
owned no chunk** and could not be reached by chunk roll-up at any ``k``.

The section index's measured advantage was therefore partly free: drop the
queries whose gold is unreachable and NQ's +0.107 section win becomes −0.001.

Three details make this the most instructive episode:

- **It was the residue of a repair, not an unfixed bug.** An earlier fix had
  repaired the *section-vector* side of the same asymmetry and deliberately
  kept the roll-up side, on measured evidence. The handicap didn't disappear;
  it moved arms. Read the git history of the thing you think you just found.
- **The first corrected headline was also wrong.** "The entire advantage is
  unreachability" — no. Repairing the attribution properly recovers ~45% of
  NQ's gap and none of Qasper's. Half bookkeeping, half real.
- **Two valid controls disagreed**, because they answer different questions.
  Dropping unreachable queries asks "is roll-up good where it works?";
  repairing the mapping asks "would fixing attribution be enough?" — and
  conditioning on an outcome correlated with difficulty makes the first the
  weaker instrument. Keep it as a diagnostic; never read a verdict off it.

The repair itself carried one final lesson (found while shipping it, a week
after the study): crediting overlapped sections at *score × overlap-share*
looked principled and destroyed vector rankings (−0.147 on NQ), because a 0–1
geometry factor dominates the narrow band vector scores occupy. Delivered as a
**tie-break** instead — equal scores order by share — the same information is
pure gain. Secondary signals on a compressed scale must never multiply.

Episode 3: the coverage thesis — wrong constant, then wrong instrument
----------------------------------------------------------------------

The thesis: retrieval degrades at large chunk sizes because text past the
encoder's context is silently discarded. We had a constant —
``chars_per_token = 3.5`` — to compute coverage.

First, **the constant was 20–25% low** (the encoder's own tokenizer says
4.38), understating published coverage by up to 16 points. Worse, a constant
*cannot* be shipped: per chunk the true ratio spans 2.00–5.67 on one corpus,
so a character cap misclassifies 28–31% of chunks near the boundary while
looking accurate in aggregate, because the errors compensate. **Aggregate
agreement is not per-item agreement** — check the item level before trusting
any proxy.

Then the thesis died anyway. We built the same corpus twice, once with a
crippled context window, and diffed the stored vectors:

.. list-table::
   :header-rows: 1
   :widths: 20 30 20 20

   * - chunk_size
     - vectors changed
     - mean cosine
     - Δ nDCG@10
   * - 219
     - 0 / 10,421
     - 1.000000
     - 0.0000
   * - 500
     - 2,077 / 4,478 (46%)
     - 0.9972
     - −0.0000
   * - 1000
     - 1,713 / 2,478 (69%)
     - 0.9360
     - −0.0571

**46% of chunks truncated, and the score did not move at four decimals.**
Clipping the tail off a chunk barely rotates its vector; damage requires
measurable displacement. Coverage was retired as a damage predictor in favour
of diffing vectors between paired builds — and the ``chunk_size=219`` row is
as important as the headline row: a null rung proving the encoder
deterministic is what makes "0 changed" elsewhere meaningful.

(The degradation at large chunk sizes is real — it is *granularity*, not
truncation: between sizes 219 and 1000, coverage is 100% everywhere and
vector nDCG still falls 0.162.)

Episode 4: "fanout gates it" — refuted from both directions
-----------------------------------------------------------

An appealing rule kept suggesting itself: aggregation effects should scale
with fanout, since fanout is literally how much there is to aggregate. It
failed twice, in opposite directions — one corpus with fanout 3.68 showed a
loss where another at 4.28 showed a gain; and the corpus with the *highest*
fanout in the study showed no pool-width effect at all.

What survived is a decomposition that has held since: **fanout sets the
magnitude of an aggregator's deviation from max; the target unit sets the
sign.** At fanout ~1 every aggregator is a no-op — one of our "neutral"
results turned out to be an artifact of a corpus where the aggregator under
test *equals* max by construction. Hence a standing house rule: read the
fanout column before the nDCG column.

Episode 5: the rule that died when the corpus got bigger
--------------------------------------------------------

The most valuable episode, because the mistake survived a careful
generalisation check.

A rule ("pool width and aggregator must move together") was derived on one
corpus and confirmed independently on the real query path — same corpus. We
then rebuilt that corpus at **4× the size**: same papers, same encoder, same
code, more distractors. Half the rule flipped from null to significantly
positive. The rule was a small-haystack artifact.

The control that makes it airtight: "bigger corpus behaves differently" is
confounded between more distractors and a differently-drawn query set, so we
scored the *identical* original queries against the enlarged index — one
variable — and the flip reproduced. Mechanism: on a small corpus the gold
document is usually already inside the candidate pool, so widening admits
distractors; once the corpus outgrows the pool, widening genuinely repairs
recall. Both of our earlier readings were true — *of different regimes*.

  We tested generalisation across three corpora and two encoders with
  confidence intervals throughout, and still missed this — because **corpus
  size was never an axis**. Nothing new was needed to refute the rule: same
  data, same code, more of it.

Episode 6: point estimates lie, with a tally
--------------------------------------------

Once paired-bootstrap confidence intervals were attached to everything, **2 of
8 cross-level comparisons flipped from "win" to "tie"**. At n=100–200, a 95%
interval on nDCG@10 is roughly ±0.06 — wider than several effects that had
been quoted to four decimal places. On a separate topology question, eight
arms of point estimates all pointed one way and every interval straddled zero;
we shipped the other design on code-simplicity grounds and recorded the
question as *measured-and-declined*, not resolved.

Pairing is what makes the intervals affordable: arms scored on the same
queries are strongly correlated, and an unpaired test throws that power away.

The house rules
---------------

Each earned by a specific mistake above:

1. **Never compare across ``search_type``.** The gate refuses.
2. **Attach a confidence interval to every cross-level claim.**
3. **Diff the vectors of paired builds** before believing a score delta or a
   coverage number — and always include a null rung.
4. **Aggregate agreement is not per-item agreement.**
5. **Read the fanout column before the nDCG column.**
6. **When a provable bound is violated, ask whether the two things are the
   same system.**
7. **Doubt the effect, not just the mechanism** — a refuted mechanism is a
   reason to re-measure the effect, not to keep the effect and shop for a new
   mechanism.
8. **Check the incidence of the thing you changed in your gate corpus before
   trusting the gate** — see below.

Gates, and the art of not being blind
-------------------------------------

Three shipped changes in a row passed every regression gate at exactly
+0.0000 — *by construction*, because the gate corpora could not express the
thing that changed: a scoring default the gates never exercised, a
phrase-handling fix on corpora with essentially no quoted queries, and
aggregation changes on a corpus with 1.08 chunks per document. Each +0.0000
proved inertness for existing callers, which is genuinely valuable — and
proved nothing about the change working.

The remedy was not discipline; it was **gate corpora with the properties we
kept needing**. The suite now runs:

- ``benchmarks/eval_retrieval.py --dataset all --check`` — two corpora
  (SciFact, plus Qasper for real fanout and headroom), each sweep including
  two **derived quoted-query arms**: a frozen transform quotes a bigram in
  ~90% of gate queries, so a phrase-handling regression now craters two gated
  configurations instead of passing silently.
- ``benchmarks/eval_hier_gate.py --leg both --check`` — the hierarchical path
  on a real corpus plus a synthetic long-section leg, since each leg is blind
  to the other's failure mode.
- ``benchmarks/eval_extraction.py --check`` — fingerprints of the file
  extractor's markdown over committed fixtures, because neither retrieval
  gate touches extraction, and a dependency upgrade once changed the
  extracted text of 30/30 PDFs (headings −7.2%) while every gated number
  stayed +0.0000. The fixtures include a PDF engineered around the known
  failure mode (hyphen-wrapped, multi-line headings).

The instruments
---------------

All under ``benchmarks/``, all zero-embedding by default — they read a vector
cache and *raise* on a miss, so an unseen text stops the run rather than
quietly costing money. Two design disciplines did the most work:

**Capture once, sweep offline.** Run retrieval a single time at the widest
pool, keep the per-query candidate lists, then apply every variant in numpy.
It turns 96 cells into one retrieval pass — and it is why attaching a
confidence interval to everything was affordable.

**Verify fidelity before printing a number.** No harness that substitutes a
function reports anything until a verification pass proves the substitution
reproduces the untouched path per query, exactly, at the shipped setting.
This has caught real bugs — one harness carried up to 2× the candidates in
one leg of a weighted comparison. The effect was small, but it is the
difference between measuring the operator that ships and one that doesn't.

Honest caveats
--------------

- Effects below ~0.011 are inside our measured build-to-build noise and are
  reported only with intervals.
- Most corpora were measured under one encoder each; where a finding needed
  encoder generality we replicated on a second, but not everywhere.
- One synthetic leg (the long-section gate corpus) exists precisely because
  no real corpus with parseable long sections was available to the section
  detector; its absolute numbers are pessimistic by construction and only its
  deltas are read.
- Several conclusions are about *this* stack (SQLite FTS5, FAISS flat
  indexes, mean-pooled span vectors). The mechanisms — scale mismatch,
  baseline attribution, regime-dependence — are the parts we would expect to
  transfer.

.. seealso::

   :doc:`retrieval-study`
      The findings themselves, organised by conclusion.
