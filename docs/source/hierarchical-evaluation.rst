Raw-Span Hierarchical Retrieval: Method and Evaluation
======================================================

This is the "why it works" companion to :doc:`hierarchical` (which is the
"how to use it" reference). It explains the technique behind
``search_level="sections"`` / ``"fused"`` from first principles, then presents a
controlled study measuring it across **local** embedding encoders.

If you just want to turn it on, read :doc:`hierarchical`. If you want to know
whether it is worth turning on for *your* corpus and encoder — and why — read on.

.. contents::
   :local:
   :depth: 2

The problem: flat chunk retrieval loses the forest for the trees
----------------------------------------------------------------

A vector database embeds documents by cutting them into fixed-size **chunks** and
indexing one vector per chunk. For a query whose answer is a single sentence,
this is close to optimal: the chunk *is* the answer, and a precise chunk match
beats anything coarser.

It breaks down when relevance is **diffuse**. In a long, structured document —
a paper, a manual, a report — the material that answers a question is often
*spread across a whole section* rather than concentrated in one passage. No
single chunk scores highly, so the right document ranks below a short document
that happens to contain one lexically-similar sentence. Flat chunking has thrown
away the document's structure, and with it the notion that a *region* can be
relevant even when no single *point* is.

The method: a three-level hierarchy
-----------------------------------

LocalVectorDB's answer is to index the same document at three granularities and
let the query choose:

.. code-block:: text

   document   ← one vector for the whole document (coarsest)
     └── section   ← a run of chunks under one heading
           └── chunk   ← the usual fine-grained unit (default)

Sections are an **overlay**, not a different chunking strategy: the document is
chunked exactly as before, sections are detected from its heading structure, and
each chunk is assigned to its containing section. Each level gets its own FAISS
index, so one database can be searched at whichever granularity fits the
question. See :doc:`hierarchical` for the API.

The crux: what vector represents a section?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything hinges on how you build the *section* vector. There are two options,
and the difference is the whole point of this study.

**Centroid (the obvious, free option).** Average the section's chunk vectors.
It costs nothing extra — the chunk embeddings already exist — but a centroid is a
*blur*. Averaging unit vectors discards every cross-chunk interaction:
coreference, the way a topic composes across paragraphs, the arc of an argument.
What survives is a mean direction, which for a multi-topic section points
"somewhere in the middle" and matches nothing sharply.

**Raw-span (embed the section's actual text).** Concatenate the section's chunks
back into their original span and embed *that text* directly. The encoder sees
the section as continuous prose and produces a vector that reflects its full
composition, not a mean of fragments. This is the LocalVectorDB default
(``section_vector_strategy="rawspan"``).

Raw-span costs one extra embedding call per section at ingest — but sections are
far fewer than chunks, so the overhead is modest, and the hypothesis is that the
representation is meaningfully better. This study tests that hypothesis.

Handling sections longer than the encoder's context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A section can be much larger than a chunk — larger, sometimes, than the encoder's
context window. Truncating would silently drop the section's tail, and the
section vector is the load-bearing arm, so LocalVectorDB never truncates. Instead
it splits an over-long span into **windows** sized to the encoder's context,
embeds each, and **mean-pools** the window vectors. A section that fits in one
window is embedded whole; a longer one is represented in full by the pool.

Pooling is not free of cost, though — it reintroduces a little of the
averaging blur that raw-span exists to avoid. So the *fewer* windows a section
needs, the cleaner its vector. This is exactly why encoder context length
matters, and it shows up in the results below.

Fused retrieval: combine the levels instead of choosing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The chunk level and the section level make **different mistakes**. A chunk match
is precise but narrow; it misses diffuse relevance. A section match captures
diffuse relevance but is coarse; it can rank a broadly-on-topic section above the
document that actually answers the question. ``search_level="fused"`` runs both
searches, puts the two score distributions on a common scale by **min-max
normalising each within the query's own candidate pool**, and blends them with a
``section_weight`` (default 0.65, leaning toward the section leg). This is the
same relative-score fusion LocalVectorDB uses for hybrid vector+keyword search.

The Ollama context trap (a real-world prerequisite)
---------------------------------------------------

Raw-span only pays off with a **long-context encoder** — otherwise every section
is chopped into many windows and the pooling blur dominates. That makes the
encoder's *effective* context window a first-class variable, and on Ollama it
hides a trap worth documenting.

Ollama's ``/api/embed`` silently caps each input at ``n_batch`` — **default
2048** — *not* at ``num_ctx``. For an encoder model (non-causal, whole input
processed in one batch) that batch size is the true input ceiling. Raise
``num_ctx`` to 8192 and, by itself, **nothing changes**: inputs past 2048 tokens
are still truncated, and an "8k" run is bit-for-bit an "2k" run. This is not
documented in Ollama's API reference.

The fix is to also raise ``options.num_batch``. LocalVectorDB's
:class:`~localvectordb.embeddings.OllamaEmbeddings` now auto-sets
``num_batch = num_ctx``, so asking for an 8k context actually gives you one. The
study below verifies the fix bites: the 8k arm needed to window-pool only 5
over-long spans, against 20 on the 2k arms — direct evidence that more spans fit
in a single window once the batch ceiling is lifted.

Experimental design
-------------------

**Dataset.** Qasper ``dev`` — real NLP papers with natural section structure and
evidence-paragraph relevance judgments. This study uses a 15-paper slice
(48 queries; 48/50 questions kept, 107/110 evidence spans located). The
reference-encoder baseline above used the full set (275 papers, 882 questions)
plus synthetic section corpora built from BEIR (FiQA, NFCorpus).

**Metric.** nDCG@10.

**Two relevance targets**, because "hierarchical retrieval helps" means different
things depending on what you are ranking:

- **DOC** — which *paper* holds the answer.
- **SECTION** — which *section* holds the answer. This is the direct test of the
  hierarchy premise: if section vectors are good, they should shine here.

**Arms.** For each encoder we score, per query: ``chunk`` (the flat baseline),
``rawspan-section`` / ``centroid-section`` (the two section strategies),
``rawspan-doc`` / ``centroid-doc`` (document level), and the two fused rankings
``fusion-rawspan`` / ``fusion-centroid``. An *oracle* that picks the best level
per query bounds the headroom.

**Encoders and context.** Three local Ollama encoders, to test whether a
technique tuned on OpenAI generalizes:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - encoder
     - ``num_ctx``
     - note
   * - ``nomic-embed-text``
     - 2048
     - architecture-capped at 2048
   * - ``embeddinggemma:300m``
     - 2048
     - architecture-capped at 2048
   * - ``bge-m3``
     - 2048
     - 2k baseline for the context axis (finding #5)
   * - ``bge-m3``
     - 8192
     - true 8k, reachable only via the ``num_batch`` fix

**Two chunk sizes** — 500 tokens (the shipped default) and 1000 tokens — to test
whether the hierarchy's advantage is merely an artefact of small chunks. At
Qasper's ~4.3 chars/token a 1000-token chunk is ~5.5k chars, still inside a
2k-context model's ~6144-char window, so it is embedded in **one** pass — a
faithful large-chunk baseline, not a pooling artefact.

**Harness.** ``benchmarks/eval_hierarchical.py`` embeds every span through the
real provider, caches vectors per ``(model, text)`` on disk, and scores each arm
by exact cosine in NumPy — no FAISS, no database — so the numbers isolate the
representation, not index or fusion plumbing. The full reproducibility record,
including the exact commands, is ``benchmarks/HIERARCHICAL_LOCAL_ENCODERS.md``.

The reference-encoder baseline (OpenAI)
---------------------------------------

Raw-span shipped as the default because a *first* evaluation — on OpenAI
``text-embedding-3-small`` (an 8k-token encoder) — measured it beating both the
chunk baseline and the centroid. That evaluation scored nDCG@10 on two kinds of
corpus: **synthetic multi-section super-documents** built from judged BEIR
passage corpora (FiQA, NFCorpus — which give aligned relevance labels at
document, section, and passage granularity), and **Qasper** at full size
(275 papers, 882 questions).

On Qasper, the raw-span section beat both the chunk baseline and the centroid at
every target:

.. list-table::
   :header-rows: 1
   :widths: 44 28 28

   * - arm
     - doc nDCG
     - sec nDCG
   * - chunk (baseline)
     - 0.375
     - 0.177
   * - centroid-section
     - 0.371
     - 0.173
   * - **rawspan-section**
     - **0.398**
     - **0.200**

Fusing raw-span sections with chunks lifted the document target further still:

.. list-table::
   :header-rows: 1
   :widths: 44 28 28

   * - arm
     - doc nDCG
     - sec nDCG
   * - chunk (baseline)
     - 0.375
     - 0.177
   * - rawspan-section
     - 0.398
     - 0.200
   * - **fusion (chunk+sec)**
     - **0.404**
     - 0.196

The absolute gains on real long documents are **modest** — low single-digit
nDCG@10 (~0.02–0.03) — a genuine quality-per-cost judgement, not a slam dunk. The
effect is far larger on section-dense corpora: on the synthetic FiQA section set
both fusion and the raw-span section cleared ~0.79 against a chunk baseline of
~0.67. And averaging really does throw away signal — on some corpora the
*centroid* section scored *below* chunk-only while the raw-span section scored
well above it.

The study below asks the obvious next question: does this hold on encoders other
than OpenAI?

Local-encoder results
---------------------

nDCG@10, Qasper dev, 15 papers / 48 queries. Best arm per column in **bold**.

**DOC target** — which paper holds the answer:

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 10 10 10 10 10 10

   * - arm
     - nomic·500
     - nomic·1000
     - egemma·500
     - egemma·1000
     - bge2k·500
     - bge2k·1000
     - bge8k·500
     - bge8k·1000
   * - **rawspan-section**
     - **0.763**
     - **0.763**
     - 0.756
     - 0.756
     - **0.796**
     - **0.796**
     - **0.784**
     - **0.784**
   * - fusion-rawspan
     - 0.760
     - 0.737
     - **0.780**
     - **0.764**
     - 0.768
     - 0.752
     - 0.742
     - 0.750
   * - chunk (baseline)
     - 0.707
     - 0.683
     - 0.678
     - 0.723
     - 0.741
     - 0.719
     - 0.741
     - 0.719
   * - centroid-section
     - 0.693
     - 0.690
     - 0.661
     - 0.722
     - 0.718
     - 0.725
     - 0.718
     - 0.725
   * - rawspan-doc
     - 0.664
     - 0.664
     - 0.703
     - 0.703
     - 0.711
     - 0.711
     - 0.686
     - 0.686

**SECTION target** — which section holds the answer:

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 10 10 10 10 10 10

   * - arm
     - nomic·500
     - nomic·1000
     - egemma·500
     - egemma·1000
     - bge2k·500
     - bge2k·1000
     - bge8k·500
     - bge8k·1000
   * - **rawspan-section**
     - **0.367**
     - **0.367**
     - **0.315**
     - **0.315**
     - **0.419**
     - **0.419**
     - **0.415**
     - **0.415**
   * - fusion-rawspan
     - 0.294
     - 0.259
     - 0.304
     - 0.243
     - 0.312
     - 0.335
     - 0.315
     - 0.334
   * - chunk (baseline)
     - 0.256
     - 0.198
     - 0.246
     - 0.211
     - 0.267
     - 0.259
     - 0.267
     - 0.259
   * - centroid-section
     - 0.238
     - 0.200
     - 0.236
     - 0.203
     - 0.285
     - 0.275
     - 0.285
     - 0.275

What the numbers say
--------------------

**1. Raw-span sections win on every local encoder, both chunk sizes, both
targets.** The reference-encoder baseline established the effect on OpenAI; it now
holds on three local encoders too, with ``rawspan-section`` leading the chunk
baseline by **+0.03–0.08 on DOC** and **+0.07–0.17 on SECTION**. The premise —
that a section's own embedding is a better retrieval unit than its chunks'
average, or than the chunks alone — is encoder-independent.

**2. Raw span decisively beats the centroid.** ``rawspan-section`` outscores
``centroid-section`` almost everywhere, and the oracle bound makes the gap
starker: letting an oracle mix in raw-span sections lifts nDCG by +0.08–0.13
(DOC) and +0.20–0.25 (SECTION) over chunk-only, while the *centroid* oracle adds
almost nothing (+0.004–0.08). Averaging really does discard the signal that
embedding the span keeps — exactly the effect the method was designed around.

**3. Bigger chunks do not erode the advantage — they widen it.** A natural worry
is that the hierarchy only helps because chunks are small; make chunks bigger and
each already captures a section's worth of context, so the section level becomes
redundant. The data shows the reverse. On the SECTION target, a 1000-token chunk
makes the *chunk* baseline **worse** (a large chunk straddles section boundaries
and blurs them), while ``rawspan-section`` is chunk-size invariant. The gap
*grows* with larger chunks. Section-aware retrieval is not a small-chunk crutch;
for coarse-grained relevance it is doing something chunking cannot.

**4. Fusion helps at the document level but dilutes section precision.**
``fusion-rawspan`` sometimes tops the DOC target, but on the SECTION target it
loses to *pure* ``rawspan-section`` in every cell — blending the chunk leg back
in adds noise when the query's relevance is genuinely section-shaped. The default
``section_weight=0.65`` is a document-favouring compromise; there is headroom for
a query-adaptive weight, but the target is unknown at query time, so a fixed lean
toward the section leg is the pragmatic choice. Match the level to your data (see
:doc:`hierarchical`).

**5. The winner is the encoder, not the context — and pooling is enough.**
``bge-m3`` is the strongest arm overall, but running it at **both** 2k and 8k
context isolates *why*. Only the raw-span arms embed spans long enough to feel
the context window; ``chunk`` and ``centroid-section`` embed sub-window text and
are context-invariant — and indeed their 2k and 8k numbers match to the digit, a
built-in sanity check that the two runs differ only where they should.

.. list-table:: ``bge-m3`` ``rawspan-section`` nDCG@10, same model at two contexts
   :header-rows: 1
   :widths: 30 20 20 30

   * - target
     - 2k context
     - 8k context
     - over-long spans pooled
   * - DOC
     - **0.796**
     - 0.784
     - 20 → 5
   * - SECTION
     - **0.419**
     - 0.415
     - 20 → 5

Raising context from 2k to 8k did **not** improve retrieval — it is flat, even a
hair lower — despite cutting window-pooling from 20 over-long spans to 5.
Window-mean-pooling at 2k already represents a Qasper section well enough that the
extra context buys nothing here; ``bge-m3@2048`` is in fact the single best
section arm in the whole study (0.419). So ``bge-m3``'s lead over ``nomic`` and
``embeddinggemma`` is **model quality**, not context length.

Two consequences. The ``num_batch`` fix is *validated* — the 8k arm genuinely
embeds 8k tokens (pooling drops to 5) — but on this corpus its retrieval **payoff
is nil**, which is good news: you do not need to chase a long-context encoder (or
fight Ollama's batch ceiling) for good section retrieval, because pooling covers
the overflow. The earlier limitation still holds at the *short* end — a 512-token
encoder pools so aggressively that raw-span degrades — but between 2k and 8k, on
this data, context is not the lever.

Threats to validity
--------------------

- **Sample size.** 48 queries is small; treat every single cell as noisy. The
  confidence here comes from the *consistency* of the pattern across three
  encoders, two chunk sizes, and two targets — not from any one number.
- **Context vs encoder-quality confound: resolved.** An earlier draft could not
  separate "8k context" from "``bge-m3`` is a stronger encoder", because the
  ``bge-m3@2048`` arm had not completed (``bge-m3`` is a 1.2 GB model and exceeded
  the client's embedding timeout on a CPU-only box; a longer timeout fixed it).
  That arm is now in, and finding #5 reports the controlled result: context is not
  the driver — the same model at 2k and 8k retrieves within noise, so the lead is
  encoder quality. One confound remains that this design cannot break: ``bge-m3``
  is both stronger *and* higher-dimensional (1024 vs 768), so "encoder quality"
  bundles capacity with dimensionality.
- **Corpus.** The local-encoder study is Qasper-only. The reference-encoder
  baseline above additionally covers synthetic BEIR section corpora, where the
  effect is larger.
- **Hardware.** Timings are from a memory-constrained CPU box and are not
  representative; only the relative nDCG numbers transfer.

Reproducing this study
----------------------

The whole sweep, one process per ``encoder × chunk-size``:

.. code-block:: bash

   bash benchmarks/run_hier_ollama.sh 15

A single arm (here the ``bge-m3@2048`` cell, with the longer timeout ``bge-m3``
needs on a CPU box):

.. code-block:: bash

   ./.venv/Scripts/python.exe benchmarks/eval_hierarchical.py \
     --dataset qasper --split dev --max-papers 15 --mode section \
     --provider ollama --model bge-m3 --num-ctx 2048 --chunk-tokens 500 --timeout 1800

Vectors are cached per ``(model, num_ctx, text)``, so re-runs only embed spans
they have not seen. The full record — commands, raw tables, the timeout gap, and
the caveats above — lives in ``benchmarks/HIERARCHICAL_LOCAL_ENCODERS.md``.

.. seealso::

   :doc:`hierarchical`
       The user-facing reference: enabling hierarchical embeddings, section
       detection, ``search_level``, and ``section_weight`` tuning.
