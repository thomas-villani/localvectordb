# Examples

Runnable scripts, not snippets. Each one is checked by the test suite so it
cannot quietly rot (see `tests/test_examples.py`).

## Prerequisites

These examples use **real embeddings**, because the things they demonstrate are
about relevance and relevance cannot be demonstrated with fake vectors. You need
one of:

```bash
# Option A - Ollama (the library default)
ollama serve
ollama pull nomic-embed-text

# Option B - fully local, no server
pip install sentence-transformers
```

The scripts auto-detect which is available, preferring Ollama. Override with
`--provider`.

They deliberately **refuse** `--provider mock`. `MockEmbeddings` seeds numpy's
RNG from a hash of the input text: deterministic, and semantically meaningless.
It is the right tool for exercising plumbing and the wrong tool for measuring
whether the right thing ranks first — a mock run prints a table of noise that is
indistinguishable from a table of results.

---

## `section_vs_chunk_retrieval.py`

Reproduces LocalVectorDB's central retrieval claim — that embedding each
section's own text and retrieving it alongside chunks beats chunk-only
retrieval on long structured documents — **on a corpus you choose**.

```bash
# Bundled sample corpus, with judgments, end to end
python examples/section_vs_chunk_retrieval.py

# Your documents, your judgments -> nDCG@10 and recall@k per mode
python examples/section_vs_chunk_retrieval.py \
    --corpus ./docs --judgments ./my_judgments.json

# Your documents, no judgments -> side-by-side of what each mode returns
python examples/section_vs_chunk_retrieval.py --corpus ./docs --query "..."
```

It builds **one** hierarchical database and queries it three ways, so the arms
differ only in how they are queried:

| Arm | Query | What it does |
|---|---|---|
| chunk-only (baseline) | `search_level="chunks"` | ordinary chunk retrieval — what you get without the feature |
| raw-span sections | `search_level="sections"` | retrieves section vectors, each embedded from the section's own text |
| fused | `search_level="fused", section_weight=0.65` | blends the two |

### Writing judgments for your own corpus

Copy the shape of `sample_corpus/judgments.json`:

```json
{
  "queries": [
    {
      "query": "how long is customer data kept after an account is closed?",
      "relevant_docs": ["data-retention"],
      "relevant_sections": ["data-retention::Retention Periods"]
    }
  ]
}
```

A document id defaults to the filename stem (`data-retention.md` →
`data-retention`), because that is what `upsert_from_file()` does. Sections are
addressed as `<doc id>::<section heading>` rather than by index, so you can read
them off the document instead of counting.

Twenty or thirty queries is enough to see a real signal. Twelve, as in the
sample, is not — the script says so in its output.

### What the sample corpus can and cannot tell you

The sample is three short documents and twelve queries, and it is honest about
its own limits when you run it:

- **The document target saturates.** With three documents and `k=10`, every mode
  retrieves the entire corpus and all of them score 1.0. That measures the
  corpus, not the modes.
- **The section margin is inflated.** The sample scores section retrieval about
  +0.5 nDCG@10 over the baseline. The published study finds +0.07 to +0.17 on
  real papers. The sample's documents are short, cleanly sectioned, and each
  query has one obviously-correct section — close to the best case. Do not quote
  the sample's number.
- **Fused does not beat pure section retrieval at the section target**, in the
  sample and in [the study][study] alike (its finding #4, in every cell). Fusion
  helps at the *document* target. This is the reason the README claims "raw-span
  section retrieval, alone or fused with chunks" rather than claiming fusion
  always wins.

The evidence for the claim is [the study][study] — 15 papers, 48 queries, three
local encoders. The point of this script is to let you get the same numbers for
*your* documents, which are the only ones that decide whether the feature is
worth it to you.

[study]: https://thomas-villani.github.io/localvectordb/hierarchical-evaluation.html
