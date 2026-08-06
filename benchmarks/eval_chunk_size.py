"""DIAGNOSTIC: sweep ``chunk_size`` at a fixed encoder to find the r~=1 crossover.

THE RULE UNDER TEST. ``section_rawspan`` and ``section_centroid`` are one
mean-pooling operator at two granularities, and the finer pool wins. That makes
the winner predictable from configuration alone:

    rawspan piece = max_input_tokens x 3.0   (_span_embed._WINDOW_CHARS_PER_TOKEN)
    centroid piece = chunk_size      x 4.2   (_CHARS_PER_TOKEN_FALLBACK, MEASURED)
    r = rawspan / centroid   ->   r < 1 favours rawspan, r > 1 favours centroid

It has been checked at exactly **two** points, on opposite sides of r=1 across a
32x range -- MiniLM @ chunk 500 (r=0.44, rawspan won 6/6, §6.32) and openai @ an
8,191-token window (r=14.0, centroid won by 0.36, §6.20) -- and **never near
r=1**, which is the only place a tuning guide actually gets consulted. Neither
point varied ``chunk_size``. This file varies it, holding the encoder fixed, so
the crossover is measured rather than interpolated.

THE CONFOUND THE RULE MISSES. ``all-MiniLM-L6-v2`` has ``max_seq_length`` 256 and
sentence-transformers truncates internally, so a chunk vector encodes at most
~896 chars **no matter how large chunk_size is**. Above ~256 tokens the centroid
piece stops getting coarser as text and only gets coarser as *attribution*. So
the shipped ``chunk_size=500`` is r=0.44 nominal but r=0.86 effective, and the
existing two-point fit cannot tell which quantity governs. The grid straddles
both thresholds deliberately:

    C=64,128       nominal r > 1, no truncation      -> centroid predicted
    C=219          nominal r = 1.00, no truncation   -> predicted TIE
    C=256          r=0.86, truncation just binding
    C=500,1000     nominal r 0.44/0.22, effective r pinned at 0.86

If nominal chunk_size governs, rawspan's margin keeps widening from 256 to 1000.
If the encoded piece governs, the margin flattens there. The two rungs above 256
are the discriminator, and they cost the least (truncation means less text is
actually encoded).

r IS COMPUTED FROM MEASURED CHUNK LENGTHS, not from the 3.5 chars/token constant.
``chunk_overlap`` is 1 *sentence*, which is a few percent of a 500-token chunk but
a large fraction of a 64-token one, so the nominal size overstates how much fresh
text small chunks cover. The mean chunk length is read back out of each built DB
and both r values are reported from it.

LEG. The §6.32 density ladder at its top rung (12.5% gold, 100 queries), because
that is where the section arms score highest and the rawspan-vs-centroid gap is
most resolvable. Its ``chunk_size=500`` DBs are already built, so that rung is
free and reproduces published numbers (rawspan 0.4536 / centroid 0.3537 doc,
0.4426 / 0.3308 section) as a cache-integrity check on the whole sweep.

THE SECOND ENCODER, AND WHY ONE LADDER CANNOT CONCLUDE. The MiniLM ladder found
plain ``search_level="chunks"`` nDCG to be an inverted U peaking at c=219, close
enough to MiniLM's own 256-token context to look causal. It is confounded: with
one encoder "the optimum tracks the encoder's context window" and "the optimum is
~220 tokens because longer chunks dilute the query match" predict the identical
curve, and they give **opposite** advice about buying long-context models. So the
ladder is re-run on ``embeddinggemma:300m`` (ollama, 2048-token context, 8x
MiniLM's) over the same leg, changing only the encoder:

    optimum stays at ~219   -> dilution; context length buys nothing here
    optimum moves to ~1750  -> it tracks the encoder; long context earns its keep

Rungs are ordered so the decisive pair (1000 vs 219 -- 1000 was MiniLM's *worst*
rung) resolves first, then egemma's own r=1 point at 1750, the shipped 500, a
rung past its context at 3000, and finally 128/64 for comparability with the
MiniLM curve's fine end. Small chunk sizes cost the MOST wall-clock, not the
least: ``chunk_overlap=1`` is one sentence at every size, so c=64 re-encodes 10.7M
chars against c=1000's 7.7M.

THE THIRD POINT, AND WHY IT IS NOT A THIRD ENCODER (``--num-ctx``). Two ladders
gave a **plateau with a cliff**: 219->1750 spans 0.028 nDCG on egemma, then
coverage falls to 55% and the score drops 0.109. The cliff looks like a coverage
effect, but "coverage" and "encoder identity" are still perfectly confounded --
each encoder has exactly one context length, so every point on the coverage axis
is also a different model. A third encoder does not break that; it adds a third
model with a third context and a third set of weights, training corpus and
dimension.

``--num-ctx`` breaks it. Ollama truncates each input at ``num_ctx`` (and at
``num_batch``, which ``OllamaEmbeddings`` moves with it, because an encoder model
embeds its whole input in one llama.cpp batch). Lowering it on egemma holds the
weights, the corpus, the chunker and the queries **exactly** fixed and varies
only how much of each chunk the model reads.

WHAT THAT LADDER ACTUALLY RETURNED, and why coverage is the wrong instrument
(2026-08-06). ``num_ctx=512`` verifiably takes effect -- bisecting Ollama's
truncation boundary puts the cap at 1,797-2,315 chars against 8,143-9,010 at
2048, and the native build matches the 2048 build to the character, so "native"
is 2048. But the paired builds refute coverage as the mechanism:

    c      vectors CHANGED   mean cos   delta nDCG@10
    219      0 / 10,421      1.000000     0.0000
    500  2,077 /  4,478      0.9972      -0.0000
    1000 1,713 /  2,478      0.9360      -0.0571
    1750   981 /  1,577      0.8967      -0.0396

Read the c=500 row twice. **46% of its chunks were truncated and the score did
not move at four decimal places.** So "was it truncated" is not the question --
"how far did the vector move" is. Displacement predicts the damage; coverage does
not, because lopping the tail off a chunk that only just exceeds the cap barely
moves its vector. Damage needs 1-cos above ~0.01; below that truncation is free.

The c=219 row also establishes that the encoder is deterministic ACROSS BUILDS
(10,421 vectors, zero drift at 1e-6), which is what licenses reading any of the
other rows as truncation rather than noise.

The honest limit: a 2048-trained model run at 512 is not the same thing as a
natively-512 model, so this establishes MECHANISM, not external validity. A third
real encoder is the complementary experiment, not a substitute -- but it is the
expensive one, and it answers the weaker question.

Zero API spend either way -- sentence-transformers is in-process, ollama is local.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.config import EVAL_EMBEDDING_MODEL, EVAL_EMBEDDING_PROVIDER  # noqa: E402
from benchmarks.eval_hier_gate import Leg, build_configs, build_db, run_config  # noqa: E402

logger = logging.getLogger("eval_chunk_size")

SECTIONS = 3
PASSAGES = 32
QUERIES = 100
GOLD = 4
MIN_QUERY_GOLD = 4
SEED = 0

LIBRARY_DEFAULT_CHUNK_SIZE = 500
# Ordered by information value, not by size: the free cache-anchored rung first,
# then the predicted crossover, then the truncation discriminator. A run cut
# short still answers the question it was launched for.
CHUNK_SIZES: Tuple[int, ...] = (500, 219, 128, 1000, 256, 64)

_WINDOW_CHARS_PER_TOKEN = 3.0  # mirrors _span_embed

# MEASURED, not assumed (2026-08-06). The old value here was 3.5, mirrored from
# _span_embed, where it was a guess. It is 20-25% LOW, and because chunk lengths
# concentrate right at the cap, a constant error of that size understated reported
# coverage by up to **16.4 points** (c=256: 81.3% reported against 97.7% true).
#
#   all-MiniLM-L6-v2   4.38   BertTokenizer, 2,400 chunks across 6 rungs of this
#                             very leg; stable 4.35-4.43, so it is a corpus
#                             property, not a rung artifact.
#   embeddinggemma     ~4.2   no public tokenizer (gated); bisected Ollama's real
#                             truncation boundary instead -- embed(T[:m]) == embed(T)
#                             iff m >= cap -- over 4 long chunks x {512, 2048} ctx.
#                             Caps landed at 1797-2315 chars for a ~503-token
#                             content budget and 8143-9010 for ~2039. (Ollama's
#                             /api/ps independently reports context_length 2048.)
#
# The fallback is 4.2 rather than 3.5 so an unknown model errs toward the measured
# range instead of toward a number nothing supports. It is still only a fallback:
# per chunk this ratio spans 2.00-5.67 on one corpus, so where the length
# distribution sits ON the cap it misclassifies ~30% of chunks either way. Use
# _token_counter when the model has a reachable tokenizer.
_CHARS_PER_TOKEN_FALLBACK = 4.2
_MEASURED_CHARS_PER_TOKEN: Dict[str, float] = {
    "all-MiniLM-L6-v2": 4.38,
    "sentence-transformers/all-MiniLM-L6-v2": 4.38,
    "embeddinggemma:300m": 4.2,
}


def _chars_per_token(model: str) -> float:
    """The measured chars/token for ``model``, or the measured-range fallback."""
    return _MEASURED_CHARS_PER_TOKEN.get(model, _CHARS_PER_TOKEN_FALLBACK)


def _token_counter(model: str):
    """A ``text -> n_tokens`` function using the model's REAL tokenizer, or None.

    Exact token counts make the coverage question exact, and there is no reason to
    estimate what can be counted. Strictly offline: this is a diagnostic and must
    not reach for the network, so a model whose tokenizer is not already cached
    simply falls back to the char proxy.

    Ollama-hosted models have no importable tokenizer (embeddinggemma's is gated),
    which is why the char proxy has to survive at all.
    """
    name = model if "/" in model else f"sentence-transformers/{model}"
    try:
        from transformers import AutoTokenizer

        # local_files_only, not the HF_HUB_OFFLINE env var: the env var is read at
        # huggingface_hub import time, so setting it here is a no-op whenever
        # anything has already imported transformers -- which, in this harness, it
        # has. This argument is checked per call and cannot be outrun by import order.
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 -- any failure means "use the proxy"
        logger.info("no local tokenizer for %s (%s); coverage falls back to the char proxy", model, type(exc).__name__)
        return None

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return count


def density_leg() -> Leg:
    from benchmarks import beir_data
    from benchmarks.superdocs import build_synthetic_benchmark

    def _load():
        return build_synthetic_benchmark(
            beir_data.load("fiqa"),
            sections_per_doc=SECTIONS,
            passages_per_section=PASSAGES,
            seed=SEED,
            max_queries=QUERIES,
            mode="section",
            min_section_gold=MIN_QUERY_GOLD,
            min_query_gold=MIN_QUERY_GOLD,
            max_section_gold=GOLD,
        )

    key = f"density_fiqa_s{SECTIONS}p{PASSAGES}q{QUERIES}g{GOLD}seed{SEED}"
    return Leg(name="density_g4", cache_key=key, load=_load)


def _token_coverage(db, cap_tokens: int, count: Any, sample: int = 400) -> Dict[str, float]:
    """EXACT coverage, in tokens, over a strided sample of the chunk table.

    This is the quantity every ``coverage`` number in this project was trying to
    approximate. Where a real tokenizer exists there is no reason to approximate
    it: ``sum(min(tok, budget)) / sum(tok)``.

    ``cap_tokens`` is the context; the budget is two less, because the encoder
    spends two slots on [CLS]/[SEP] and an instruction-prefixed model spends
    several more. Sampling is strided rather than random so the number is stable
    across runs of the same index -- a diagnostic that moves when nothing changed
    is worse than no diagnostic.
    """
    with db.connection_pool.get_connection() as conn:
        rows = [r[0] for r in conn.execute("SELECT content FROM chunks ORDER BY id")]
    step = max(1, len(rows) // sample)
    texts = rows[::step][:sample]
    ntok = [count(t) for t in texts]
    budget = max(1, cap_tokens - 2)
    total = sum(ntok)
    return {
        "coverage_tokens": (sum(min(n, budget) for n in ntok) / total) if total else float("nan"),
        "frac_chunks_truncated_tokens": sum(n > budget for n in ntok) / len(ntok),
        "mean_chunk_tokens": total / len(ntok),
        "measured_chars_per_token": sum(len(t) for t in texts) / total if total else float("nan"),
        "token_sample": len(texts),
    }


def _straddle(db, encoder_cap_chars: float, band: float = 0.35) -> float:
    """Fraction of chunks within +-``band`` of the char cap -- the proxy's blind zone.

    A char cap can only classify a chunk when the chunk is far from it. Chars per
    token is not constant across chunks: on this leg it spans at least 2.1 to 5.3,
    so on the c=500 rung, where the length distribution sits ON the cap, NO single
    char threshold reproduces which vectors actually changed (best fit still
    misclassifies 31%). On c=1000, where chunks are far above the cap, a threshold
    separates them to 0.1%.

    This bounds ``frac_chunks_truncated`` specifically -- whether a given chunk is
    over the line. It is NOT the error bar on ``coverage``: when chunks sit far
    ABOVE the cap, straddle is ~0 and every chunk is correctly called truncated,
    yet coverage is still just cap/mean_len and so scales linearly with whatever
    constant was assumed. The two failure modes are different; this catches one.
    """
    lo, hi = int(encoder_cap_chars * (1 - band)), int(encoder_cap_chars * (1 + band))
    with db.connection_pool.get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, SUM(CASE WHEN LENGTH(content) BETWEEN ? AND ? THEN 1 ELSE 0 END) AS s FROM chunks",
            (lo, hi),
        ).fetchone()
    return (int(row["s"] or 0) / int(row["n"])) if row["n"] else float("nan")


def _corpus_shape(db, encoder_cap_chars: float) -> Dict[str, float]:
    """Measured chunk/section geometry of a built index, including COVERAGE.

    The *nominal* chunk_size overstates how much distinct text a chunk covers,
    because ``chunk_overlap=1`` is one sentence regardless of chunk size. Reading
    the real mean back out is what makes the reported r a measurement.

    ``coverage`` is the fraction of corpus text the encoder actually reads,
    ``sum(min(len, cap)) / sum(len)``. It is reported separately from anything
    derived from the MEAN because the mean is the wrong statistic for a
    truncation question and reporting it as one has already misled this project
    once: at c=3000 the mean-based ``r_effective`` said "barely truncated" (0.86)
    while **44.9%** of the corpus was being discarded. Chunk length here is
    strongly bimodal -- median 10,759 against a mean of 7,229, because CharChunker
    emits a short remainder fragment per document -- so the mean sits in a valley
    the distribution barely occupies. Coverage is the quantity the cliff tracks;
    prefer it to r_effective in any claim about truncation.

    CAVEAT (2026-08-06): everything here is a CHAR proxy for a TOKEN quantity, and
    its error is not uniform -- it concentrates exactly where the length
    distribution straddles the cap, which is the regime worth tuning. Prefer
    ``coverage_tokens`` when a tokenizer is available, and read ``straddle_frac``
    before trusting this one. Ground truth, where a paired build exists, is
    neither: it is how far the stored VECTOR moved (see the module docstring).
    """
    with db.connection_pool.get_connection() as conn:
        chunks = conn.execute(
            "SELECT COUNT(*) AS n, AVG(LENGTH(content)) AS mean, "
            "       SUM(LENGTH(content)) AS total, "
            "       SUM(MIN(LENGTH(content), ?)) AS encoded, "
            "       SUM(CASE WHEN LENGTH(content) > ? THEN 1 ELSE 0 END) AS n_truncated "
            "FROM chunks",
            (int(encoder_cap_chars), int(encoder_cap_chars)),
        ).fetchone()
        sections = conn.execute("SELECT COUNT(*) AS n FROM sections").fetchone()
    total = float(chunks["total"] or 0.0)
    return {
        "n_chunks": int(chunks["n"] or 0),
        "mean_chunk_chars": float(chunks["mean"] or 0.0),
        "n_sections": int(sections["n"] or 0),
        "total_chunk_chars": total,
        "coverage": (float(chunks["encoded"] or 0.0) / total) if total else float("nan"),
        "frac_chunks_truncated": (int(chunks["n_truncated"] or 0) / int(chunks["n"])) if chunks["n"] else float("nan"),
    }


def _r_values(mean_chunk_chars: float, window_chars: int, encoder_cap_chars: float) -> Dict[str, float]:
    """Nominal and effective granularity ratios from the measured chunk length."""
    nominal = window_chars / mean_chunk_chars if mean_chunk_chars else float("nan")
    encoded = min(mean_chunk_chars, encoder_cap_chars)
    return {
        "r_nominal": nominal,
        "r_effective": window_chars / encoded if encoded else float("nan"),
        "encoded_chunk_chars": encoded,
        "truncated": mean_chunk_chars > encoder_cap_chars,
    }


def _encoder_geometry(db, override: Optional[int] = None) -> Tuple[int, float, int]:
    """(context_tokens, encoder_cap_chars, window_chars) from a DB's live provider.

    Read off the opened database rather than a freshly constructed provider for
    two reasons. It is the object ``_span_embed`` will actually interrogate, so
    the window size reported here is the one the section vectors were built with,
    not a re-derivation that could drift. And constructing a provider here
    directly triggers ``transformers``' lazy ``trainer`` import, which resolves
    ``import datasets`` to ``benchmarks/datasets.py`` -- the script's own
    directory is ``sys.path[0]`` -- and dies.
    """
    # src/'s own sizing, imported rather than mirrored: a copy would silently
    # stop matching the moment _WINDOW_CHARS_PER_TOKEN changed.
    from localvectordb.database._span_embed import (
        _DEFAULT_WINDOW_CHARS,
        _provider_context_tokens,
        _window_chars_for,
    )

    provider = db.embedding_provider
    tokens = _provider_context_tokens(provider) or override
    if not tokens:
        raise SystemExit(
            f"{provider!r} reports no context window; the r rule is undefined without it "
            "(and _span_embed falls back to a fixed 24,000-char window). Pass --context-tokens."
        )
    if _provider_context_tokens(provider) is None:
        # NOT cosmetic. src/ sizes rawspan windows from the same missing value and
        # silently uses 24,000 chars, which overflows any sub-6,860-token encoder
        # (§6.34/4b). Harmless for a centroid-only sweep -- centroids pool chunk
        # vectors and never window -- so the override fixes the REPORTED geometry
        # only. It does not fix what a rawspan build would actually do.
        logger.warning(
            "provider reports no context window; using --context-tokens=%d for the r rule. "
            "src/ would window rawspan sections at %d chars regardless -- do not trust a "
            "rawspan arm built this way on long sections.",
            override,
            _DEFAULT_WINDOW_CHARS,
        )
    # The window uses the conservative 3.0; the cap on how much of a CHUNK the
    # model actually encodes is the same context measured in ordinary text, 3.5.
    # When the provider reports nothing, _window_chars_for returns the 24,000-char
    # default; the override describes the encoder we actually have, so report that.
    window = (
        _window_chars_for(provider) if _provider_context_tokens(provider) else int(tokens * _WINDOW_CHARS_PER_TOKEN)
    )
    return tokens, tokens * _chars_per_token(getattr(provider, "model", "")), window


def _embedding_config(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Transport-only overrides; see build_db's allowlist. Never content-affecting."""
    cfg: Dict[str, Any] = {}
    if args.embed_concurrency:
        cfg["max_concurrent_requests"] = args.embed_concurrency
    if args.embed_timeout:
        cfg["timeout"] = args.embed_timeout
    return cfg or None


def run_rung(leg: Leg, bench, chunk_size: int, args: argparse.Namespace) -> Dict[str, Any]:
    # None means "library default", which is the key every already-built DB used.
    key_size = None if chunk_size == LIBRARY_DEFAULT_CHUNK_SIZE else chunk_size

    dbs = {
        s: build_db(
            bench,
            leg=leg,
            provider=args.embedding_provider,
            model=args.embedding_model,
            strategy=s,
            rebuild=args.rebuild,
            chunk_size=key_size,
            num_ctx=args.num_ctx,
            embedding_config=_embedding_config(args),
        )
        for s in args.strategies
    }
    for strategy, db in dbs.items():
        if db.count() != len(bench.corpus):
            raise RuntimeError(
                f"DB[{strategy}] c={chunk_size} holds {db.count()} documents, "
                f"benchmark has {len(bench.corpus)}. Stale cached index -- re-run with --rebuild."
            )

    # Geometry is a property of the corpus and the encoder, not of the section
    # strategy, so any built arm reports it identically.
    probe = next(iter(dbs.values()))
    tokens, encoder_cap_chars, window_chars = _encoder_geometry(probe, args.context_tokens)
    shape = _corpus_shape(probe, encoder_cap_chars)
    rung: Dict[str, Any] = {
        "chunk_size": chunk_size,
        "num_ctx": args.num_ctx,
        "encoder_context_tokens": tokens,
        "window_chars": window_chars,
        "encoder_cap_chars": encoder_cap_chars,
        "chars_per_token": _chars_per_token(getattr(probe.embedding_provider, "model", "")),
        "straddle_frac": _straddle(probe, encoder_cap_chars),
        **shape,
        **_r_values(shape["mean_chunk_chars"], window_chars, encoder_cap_chars),
    }
    counter = _token_counter(getattr(probe.embedding_provider, "model", ""))
    if counter is not None:
        rung.update(_token_coverage(probe, tokens, counter))
    logger.info(
        "[c=%d] %d chunks, mean %.0f chars, COVERAGE %.1f%% chars / %s tokens "
        "(%.0f%% truncated at %.0f chars; straddle %.0f%%)",
        chunk_size,
        rung["n_chunks"],
        rung["mean_chunk_chars"],
        100.0 * rung["coverage"],
        f"{100.0 * rung['coverage_tokens']:.1f}%" if "coverage_tokens" in rung else "n/a",
        100.0 * rung["frac_chunks_truncated"],
        encoder_cap_chars,
        100.0 * rung["straddle_frac"],
    )

    for cfg in build_configs():
        if cfg.strategy not in dbs:
            continue
        scores = run_config(dbs[cfg.strategy], bench, cfg, search_type=args.search_type)
        rung[cfg.label] = scores
        logger.info(
            "[c=%d] %-24s ndcg@10=%.4f  sec=%s",
            chunk_size,
            cfg.label,
            scores["ndcg@10"],
            f"{scores['ndcg@10_sections']:.4f}" if "ndcg@10_sections" in scores else "-",
        )
    for db in dbs.values():
        db.close()
    return rung


# §6.32 was measured on this encoder alone, so the anchor is only meaningful there.
ANCHOR_MODEL = "all-MiniLM-L6-v2"


def _check_anchor(rung: Dict[str, Any]) -> List[str]:
    """The cached c=500 rung must reproduce §6.32, or the whole sweep is suspect."""
    published = {
        ("sections · rawspan", "ndcg@10"): 0.4536,
        ("sections · centroid", "ndcg@10"): 0.3537,
        ("sections · rawspan", "ndcg@10_sections"): 0.4426,
        ("sections · centroid", "ndcg@10_sections"): 0.3308,
    }
    problems = []
    for (label, metric), expected in published.items():
        got = rung.get(label, {}).get(metric)
        if got is None:
            problems.append(f"{label}/{metric} missing")
        elif abs(got - expected) > 0.0005:
            problems.append(f"{label}/{metric} {got:.4f} != published {expected:.4f}")
    return problems


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-provider", default=EVAL_EMBEDDING_PROVIDER)
    p.add_argument("--embedding-model", default=EVAL_EMBEDDING_MODEL)
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=list(CHUNK_SIZES))
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["rawspan", "centroid"],
        choices=["rawspan", "centroid"],
        help="which section_vector_strategy arms to build. ``centroid`` alone is enough for the "
        "plain chunk-retrieval curve -- HierConfig('chunks', ...) reads the centroid DB, and a "
        "centroid section vector is POOLED from chunk vectors, so that build embeds no section "
        "windows at all. ``rawspan`` is the expensive arm and buys only the section-level "
        "rawspan-vs-centroid gap: this leg's sections average ~26k chars against a 6k-char window "
        "on egemma, so _span_embed splits each into ~5 windows and the section half of a build "
        "costs MORE than the chunk half (~2.6M vs ~1.9M tokens at c=1000).",
    )
    p.add_argument(
        "--embed-concurrency",
        type=int,
        default=None,
        help="override the provider's max_concurrent_requests (transport only). Worth ~10-20%% against "
        "ollama, NOT the 2x an isolated one-text-per-request benchmark suggests: the provider already "
        "batches 64 texts per call, which amortizes almost all of the per-request overhead.",
    )
    p.add_argument(
        "--embed-timeout",
        type=int,
        default=None,
        help="per-request timeout in seconds (transport only; OllamaEmbeddings defaults to 300). "
        "``max_batch_size`` is a fixed COUNT of 64, so a batch's token volume scales with "
        "chunk_size: 64x500 tokens is comfortable, 64x1750 is ~112k tokens in one request and at "
        "8-way concurrency it cannot finish inside 300s. Symptom is an EMPTY error message every "
        "300s exactly (httpx.ReadTimeout stringifies to ''), four of which kill the build. Lower "
        "--embed-concurrency as well: fewer requests in flight finish faster than more that time out.",
    )
    p.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="CONTENT-AFFECTING (cache-keyed as __ctx{n}): the context window to request from Ollama, "
        "which is also what it truncates at. Lowering this on a fixed model is the cleanest available "
        "test of the coverage hypothesis -- identical weights, identical corpus, identical chunking; "
        "only how much of each chunk the encoder READS changes. A third encoder confounds coverage "
        "with model quality, training data and dimension, which is the confound that made the first "
        "ladder ambiguous in the first place. Setting this also makes _provider_context_tokens report "
        "a real value, so --context-tokens is unnecessary and _span_embed stops using its 24,000-char "
        "fallback. OllamaEmbeddings moves num_batch with it (llama.cpp's n_batch is the true ceiling).",
    )
    p.add_argument(
        "--context-tokens",
        type=int,
        default=None,
        help="encoder context window to use for the r rule when the provider does not report one. "
        "OllamaEmbeddings sets neither num_ctx nor max_input_tokens by default, so every "
        "ollama-backed provider reports None (see §6.34/4b). This fixes the REPORTED geometry only; "
        "src/ still sizes rawspan windows at 24,000 chars in that case.",
    )
    p.add_argument(
        "--search-type",
        default="hybrid",
        choices=["hybrid", "vector", "keyword"],
        help="ALWAYS set this deliberately. The CHUNKS curve this sweep draws was measured at the "
        "'hybrid' default, and BM25 reads the whole chunk no matter what the encoder truncates -- so "
        "the keyword leg props the score up exactly where coverage collapses, MASKING the very cliff "
        "the sweep exists to locate. Use 'vector' for any claim about encoder context or coverage; "
        "'hybrid' only to describe what a user of the shipped defaults actually gets.",
    )
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--out", default=str(_ROOT / "benchmarks" / "results" / "chunk_size_sweep.json"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    # Load the encoder BEFORE any corpus. The order is load-bearing: loading a
    # BEIR corpus first puts a ``datasets`` module in ``sys.modules`` that shadows
    # HuggingFace's (``benchmarks/datasets.py``, reachable because the script's own
    # directory is ``sys.path[0]``), and ``transformers``' lazy import of it then
    # fails -- surfacing as ``LocalVectorDB``'s bare "model is not available".
    from benchmarks.eval_retrieval import preflight_embedding_model

    preflight_embedding_model(args.embedding_provider, args.embedding_model)

    leg = density_leg()
    bench = leg.load()
    logger.info("[%s] %d docs, %d queries", leg.name, len(bench.corpus), len(bench.queries))

    rungs: Dict[str, Any] = {}
    anchor_problems: List[str] = []
    for chunk_size in args.chunk_sizes:
        rung = run_rung(leg, bench, chunk_size, args)
        rungs[str(chunk_size)] = rung
        if (
            chunk_size == LIBRARY_DEFAULT_CHUNK_SIZE
            and not args.rebuild
            and args.embedding_model == ANCHOR_MODEL
            # The published anchors are HYBRID scores (S6.36). Checking them
            # against a vector run would report a mismatch that is really just
            # the missing BM25 leg, turning a correct run into a false alarm.
            and args.search_type == "hybrid"
        ):
            anchor_problems = _check_anchor(rung)
            if anchor_problems:
                logger.error("ANCHOR MISMATCH at c=500: %s", "; ".join(anchor_problems))
            else:
                logger.info("anchor ok: c=500 reproduces the published §6.32 rung exactly")

    ordered = sorted(rungs.values(), key=lambda r: r["chunk_size"])
    have_rawspan = any("sections · rawspan" in r for r in ordered)

    def _cell(rung: Dict[str, Any], label: str, metric: str) -> float:
        return rung.get(label, {}).get(metric, float("nan"))

    have_tokens = any("coverage_tokens" in r for r in ordered)
    print(
        f"\n{'chunk':>6} {'chunks':>8} {'mean ch':>8} {'cover':>7} "
        f"{'covTOK':>7} {'strad':>6} {'r_nom':>6} {'CHUNKS':>9}",
        end="",
    )
    print(f" {'raw doc':>9} {'cen doc':>9} {'raw sec':>9} {'cen sec':>9} {'sec gap':>9}" if have_rawspan else "")
    for r in ordered:
        cov_tok = f"{100.0 * r['coverage_tokens']:>6.1f}%" if "coverage_tokens" in r else f"{'--':>7}"
        row = (
            f"{r['chunk_size']:>6} {r['n_chunks']:>8} {r['mean_chunk_chars']:>8.0f} "
            f"{100.0 * r['coverage']:>6.1f}% {cov_tok} {100.0 * r.get('straddle_frac', float('nan')):>5.0f}% "
            f"{r['r_nominal']:>6.2f} {_cell(r, 'chunks', 'ndcg@10'):>9.4f}"
        )
        if have_rawspan:
            raw_s = _cell(r, "sections · rawspan", "ndcg@10_sections")
            cen_s = _cell(r, "sections · centroid", "ndcg@10_sections")
            row += (
                f" {_cell(r, 'sections · rawspan', 'ndcg@10'):>9.4f}"
                f" {_cell(r, 'sections · centroid', 'ndcg@10'):>9.4f}"
                f" {raw_s:>9.4f} {cen_s:>9.4f} {raw_s - cen_s:>+9.4f}"
            )
        print(row)
    print("\nCHUNKS is plain search_level='chunks' -- the curve this sweep exists to draw.")
    print("'cover' is sum(min(len,cap))/sum(len) in CHARS -- a proxy, and a biased one.")
    if have_tokens:
        print("'covTOK' is the same quantity counted in TOKENS by the real tokenizer. Prefer it.")
    print("'strad' is the share of chunks sitting ON the cap (+-35%), where 'truncated?' is a coin-flip.")
    if have_rawspan:
        print("'sec gap' > 0 means rawspan wins at the section level; the rule predicts it flips as r crosses 1.")
    else:
        print("Section arms not built (--strategies centroid); 'sections · centroid' is scored, rawspan is not.")
    if anchor_problems:
        print("ANCHOR MISMATCH -- cached c=500 rung does not reproduce §6.32: " + "; ".join(anchor_problems))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(),
                "model": args.embedding_model,
                "search_type": args.search_type,
                "leg": f"fiqa-superdocs s{SECTIONS}p{PASSAGES} gold{GOLD} q{QUERIES} seed{SEED}",
                "anchor_problems": anchor_problems,
                "rungs": rungs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 1 if anchor_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
