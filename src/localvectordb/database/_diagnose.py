"""Corpus diagnostics: which retrieval regime is this database in?

The retrieval study behind this module ended on one product conclusion: six of
the knobs this library ships (``chunk_size``, ``search_level``,
``section_weight``, ``section_vector_strategy``, ``vector_weight``,
``frequency_bias``) have **no defensible global default** -- their argmaxes are
properties of the user's corpus, and they disagree across corpora by more than
any effect worth tuning. Everything a user needs to know which regime they are
in is derivable from a built index in seconds. This module derives it.

``diagnose()`` reports; ``_maybe_warn_truncation()`` warns. The warning exists
for the one case where the library *knows* something is wrong and used to say
nothing: text past the encoder's context window is silently discarded before it
enters any vector. At the shipped ``chunk_size=500`` on a 256-token encoder,
49% of a corpus never reached the index -- no error, no warning.

Two measured rules govern how both are built:

* **Count tokens; never trust a chars/token constant.** Per chunk the ratio
  spans 2.00-5.67 within a single corpus, so a character cap misclassifies
  ~28-31% of chunks exactly when the length distribution sits on the cap -- the
  regime a warning fires in. Where the encoder's own tokenizer is importable we
  use it; where it is not (Ollama models have none), counts are labelled
  **estimated** rather than printed as confident percentages -- our own
  published coverage figures were understated by up to 16 points by a constant.
* **Mild truncation is free; warn only on substantial loss.** Paired builds
  showed 46% of chunks truncated with nDCG unchanged to four decimals; damage
  appeared once measured coverage fell to ~65%, where it cost 0.109 nDCG@10 --
  four times the entire chunk-size plateau. A warning that fires on any
  truncation cries wolf and would be ignored by the time it mattered.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from localvectordb.database._span_embed import _provider_context_tokens
from localvectordb.database.base import LocalVectorDBBase

logger = logging.getLogger(__name__)

# Below this measured coverage the ingest warning fires. Calibrated between the
# two measured anchors above: 46%-of-chunks-truncated (tail clipping, coverage
# still high) moved nothing, and 65% coverage cost 0.109 nDCG@10. 0.80 leaves
# margin above the cliff without firing on the free region.
_COVERAGE_WARN_THRESHOLD = 0.80

# The ingest-time check is skipped below this many chunks: a percentage over a
# handful of chunks is noise, and a tiny corpus is cheap to re-ingest anyway.
_MIN_CHUNKS_FOR_WARNING = 32

# Estimated chars/token when even tiktoken is unavailable. Measured: 4.38 with
# MiniLM's tokenizer, ~4.2 for embeddinggemma (by bisecting Ollama's real
# truncation boundary). Any such constant is per-chunk unreliable -- which is
# why every path using it is labelled estimated.
_ESTIMATE_CHARS_PER_TOKEN = 4.2

# Cap on how many chunks / sections diagnose() re-tokenizes with a real
# tokenizer. Sampling is evenly strided over the id order, so it is
# deterministic across runs on an unchanged database.
_DEFAULT_SAMPLE = 5000
_SECTION_SAMPLE_CAP = 1000

# Above this share of sections exceeding one encoder window, the summary
# recommends the centroid strategy: rawspan (windowed + mean-pooled) loses
# 0.25-0.36 nDCG to the centroid on long sections (median >8k tokens) on every
# encoder tested, while winning on short ones.
_LONG_SECTION_NOTE_THRESHOLD = 0.25


def _coverage_from_counts(token_counts: Sequence[int], cap: int) -> Tuple[float, float]:
    """(coverage, truncated_share) for per-item token counts against a context cap.

    Coverage is sum(min(t, cap)) / sum(t) -- the share of ingested text the
    encoder actually reads -- NOT a mean-based estimate: chunk lengths are
    bimodal (remainder fragments drag the mean down), and at one measured rung
    the mean said "barely truncated" while 44.9% of text was being discarded.
    """
    total = sum(token_counts)
    if total <= 0:
        return 1.0, 0.0
    kept = sum(min(t, cap) for t in token_counts)
    truncated = sum(1 for t in token_counts if t > cap)
    return kept / total, truncated / len(token_counts)


def _resolve_token_counter(provider: Any) -> Tuple[Callable[[str], int], bool, str]:
    """(counter, exact, description) for counting tokens the way the encoder does.

    ``exact`` is True only when the counter is the encoder's own tokenizer.
    tiktoken's ``cl100k_base`` is exact for OpenAI embedding models and an
    *estimate* for everything else -- a wrong-model tokenizer tracks real counts
    far better than a chars/token constant, but it is still not the encoder's,
    so it must not be reported as measured.
    """
    provider_name = getattr(provider, "provider_name", "")

    if provider_name == "sentence_transformers":
        try:
            tokenizer = provider._load_model().tokenizer
            return (lambda t: len(tokenizer.encode(t)), True, "the model's own tokenizer")
        except Exception as e:  # pragma: no cover - model load can fail offline
            logger.debug(f"Could not load sentence-transformers tokenizer: {e}")

    if provider_name == "huggingface_local":
        try:
            tokenizer = getattr(provider, "_tokenizer", None)
            if tokenizer is None:
                tokenizer, _ = provider._load_model()
            return (lambda t: len(tokenizer.encode(t)), True, "the model's own tokenizer")
        except Exception as e:  # pragma: no cover
            logger.debug(f"Could not load HuggingFace tokenizer: {e}")

    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        if provider_name == "openai":
            return (lambda t: len(encoder.encode(t)), True, "tiktoken cl100k_base (the model's tokenizer)")
        return (
            lambda t: len(encoder.encode(t)),
            False,
            "estimated with tiktoken cl100k_base -- the encoder's own tokenizer is not importable, "
            "so treat percentages as approximate",
        )
    except Exception as e:  # pragma: no cover - tiktoken is a hard dependency
        logger.debug(f"tiktoken unavailable: {e}")

    return (
        lambda t: max(1, int(len(t) / _ESTIMATE_CHARS_PER_TOKEN)),
        False,
        f"estimated at {_ESTIMATE_CHARS_PER_TOKEN} chars/token -- no tokenizer importable; "
        "the true ratio spans 2.0-5.7 per chunk, so treat percentages as rough",
    )


def _median(values: Sequence[int]) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    return int(ordered[len(ordered) // 2])


def _strided_sample(ids: Sequence[Any], cap: int) -> List[Any]:
    """Deterministic even sample: every len/cap-th id in id order."""
    if len(ids) <= cap:
        return list(ids)
    stride = -(-len(ids) // cap)  # ceil
    return list(ids[::stride])


@dataclass
class DiagnoseReport:
    """What ``diagnose()`` measured, and which regime it puts the corpus in."""

    database: str = ""
    documents: int = 0
    chunks: int = 0
    sections: int = 0
    embedding_provider: str = ""
    embedding_model: str = ""

    #: The encoder's context window in tokens, or None when no provider reports one.
    context_tokens: Optional[int] = None
    #: True when token counts came from the encoder's own tokenizer.
    tokens_exact: bool = False
    token_source: str = ""

    #: sum(min(tokens, context)) / sum(tokens) over the measured chunks; None
    #: when the context is unknown (there is no cap to measure against).
    chunk_coverage: Optional[float] = None
    truncated_chunk_share: Optional[float] = None
    chunks_measured: int = 0
    median_chunk_tokens: Optional[int] = None

    #: Share of sections whose text exceeds one encoder window. These are not
    #: truncated -- rawspan vectors are windowed and mean-pooled -- but pooled
    #: vectors degrade as the span grows, which is a regime, not a defect.
    sections_over_context_share: Optional[float] = None
    median_section_tokens: Optional[int] = None
    sections_measured: int = 0

    #: Share of sections owning no chunk. Those sections cannot be returned by
    #: chunk->section roll-up at any k, so 1 - share is a hard recall ceiling
    #: for return_type="sections" on the default search path.
    chunkless_section_share: Optional[float] = None

    mean_chunks_per_document: Optional[float] = None
    median_chunks_per_document: Optional[int] = None
    mean_chunks_per_section: Optional[float] = None

    #: Per FTS table: "ok", "missing", "disabled", or "stale (n vs m)".
    fts_status: Dict[str, str] = field(default_factory=dict)

    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.warnings

    @property
    def summary(self) -> str:
        lines = [
            f"Diagnosis of '{self.database}' -- "
            f"{self.documents:,} documents, {self.chunks:,} chunks, {self.sections:,} sections",
            f"Encoder: {self.embedding_model} ({self.embedding_provider}), "
            + (f"context {self.context_tokens:,} tokens" if self.context_tokens else "context unknown"),
            f"Token counts: {'exact -- ' if self.tokens_exact else ''}{self.token_source}",
            "",
        ]

        if self.chunk_coverage is not None:
            lost = 1.0 - self.chunk_coverage
            lines.append(
                f"Encoder coverage of chunk text: {self.chunk_coverage:.1%} "
                f"({lost:.1%} of ingested text lies past the context window and never "
                f"enters any vector); {self.truncated_chunk_share:.1%} of chunks are truncated. "
                f"[{self.chunks_measured:,} chunks measured"
                + (f", median {self.median_chunk_tokens:,} tokens]" if self.median_chunk_tokens else "]")
            )
        else:
            lines.append(
                "Encoder coverage of chunk text: not measurable -- the provider reports no "
                "context window, so text past it cannot be detected. If the model's context "
                "is known, compare it with the median chunk length"
                + (f" ({self.median_chunk_tokens:,} tokens, {self.token_source})." if self.median_chunk_tokens else ".")
            )

        if self.sections:
            if self.sections_over_context_share is not None:
                lines.append(
                    f"Sections: median {self.median_section_tokens or 0:,} tokens; "
                    f"{self.sections_over_context_share:.1%} exceed one encoder window "
                    f"[{self.sections_measured:,} measured]. Long sections are windowed and "
                    "mean-pooled, never truncated -- but pooled (rawspan) vectors degrade as "
                    "spans grow (measured -0.25 to -0.36 nDCG at >8k tokens), so a mostly-long "
                    "corpus wants section_vector_strategy='centroid'."
                )
            if self.chunkless_section_share is not None:
                reachable = 1.0 - self.chunkless_section_share
                lines.append(
                    f"Section reachability: {self.chunkless_section_share:.1%} of sections own no "
                    f"chunk, so chunk->section roll-up (return_type='sections') can reach at most "
                    f"{reachable:.1%} of them at any k. This happens when chunk_size exceeds the "
                    "median section length -- a chunk's midpoint credits only one of the sections "
                    "it spans."
                )

        fanout = []
        if self.mean_chunks_per_document is not None:
            fanout.append(
                f"{self.mean_chunks_per_document:.1f} chunks/document"
                + (f" (median {self.median_chunks_per_document})" if self.median_chunks_per_document else "")
            )
        if self.mean_chunks_per_section is not None:
            fanout.append(f"{self.mean_chunks_per_section:.1f} chunks/section (over sections owning chunks)")
        if fanout:
            lines.append(
                "Fanout: " + ", ".join(fanout) + ". At fanout ~1 every document-scoring "
                "aggregator is a no-op -- there is nothing to aggregate."
            )

        if self.fts_status:
            lines.append("Keyword legs: " + ", ".join(f"{t} {s}" for t, s in sorted(self.fts_status.items())) + ".")

        for note in self.notes:
            lines.append(f"Note: {note}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)


class DiagnoseMixin(LocalVectorDBBase, ABC):
    """Mixed into LocalVectorDB. Read-only: diagnose never mutates the database."""

    # Declared on the composed class (LocalVectorDBCore), not on LocalVectorDBBase.
    name: str
    # Warn-once state; read via getattr with defaults, so no __init__ is needed.
    _truncation_warned: bool
    _truncation_checked_at: int

    def diagnose(self, sample: int = _DEFAULT_SAMPLE) -> DiagnoseReport:
        """Measure which retrieval regime this corpus is in.

        Reports encoder coverage of chunk text (text past the context window
        never enters any vector), section length against the encoder window,
        the chunk->section reachability ceiling, chunks-per-document /
        chunks-per-section fanout, and the health of each keyword (FTS) leg.

        Token counts use the encoder's own tokenizer where one is importable
        (sentence-transformers, local HuggingFace, OpenAI); otherwise they are
        estimates and the report says so rather than printing a confident
        percentage.

        Parameters
        ----------
        sample : int
            Maximum number of chunks to re-tokenize when an exact tokenizer is
            available (sections are capped lower). Sampling is evenly strided
            over id order, so repeat runs on an unchanged database measure the
            same rows. With no exact tokenizer, chunk coverage uses the stored
            per-chunk token counts over the full corpus instead.

        Returns
        -------
        DiagnoseReport
            Structured results; ``report.summary`` renders them for humans.
        """
        provider = self.embedding_provider
        report = DiagnoseReport(
            database=str(self.name),
            embedding_provider=getattr(provider, "provider_name", "?"),
            embedding_model=getattr(provider, "model", "?"),
            context_tokens=_provider_context_tokens(provider),
        )
        counter, exact, source = _resolve_token_counter(provider)
        report.tokens_exact = exact
        report.token_source = source

        with self.connection_pool.get_connection() as conn:
            report.documents = int(conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"])
            report.chunks = int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
            report.sections = int(conn.execute("SELECT COUNT(*) AS n FROM sections").fetchone()["n"])

            self._measure_chunks(conn, report, counter, exact, sample)
            if report.sections:
                self._measure_sections(conn, report, counter)
            self._measure_fanout(conn, report)
            self._check_fts(conn, report)

        self._assemble_warnings(report)
        return report

    async def diagnose_async(self, sample: int = _DEFAULT_SAMPLE) -> DiagnoseReport:
        """Async variant of :meth:`diagnose`; same measurement, same report."""
        return self.diagnose(sample=sample)

    # -----------------
    # Measurement legs
    # -----------------
    def _measure_chunks(
        self, conn: Any, report: DiagnoseReport, counter: Callable[[str], int], exact: bool, sample: int
    ) -> None:
        if not report.chunks:
            return
        cap = report.context_tokens

        if exact:
            # Re-tokenize a strided sample with the encoder's own tokenizer.
            # The stored chunks.tokens column is cl100k and would silently
            # reintroduce the wrong-tokenizer error this module exists to avoid.
            ids = [row["id"] for row in conn.execute("SELECT id FROM chunks ORDER BY id")]
            chosen = _strided_sample(ids, max(1, sample))
            counts: List[int] = []
            for i in range(0, len(chosen), 500):
                batch = chosen[i : i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(f"SELECT content FROM chunks WHERE id IN ({placeholders})", batch).fetchall()
                counts.extend(counter(row["content"] or "") for row in rows)
            report.chunks_measured = len(counts)
            report.median_chunk_tokens = _median(counts)
            if cap:
                report.chunk_coverage, report.truncated_chunk_share = _coverage_from_counts(counts, cap)
        else:
            # No exact tokenizer: the stored per-chunk counts (cl100k, written at
            # ingest) cover the full corpus for free and beat any chars/token
            # constant. Still an estimate, and reported as one.
            report.chunks_measured = report.chunks
            half = report.chunks // 2
            row = conn.execute("SELECT tokens FROM chunks ORDER BY tokens LIMIT 1 OFFSET ?", (half,)).fetchone()
            report.median_chunk_tokens = int(row["tokens"]) if row else None
            if cap:
                agg = conn.execute(
                    "SELECT SUM(MIN(tokens, ?)) AS kept, SUM(tokens) AS total, "
                    "SUM(CASE WHEN tokens > ? THEN 1 ELSE 0 END) AS truncated FROM chunks",
                    (cap, cap),
                ).fetchone()
                total = agg["total"] or 0
                if total:
                    report.chunk_coverage = agg["kept"] / total
                    report.truncated_chunk_share = agg["truncated"] / report.chunks

    def _measure_sections(self, conn: Any, report: DiagnoseReport, counter: Callable[[str], int]) -> None:
        ids = [row["id"] for row in conn.execute("SELECT id FROM sections ORDER BY id")]
        chosen = _strided_sample(ids, _SECTION_SAMPLE_CAP)
        counts: List[int] = []
        for i in range(0, len(chosen), 200):
            batch = chosen[i : i + 200]
            placeholders = ",".join("?" * len(batch))
            # Sections store no text; a section is a span of its parent document.
            rows = conn.execute(
                f"SELECT substr(d.content, s.start_pos + 1, s.end_pos - s.start_pos) AS text "
                f"FROM sections s JOIN documents d ON d.id = s.document_id WHERE s.id IN ({placeholders})",
                batch,
            ).fetchall()
            counts.extend(counter(row["text"] or "") for row in rows)
        report.sections_measured = len(counts)
        report.median_section_tokens = _median(counts)
        if report.context_tokens and counts:
            over = sum(1 for t in counts if t > report.context_tokens)
            report.sections_over_context_share = over / len(counts)

        owned = int(
            conn.execute("SELECT COUNT(DISTINCT section_id) AS n FROM chunks WHERE section_id IS NOT NULL").fetchone()[
                "n"
            ]
        )
        report.chunkless_section_share = 1.0 - (owned / report.sections)

    def _measure_fanout(self, conn: Any, report: DiagnoseReport) -> None:
        if report.documents and report.chunks:
            per_doc = [int(row["c"]) for row in conn.execute("SELECT COUNT(*) AS c FROM chunks GROUP BY document_id")]
            if per_doc:
                report.mean_chunks_per_document = sum(per_doc) / len(per_doc)
                report.median_chunks_per_document = _median(per_doc)
        if report.sections:
            row = conn.execute(
                "SELECT COUNT(*) AS chunks, COUNT(DISTINCT section_id) AS sections "
                "FROM chunks WHERE section_id IS NOT NULL"
            ).fetchone()
            if row and row["sections"]:
                report.mean_chunks_per_section = row["chunks"] / row["sections"]

    def _check_fts(self, conn: Any, report: DiagnoseReport) -> None:
        if not self.fts_enabled:
            report.fts_status = {"fts": "disabled"}
            return
        checks = [("chunks_fts", report.chunks), ("documents_fts", report.documents)]
        if report.sections:
            checks.append(("sections_fts", report.sections))
        for table, expected in checks:
            try:
                n = int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])  # nosec B608
            except Exception:
                report.fts_status[table] = "missing"
                continue
            report.fts_status[table] = "ok" if n == expected else f"stale ({n:,} rows vs {expected:,})"

    def _assemble_warnings(self, report: DiagnoseReport) -> None:
        if report.chunk_coverage is not None and report.chunk_coverage < _COVERAGE_WARN_THRESHOLD:
            report.warnings.append(
                _coverage_warning_text(report.chunk_coverage, report.context_tokens or 0, report.tokens_exact)
            )
        stale = [t for t, s in report.fts_status.items() if s not in ("ok", "disabled")]
        if stale:
            report.warnings.append(
                f"Keyword (FTS) tables out of step with their base tables: {', '.join(stale)}. "
                "The keyword leg is worth +0.08 to +0.13 nDCG@10 where present -- more than any "
                "vector-side knob. Reopening the database backfills sections_fts; for the others, "
                "rebuild via repair()."
            )
        if report.chunkless_section_share is not None and report.chunkless_section_share > _LONG_SECTION_NOTE_THRESHOLD:
            report.warnings.append(
                f"{report.chunkless_section_share:.1%} of sections own no chunk and cannot be "
                f"returned by return_type='sections' roll-up at any k (hard recall ceiling "
                f"{1.0 - report.chunkless_section_share:.1%}). Reduce chunk_size below the median "
                "section length, or search sections directly (search_level='sections')."
            )

    # -------------------------
    # Ingest-time coverage check
    # -------------------------
    def _maybe_warn_truncation(self) -> None:
        """Warn once when measured coverage says the encoder is discarding the corpus.

        Called from ``_save_internal`` -- the one chokepoint every mutating path
        crosses -- so it fires after real ingests without instrumenting eight
        call sites. Uses the stored per-chunk token counts, so the check is one
        SQL aggregate: full-corpus, no model, no tokenization. Those counts are
        cl100k (an estimate for non-OpenAI encoders), which is fine for a
        trigger whose threshold sits 15 points above the measured damage cliff;
        the warning says it is estimating and points at diagnose() for exact
        numbers.

        Never raises: a diagnostic must not be able to break a save.
        """
        if getattr(self, "_truncation_warned", False):
            return
        try:
            with self.connection_pool.get_connection() as conn:
                n = int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
                # Re-check only when the corpus has doubled since the last look,
                # so the full-table aggregate runs O(log n) times, not per save.
                last = getattr(self, "_truncation_checked_at", 0)
                if n < _MIN_CHUNKS_FOR_WARNING or n < 2 * last:
                    return
                self._truncation_checked_at = n
                # Resolve the context only after the cheap guards pass: for a
                # sentence-transformers provider the property lazily loads the
                # model, which a delete-only session should never trigger.
                cap = _provider_context_tokens(self.embedding_provider)
                if not cap:
                    return
                agg = conn.execute(
                    "SELECT SUM(MIN(tokens, ?)) AS kept, SUM(tokens) AS total FROM chunks", (cap,)
                ).fetchone()
            total = agg["total"] or 0
            if not total:
                return
            coverage = agg["kept"] / total
            if coverage >= _COVERAGE_WARN_THRESHOLD:
                return
            self._truncation_warned = True
            exact = getattr(self.embedding_provider, "provider_name", "") == "openai"
            warnings.warn(
                _coverage_warning_text(coverage, cap, exact),
                UserWarning,
                stacklevel=2,
            )
        except Exception as e:
            logger.debug(f"Truncation check skipped: {e}")


def _coverage_warning_text(coverage: float, cap: int, exact: bool) -> str:
    qualifier = "" if exact else " (estimated -- run db.diagnose() for numbers from the encoder's own tokenizer)"
    return (
        f"~{1.0 - coverage:.0%} of ingested text lies past the embedding model's "
        f"{cap:,}-token context window{qualifier} and never enters any vector. "
        f"At a measured 65% coverage this cost 0.109 nDCG@10 -- four times the entire "
        f"chunk-size tuning range. Reduce chunk_size so chunks fit the context "
        f"(undershooting is nearly free), or use an encoder with a longer context."
    )
