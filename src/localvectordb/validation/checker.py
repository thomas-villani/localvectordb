"""FactChecker -- orchestrates claim extraction, source retrieval, and polarity classification."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

from .annotator import annotate_response
from .claims import extract_claims
from .llm import LLMProvider, detect_provider
from .polarity import PolarityResult, classify_polarity
from .result import ClaimResult, FactCheckResult, Polarity

if TYPE_CHECKING:
    from localvectordb.database import LocalVectorDB


class FactChecker:
    """Fact-check LLM-generated text against one or more LocalVectorDB instances.

    Parameters
    ----------
    databases:
        One or more :class:`LocalVectorDB` instances to search for evidence.
    llm:
        An Anthropic, OpenAI, or Google GenAI client, or any object implementing
        the :class:`LLMProvider` protocol.
    model:
        Model name passed to the LLM provider for claim extraction and polarity
        classification.  Defaults are provider-specific (Haiku for Anthropic,
        gpt-4o-mini for OpenAI, gemini-2.0-flash for Gemini).
    similarity_threshold:
        Minimum similarity score for a retrieved chunk to be considered relevant.
    min_grounding_score:
        Minimum polarity confidence for a claim to count as grounded.
    search_type:
        Search mode used when querying the databases (``"vector"``,
        ``"keyword"``, or ``"hybrid"``).
    top_k:
        Number of chunks to retrieve per claim per database.
    max_concurrent:
        Maximum number of claims to process concurrently.
    """

    def __init__(
        self,
        databases: LocalVectorDB | list[LocalVectorDB],
        llm: LLMProvider | Any,
        model: str | None = None,
        similarity_threshold: float = 0.3,
        min_grounding_score: float = 0.7,
        search_type: str = "hybrid",
        top_k: int = 5,
        max_concurrent: int = 5,
    ) -> None:
        self._databases: list[LocalVectorDB] = (
            databases if isinstance(databases, list) else [databases]
        )
        self._llm: LLMProvider = (
            llm if isinstance(llm, LLMProvider) else detect_provider(llm, model)
        )
        self._similarity_threshold = similarity_threshold
        self._min_grounding_score = min_grounding_score
        self._search_type = search_type
        self._top_k = top_k
        self._max_concurrent = max_concurrent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_async(
        self,
        text: str,
        sources: list[str] | None = None,
    ) -> FactCheckResult:
        """Fact-check *text* asynchronously.

        Parameters
        ----------
        text:
            The LLM-generated text to verify.
        sources:
            Optional list of document IDs that were used to generate *text*.
            When provided, these are searched first; the full database is only
            queried when no supporting evidence is found or a contradiction is
            detected.
        """
        claims_data = await extract_claims(self._llm, text)

        if not claims_data:
            return FactCheckResult(
                claims=[],
                overall_score=1.0,
                has_contradictions=False,
                citation_text="No factual claims detected.",
            )

        sem = asyncio.Semaphore(self._max_concurrent)

        async def _guarded(claim_data: dict) -> ClaimResult:
            async with sem:
                return await self._check_claim(claim_data, sources)

        claim_results = list(
            await asyncio.gather(*(_guarded(c) for c in claims_data))
        )

        return self._build_result(text, claim_results)

    def check(
        self,
        text: str,
        sources: list[str] | None = None,
    ) -> FactCheckResult:
        """Synchronous wrapper around :meth:`check_async`."""
        return asyncio.run(self.check_async(text, sources))

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    async def _check_claim(
        self, claim_data: dict, sources: list[str] | None
    ) -> ClaimResult:
        claim = claim_data["claim"]
        sentence = claim_data.get("sentence")

        all_chunks: list[dict] = []
        all_polarities: list[PolarityResult] = []

        # Phase 1: scoped search (only the provided source documents)
        if sources:
            scoped_chunks = await self._search_scoped(claim, sources)
            if scoped_chunks:
                polarities = await classify_polarity(
                    self._llm, claim, scoped_chunks
                )
                all_chunks.extend(scoped_chunks)
                all_polarities.extend(polarities)

                has_support = any(
                    p.polarity == Polarity.SUPPORTS
                    and p.confidence >= self._min_grounding_score
                    for p in polarities
                )
                if has_support:
                    return self._best_result(
                        claim, sentence, all_chunks, all_polarities
                    )

        # Phase 2: expanded search (all documents across all databases)
        expanded_chunks = await self._search_all(claim)
        if expanded_chunks:
            polarities = await classify_polarity(
                self._llm, claim, expanded_chunks
            )
            all_chunks.extend(expanded_chunks)
            all_polarities.extend(polarities)

        if not all_chunks:
            return ClaimResult(
                claim=claim,
                grounded=False,
                confidence=0.0,
                original_sentence=sentence,
            )

        return self._best_result(claim, sentence, all_chunks, all_polarities)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    async def _search_scoped(
        self, query: str, source_ids: list[str]
    ) -> list[dict]:
        source_set = set(source_ids)
        results: list[dict] = []

        for db in self._databases:
            qr = await asyncio.to_thread(
                db.query,
                query,
                search_type=self._search_type,
                return_type="chunks",
                k=self._top_k * 5,
                score_threshold=self._similarity_threshold,
            )
            for r in qr:
                doc_id = r.document_id or r.id
                if doc_id in source_set:
                    results.append(
                        {
                            "content": r.content,
                            "document_id": doc_id,
                            "score": r.score,
                            "database": db.name,
                            "metadata": r.metadata,
                        }
                    )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[: self._top_k]

    async def _search_all(self, query: str) -> list[dict]:
        results: list[dict] = []

        for db in self._databases:
            qr = await asyncio.to_thread(
                db.query,
                query,
                search_type=self._search_type,
                return_type="chunks",
                k=self._top_k,
                score_threshold=self._similarity_threshold,
            )
            for r in qr:
                results.append(
                    {
                        "content": r.content,
                        "document_id": r.document_id or r.id,
                        "score": r.score,
                        "database": db.name,
                        "metadata": r.metadata,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[: self._top_k]

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    @staticmethod
    def _best_result(
        claim: str,
        sentence: Optional[str],
        chunks: list[dict],
        polarities: list[PolarityResult],
    ) -> ClaimResult:
        has_contradiction = any(p.polarity == Polarity.CONTRADICTS for p in polarities)

        # Find best supporting chunk
        best_support: Optional[tuple[dict, PolarityResult]] = None
        for chunk, pol in zip(chunks, polarities, strict=True):
            if pol.polarity == Polarity.SUPPORTS:
                if best_support is None or pol.confidence > best_support[1].confidence:
                    best_support = (chunk, pol)

        if has_contradiction:
            # Find the contradicting chunk for reporting
            for chunk, pol in zip(chunks, polarities, strict=True):
                if pol.polarity == Polarity.CONTRADICTS:
                    return ClaimResult(
                        claim=claim,
                        grounded=False,
                        confidence=0.0,
                        source_id=chunk["document_id"],
                        source_excerpt=pol.excerpt,
                        contradiction=True,
                        polarity=Polarity.CONTRADICTS,
                        similarity=chunk["score"],
                        original_sentence=sentence,
                        database_name=chunk["database"],
                    )

        if best_support is not None:
            chunk, pol = best_support
            return ClaimResult(
                claim=claim,
                grounded=True,
                confidence=pol.confidence,
                source_id=chunk["document_id"],
                source_excerpt=pol.excerpt,
                contradiction=False,
                polarity=Polarity.SUPPORTS,
                similarity=chunk["score"],
                original_sentence=sentence,
                database_name=chunk["database"],
            )

        # No support, no contradiction -- ungrounded
        best_chunk = chunks[0] if chunks else None
        return ClaimResult(
            claim=claim,
            grounded=False,
            confidence=0.0,
            source_id=best_chunk["document_id"] if best_chunk else None,
            polarity=Polarity.UNRELATED,
            similarity=best_chunk["score"] if best_chunk else None,
            original_sentence=sentence,
            database_name=best_chunk["database"] if best_chunk else None,
        )

    def _build_result(
        self, text: str, claim_results: list[ClaimResult]
    ) -> FactCheckResult:
        has_contradictions = any(cr.contradiction for cr in claim_results)

        if has_contradictions:
            overall_score = 0.0
        elif claim_results:
            overall_score = sum(cr.confidence for cr in claim_results) / len(
                claim_results
            )
        else:
            overall_score = 1.0

        citation_text = self._format_citations(claim_results)
        annotated_text = annotate_response(text, claim_results)

        return FactCheckResult(
            claims=claim_results,
            overall_score=overall_score,
            has_contradictions=has_contradictions,
            citation_text=citation_text,
            annotated_text=annotated_text,
        )

    @staticmethod
    def _format_citations(claim_results: list[ClaimResult]) -> str:
        lines: list[str] = []

        # Sources
        sources_seen: set[str] = set()
        for cr in claim_results:
            if cr.source_id and cr.source_id not in sources_seen:
                sources_seen.add(cr.source_id)
                db_label = f" ({cr.database_name})" if cr.database_name else ""
                lines.append(f"- {cr.source_id}{db_label}")

        if lines:
            lines.insert(0, "Sources consulted:")

        # Contradictions
        contradictions = [cr for cr in claim_results if cr.contradiction]
        if contradictions:
            lines.append("")
            lines.append("Contradictions detected:")
            for cr in contradictions:
                excerpt = f' -- "{cr.source_excerpt}"' if cr.source_excerpt else ""
                lines.append(f"  ! \"{cr.claim}\" contradicts {cr.source_id}{excerpt}")

        # Low-confidence claims
        low_conf = [
            cr
            for cr in claim_results
            if not cr.contradiction and not cr.grounded
        ]
        if low_conf:
            lines.append("")
            lines.append("Ungrounded claims:")
            for cr in low_conf:
                lines.append(f"  ? \"{cr.claim}\"")

        return "\n".join(lines) if lines else "No sources found."
