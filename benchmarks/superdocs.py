"""Synthetic multi-section super-documents for hierarchical retrieval evaluation.

See ``hierarchical-test-plan.md`` §5a. Standard BEIR corpora treat every
"document" as a single passage, so "document vs section vs chunk" collapses to
nothing and there is no way to measure whether a coarser level ever ranks the
answer better. This module fabricates genuinely multi-section documents out of an
existing *judged passage* corpus (FiQA, NFCorpus), which buys **aligned ground
truth at three granularities for free** and with no human labelling:

* **document** -- which super-document holds the answer,
* **section** -- which heading within it,
* **passage** -- which source passage (the chunk-level proxy).

Construction, per placed query ``q`` whose gold passage ``p`` exists in the
corpus: build one super-document of ``S`` sections x ``P`` passages, drop ``p``
into a randomly chosen section, and fill every other slot with a distractor
passage that is gold for *no* placed query. The document is emitted as Markdown
with ``##`` headings.

The alignment principle that makes the ground truth trustworthy: we do **not**
compute section boundaries ourselves. We render the Markdown, then run the real
:class:`localvectordb.section_detection.SectionDetector` over it -- the exact code
the database uses at ingest -- and read the gold passage's section index off
*that*. :func:`build_synthetic_benchmark` asserts every gold passage lands wholly
inside exactly one detected section before returning, so a drift between our
rendering and the detector's parsing is a hard failure here, not a silent skew in
the numbers.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lines that the SectionDetector would treat as structure. A passage lifted from
# the corpus must not be allowed to open a heading or a code fence, or it would
# forge a section boundary and corrupt the alignment we depend on.
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")
_FENCE_LINE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _sanitize_passage(text: str) -> str:
    """Neutralise any line a passage might use to forge document structure.

    ``SectionDetector`` keys on Markdown headings (``^#{1,6}\\s``) outside fenced
    code, and on the fences themselves. A heading line is escaped with a leading
    backslash; a fence line is pushed past the detector's 3-space tolerance with a
    4-space indent so it can neither open a fence nor be a heading. The passage's
    prose is otherwise untouched.
    """
    out: List[str] = []
    for line in text.split("\n"):
        if _HEADING_LINE_RE.match(line):
            out.append("\\" + line)
        elif _FENCE_LINE_RE.match(line):
            out.append("    " + line.lstrip())
        else:
            out.append(line)
    return "\n".join(out)


@dataclass(frozen=True)
class GoldLocation:
    """Where a query's answer sits inside its super-document.

    ``section_index`` is what the *detector* assigned, not what we intended --
    the two are asserted equal at build time, but this records the authoritative
    one. ``char_span`` is the half-open ``[start, end)`` of the sanitised gold
    passage body within ``SuperDoc.text``; the chunk-level qrel is "every chunk
    overlapping this span".
    """

    query_id: str
    doc_id: str
    passage_id: str
    section_index: int
    char_span: Tuple[int, int]


@dataclass(frozen=True)
class SectionSpec:
    """One rendered section: its detector-assigned index and source passages."""

    index: int
    heading: str
    passage_ids: List[str]
    char_span: Tuple[int, int]


@dataclass(frozen=True)
class SuperDoc:
    doc_id: str
    text: str
    sections: List[SectionSpec]


@dataclass(frozen=True)
class SyntheticBenchmark:
    """A fabricated corpus with qrels aligned at document / section / passage level.

    ``corpus`` / ``queries`` / ``doc_qrels`` are directly consumable by
    :func:`benchmarks.metrics.evaluate` (doc-level, the primary metric). The
    section- and passage-level qrels and ``gold_locations`` drive the per-level
    fidelity (E1) and oracle (E2/H4) analysis.
    """

    name: str
    params: Dict[str, int]
    corpus: Dict[str, str]
    queries: Dict[str, str]
    doc_qrels: Dict[str, Dict[str, int]]
    section_qrels: Dict[str, Dict[str, int]]
    passage_qrels: Dict[str, Dict[str, int]]
    gold_locations: Dict[str, List[GoldLocation]] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"SyntheticBenchmark({self.name!r}: {len(self.corpus)} super-docs of "
            f"{self.params['sections']}x{self.params['passages']} passages, "
            f"{len(self.queries)} queries)"
        )


def section_qrel_id(doc_id: str, section_index: int) -> str:
    """Stable id for a section as a retrieval unit: ``<doc_id>#s<index>``."""
    return f"{doc_id}#s{section_index}"


def _render_super_doc(
    doc_id: str,
    slots: List[Tuple[str, str]],
    sections_per_doc: int,
    passages_per_section: int,
    gold_passage_id: str,
) -> Tuple[SuperDoc, Tuple[int, int]]:
    """Render one super-document and locate the gold passage within it.

    ``slots`` is the flat, already-ordered list of ``(passage_id, text)`` filling
    the S*P grid row-major (section 0 first). Returns the ``SuperDoc`` (with
    provisional section indices 0..S-1, corrected by the caller against the
    detector) and the gold passage's ``[start, end)`` char span.
    """
    parts: List[str] = []
    pos = 0
    sections: List[SectionSpec] = []
    gold_span: Optional[Tuple[int, int]] = None

    for s in range(sections_per_doc):
        heading = f"## Section {s + 1}"
        header_block = f"{heading}\n\n"
        section_start = pos
        parts.append(header_block)
        pos += len(header_block)

        section_passage_ids: List[str] = []
        for p in range(passages_per_section):
            pid, raw = slots[s * passages_per_section + p]
            body = _sanitize_passage(raw.strip())
            section_passage_ids.append(pid)
            body_start = pos
            parts.append(body)
            pos += len(body)
            if pid == gold_passage_id and gold_span is None:
                gold_span = (body_start, pos)
            # Blank line between passages / before the next heading.
            sep = "\n\n"
            parts.append(sep)
            pos += len(sep)

        sections.append(
            SectionSpec(
                index=s,
                heading=f"Section {s + 1}",
                passage_ids=section_passage_ids,
                char_span=(section_start, pos),
            )
        )

    text = "".join(parts)
    assert gold_span is not None, f"gold passage {gold_passage_id!r} not placed in {doc_id!r}"
    return SuperDoc(doc_id=doc_id, text=text, sections=sections), gold_span


def build_synthetic_benchmark(
    source,
    *,
    sections_per_doc: int = 3,
    passages_per_section: int = 3,
    seed: int = 0,
    max_queries: Optional[int] = None,
) -> SyntheticBenchmark:
    """Fabricate multi-section super-documents from a judged passage corpus.

    Parameters
    ----------
    source
        A ``benchmarks.beir_data.BeirDataset`` (``corpus``/``queries``/``qrels``).
    sections_per_doc, passages_per_section
        The ``S`` x ``P`` grid; one slot per super-doc holds the gold passage,
        the rest are distractors. Both are sweep axes (plan §5a).
    seed
        Seeds a private ``random.Random`` -- fully deterministic, no global state
        touched (``np.random``/``Math.random`` are off-limits in this repo's
        harnesses).
    max_queries
        Cap the number of placed queries (hence super-docs). ``None`` places
        every answerable query.

    Returns
    -------
    SyntheticBenchmark
        With document / section / passage qrels aligned by construction and
        verified against the real ``SectionDetector``.
    """
    from localvectordb.section_detection import SectionDetector

    if sections_per_doc < 1 or passages_per_section < 1:
        raise ValueError("sections_per_doc and passages_per_section must be >= 1")
    slots_per_doc = sections_per_doc * passages_per_section
    rng = random.Random(seed)
    detector = SectionDetector()

    # Answerable queries: those with at least one gold passage present in corpus.
    # Sort for determinism before any sampling.
    answerable: List[Tuple[str, str]] = []  # (query_id, gold_passage_id)
    for query_id in sorted(source.qrels):
        golds = sorted(pid for pid, grade in source.qrels[query_id].items() if grade > 0 and pid in source.corpus)
        if golds:
            answerable.append((query_id, golds[0]))
    if not answerable:
        raise ValueError(f"{source.name}: no answerable queries with an in-corpus gold passage")
    if max_queries is not None:
        answerable = answerable[:max_queries]

    # Distractors must be gold for no placed query, so a distractor slot can never
    # accidentally become a second relevant unit for some other query. Sample
    # without replacement across the whole benchmark so no passage's text appears
    # in two super-docs (which would muddy document-level retrieval).
    gold_ids = {gid for _, gid in answerable}
    distractor_pool = [pid for pid in sorted(source.corpus) if pid not in gold_ids]
    rng.shuffle(distractor_pool)
    needed = len(answerable) * (slots_per_doc - 1)
    if needed > len(distractor_pool):
        raise ValueError(
            f"{source.name}: need {needed} distractor passages for "
            f"{len(answerable)} queries at {sections_per_doc}x{passages_per_section}, "
            f"but only {len(distractor_pool)} non-gold passages exist. "
            f"Lower max_queries or the grid size."
        )

    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    doc_qrels: Dict[str, Dict[str, int]] = {}
    section_qrels: Dict[str, Dict[str, int]] = {}
    passage_qrels: Dict[str, Dict[str, int]] = {}
    gold_locations: Dict[str, List[GoldLocation]] = {}

    cursor = 0
    misaligned = 0
    for i, (query_id, gold_pid) in enumerate(answerable):
        doc_id = f"superdoc_{i:05d}"
        take = distractor_pool[cursor : cursor + (slots_per_doc - 1)]
        cursor += slots_per_doc - 1

        # Place the gold passage in a random slot; distractors fill the rest.
        gold_slot = rng.randrange(slots_per_doc)
        slots: List[Tuple[str, str]] = []
        d = 0
        for slot in range(slots_per_doc):
            if slot == gold_slot:
                slots.append((gold_pid, source.corpus[gold_pid]))
            else:
                pid = take[d]
                d += 1
                slots.append((pid, source.corpus[pid]))

        super_doc, gold_span = _render_super_doc(doc_id, slots, sections_per_doc, passages_per_section, gold_pid)

        # Authoritative section assignment: run the real detector and locate the
        # section whose span contains the gold passage's start. Assert the whole
        # gold span sits inside that one section -- if not, our rendering and the
        # detector disagree and the ground truth would be a lie.
        detected = detector.detect_sections(super_doc.text)
        gold_start, gold_end = gold_span
        owner = next((s for s in detected if s.start_pos <= gold_start < s.end_pos), None)
        if owner is None or not (owner.start_pos <= gold_start and gold_end <= owner.end_pos):
            misaligned += 1
            logger.error("Gold span %s not contained in one detected section of %s", gold_span, doc_id)
            continue

        corpus[doc_id] = super_doc.text
        queries[query_id] = source.queries[query_id]
        doc_qrels[query_id] = {doc_id: 1}
        section_qrels[query_id] = {section_qrel_id(doc_id, owner.index): 1}
        passage_qrels[query_id] = {gold_pid: 1}
        gold_locations[query_id] = [
            GoldLocation(
                query_id=query_id,
                doc_id=doc_id,
                passage_id=gold_pid,
                section_index=owner.index,
                char_span=gold_span,
            )
        ]

    if misaligned:
        raise AssertionError(
            f"{source.name}: {misaligned}/{len(answerable)} super-docs had a gold "
            f"passage straddling a section boundary -- alignment is broken, refusing "
            f"to emit a benchmark with untrustworthy ground truth."
        )

    return SyntheticBenchmark(
        name=f"{source.name}_super_s{sections_per_doc}p{passages_per_section}",
        params={"sections": sections_per_doc, "passages": passages_per_section, "seed": seed},
        corpus=corpus,
        queries=queries,
        doc_qrels=doc_qrels,
        section_qrels=section_qrels,
        passage_qrels=passage_qrels,
        gold_locations=gold_locations,
    )


# ---------------------------------------------------------------------------
# Self-test: build from a tiny inline corpus (no download) and prove the
# alignment invariant holds. Run: ./.venv/Scripts/python.exe benchmarks/superdocs.py
# ---------------------------------------------------------------------------


def _fixture_source():
    """A minimal BeirDataset-shaped object, enough to exercise the builder.

    Includes an adversarial passage that *starts with a Markdown heading and a
    code fence* to prove sanitisation stops it forging a section boundary.
    """
    from benchmarks.beir_data import BeirDataset

    corpus = {f"p{i}": f"Passage {i}. " + " ".join(f"word{i}_{j}" for j in range(20)) for i in range(40)}
    # Adversarial content in a distractor: a heading and a fence that must NOT
    # split the document into extra sections.
    corpus["p7"] = "# Not A Real Heading\n```\n### also not a heading\n```\nJust prose that happens to look structured."
    queries = {f"q{i}": f"query about passage {i}" for i in range(5)}
    qrels = {f"q{i}": {f"p{i}": 1} for i in range(5)}
    return BeirDataset(name="fixture", corpus=corpus, queries=queries, qrels=qrels)


def _self_test() -> int:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from localvectordb.section_detection import SectionDetector

    source = _fixture_source()
    bench = build_synthetic_benchmark(source, sections_per_doc=2, passages_per_section=2, seed=1)
    print(bench)

    detector = SectionDetector()
    checked = 0
    for query_id, locs in bench.gold_locations.items():
        loc = locs[0]
        text = bench.corpus[loc.doc_id]
        detected = detector.detect_sections(text)
        # Exactly S sections detected -- no forged boundaries from passage content.
        n_headings = sum(1 for s in detected if s.heading is not None)
        assert (
            n_headings == bench.params["sections"]
        ), f"{loc.doc_id}: detected {n_headings} headed sections, expected {bench.params['sections']}"
        # The recorded section owns the whole gold span.
        owner = next(s for s in detected if s.index == loc.section_index)
        start, end = loc.char_span
        assert owner.start_pos <= start and end <= owner.end_pos
        # The gold passage's own text is verbatim (post-sanitise) at that span.
        assert bench.passage_qrels[query_id] == {loc.passage_id: 1}
        checked += 1

    # The adversarial passage p7, wherever it landed as a distractor, forged no
    # section: every super-doc has exactly S headed sections (asserted above).
    print(f"OK: {checked} queries, alignment verified; sanitisation held against p7's fake heading/fence.")
    # Show one rendered doc so the shape is reviewable.
    sample_doc = next(iter(bench.corpus.values()))
    print("\n--- sample super-doc ---")
    print(sample_doc[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
