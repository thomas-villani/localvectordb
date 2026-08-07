"""Natural Questions: real *user* queries over sectioned Wikipedia articles.

The fourth corpus, and the first with queries nobody wrote for the benchmark --
NQ questions are sampled from Google search logs. That matters because every
other corpus here has a known query-provenance caveat: qasper's questions were
written by readers *of the paper they are gold for*, MAUD's are a fixed expert
questionnaire, and MLDR's are openly LLM-generated. If a finding survives here
it has survived the weakest link the others share.

What NQ adds, measured rather than assumed (see ``_self_test``):

* **Real section structure.** Wikipedia ``<H2>`` headings are emitted as ``##``
  and recovered by the real :class:`SectionDetector`, exactly as in
  ``qasper_data``. ~14 sections/article (median), median section 1,126 chars.
* **A heavier document tail.** Median article 25.9k chars, p90 **84.3k** --
  against qasper's 22.9k / 33.4k. Same section size, much longer documents: the
  one axis that moves while granularity stays fixed.
* **Gold that is *less* section-shaped than qasper's.** See the circularity note
  below; this is the reason to trust a chunks-vs-sections verdict measured here.
* **Non-prose gold.** 23.8% of long answers are tables or lists -- a modality
  qasper drops outright (its ``FLOAT SELECTED`` evidence). Pass
  ``drop_table_gold=True`` to exclude them and measure the difference.

**The circularity check, and why it also indicts qasper.** A section-level index
gets an unearned win if the gold span *is* essentially the whole section, since
then "retrieve the section" and "retrieve the answer" are the same act (the
concern ``span-length-crossover-findings`` §6.14 raised about qasper). Measuring
gold-chars / owning-section-chars on both corpora:

===========================================  =========  =========
statistic                                    NQ         qasper
===========================================  =========  =========
median coverage                              0.331      0.516
gold covers >80% of its section               19.1%      36.1%
gold covers >99% of its section                3.6%      18.0%
===========================================  =========  =========

NQ is the *harder* corpus for a section index, by a factor of five at the tail.
The finding to carry forward is the second column: on nearly a fifth of qasper's
(query, gold-section) pairs the section and the evidence are the same text, so
qasper's section win is measured on partly favourable ground.

**Section 0.** MediaWiki treats an article's lead as section 0, but it carries
no heading, so nothing before the first ``<H2>`` would belong to any section --
and **54.7%** of gold long answers live there (lead paragraph or infobox). This
loader emits ``## Introduction`` for it. That names an existing structural unit
rather than inventing relevance, and is the same move ``qasper_data`` already
makes for the abstract. With it, 100% of located gold is owned by exactly one
section and none straddles a boundary; without it, 54.7% of queries would be
dropped and the survivors would be biased against lead-answerable questions.

**Source and access.** Google's ``gs://natural_questions`` bucket no longer
serves anonymous readers (``AccessDenied`` on ``storage.objects.get``), so this
reads the HuggingFace parquet mirror of the same v1.0 dev split. That needs
``pyarrow``; it is imported lazily so the rest of ``benchmarks/`` stays free of
it. Licence: CC BY-SA 3.0 (Wikipedia-derived), Kwiatkowski et al. 2019.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.config import DATA_DIR  # noqa: E402
from benchmarks.superdocs import SyntheticBenchmark, section_qrel_id  # noqa: E402

logger = logging.getLogger(__name__)

_PARQUET_INDEX = "https://huggingface.co/api/datasets/google-research-datasets/natural_questions/parquet"
N_SHARDS = 6  # default/validation
LEAD_HEADING = "Introduction"

_H_OPEN = re.compile(r"^<H([1-6])>$")
_H_CLOSE = re.compile(r"^</H([1-6])>$")
_FORGES_HEADING = re.compile(r"^#{1,6}$")
# Long-answer candidates opening with one of these are tables/lists, not prose.
_NON_PROSE = ("<Table", "<Tr", "<Ul", "<Ol", "<Dl", "<Li", "<Dd")


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "benchmarks.nq_data reads the HuggingFace parquet mirror and needs pyarrow "
            "(`uv sync --dev`). Google's original gs://natural_questions bucket is no "
            "longer anonymously readable, so there is no jsonl.gz path to fall back to."
        ) from exc
    return pq


def download(
    *,
    shards: Optional[Sequence[int]] = None,
    data_dir: Optional[Path] = None,
    force: bool = False,
) -> List[Path]:
    """Fetch NQ dev parquet shards; return their local paths in shard order.

    The full dev split is ~1.3 GB across 6 shards. ``shards`` restricts the
    download (``range(N_SHARDS)`` by default); shard order is deterministic, so a
    subset is a reproducible sample -- but prefer capping with ``max_queries`` in
    :func:`load_nq`, which keeps the sampled population well defined.
    """
    import httpx

    wanted = list(range(N_SHARDS)) if shards is None else list(shards)
    root = (data_dir or DATA_DIR) / "nq"
    root.mkdir(parents=True, exist_ok=True)

    urls: Optional[List[str]] = None
    out: List[Path] = []
    for i in wanted:
        dest = root / f"validation-{i}.parquet"
        if dest.exists() and not force:
            out.append(dest)
            continue
        if urls is None:
            index = httpx.get(_PARQUET_INDEX, follow_redirects=True, timeout=60).json()
            urls = index["default"]["validation"]
        logger.info("Downloading NQ dev shard %d ...", i)
        with httpx.stream("GET", urls[i], follow_redirects=True, timeout=1800) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)  # atomic: a killed download must not look complete
        logger.info("Wrote %s (%.0f MB)", dest, dest.stat().st_size / 1e6)
        out.append(dest)
    return out


def render_document(
    tokens: dict, *, lead_heading: Optional[str] = LEAD_HEADING
) -> Tuple[str, List[Optional[Tuple[int, int]]], str]:
    """Render NQ document tokens to Markdown.

    Returns ``(text, token_char_spans, title)``. ``token_char_spans[i]`` is the
    ``[start, end)`` span of token ``i`` within ``text``, or ``None`` when the
    token produced no output (HTML markup, heading text). NQ's long-answer
    annotations index this same token array, so a gold span maps straight through
    without parsing ``document.html``.

    Only ``<H2>``..``<H6>`` become sections; ``<H1>`` is the page title. Body
    tokens are emitted space-joined on a single line, so no body text can forge a
    heading -- except a lone ``#``-run landing first on a line, which is escaped.
    """
    toks, is_html = tokens["token"], tokens["is_html"]
    out: List[str] = []
    spans: List[Optional[Tuple[int, int]]] = []
    pos = 0
    title = ""
    heading_level = 0
    heading_buf: List[str] = []
    at_line_start = True

    def emit(s: str) -> Tuple[int, int]:
        nonlocal pos
        out.append(s)
        start = pos
        pos += len(s)
        return start, pos

    if lead_heading:
        emit(f"## {lead_heading}\n\n")

    for tok, ish in zip(toks, is_html, strict=True):
        if ish:
            spans.append(None)
            opened = _H_OPEN.match(tok)
            if opened:
                heading_level, heading_buf = int(opened.group(1)), []
                continue
            closed = _H_CLOSE.match(tok)
            if closed and heading_level:
                text = " ".join(heading_buf).strip()
                if heading_level == 1:
                    title = title or text
                elif text:
                    emit(("\n" if pos else "") + f"## {text}\n\n")
                    at_line_start = True
                heading_level, heading_buf = 0, []
            continue

        if heading_level:
            heading_buf.append(tok)
            spans.append(None)
            continue

        body = tok.replace("\n", " ")
        if at_line_start and _FORGES_HEADING.match(body):
            body = "\\" + body
        spans.append(emit(("" if at_line_start else " ") + body))
        at_line_start = False

    return "".join(out), spans, title


def _char_span(
    spans: Sequence[Optional[Tuple[int, int]]], start_token: int, end_token: int
) -> Optional[Tuple[int, int]]:
    """Char span covering tokens ``[start_token, end_token)``, skipping markup."""
    present = [sp for sp in spans[start_token:end_token] if sp is not None]
    if not present:
        return None
    return present[0][0], present[-1][1]


def _owner_section(detected: Iterable, start: int, end: int):
    """The detected headed section wholly containing ``[start, end)``, else None."""
    owner = next((s for s in detected if s.start_pos <= start < s.end_pos), None)
    if owner is None or owner.heading is None or end > owner.end_pos:
        return None
    return owner


def load_nq(
    *,
    shards: Optional[Sequence[int]] = None,
    data_dir: Optional[Path] = None,
    max_queries: Optional[int] = None,
    lead_heading: Optional[str] = LEAD_HEADING,
    drop_table_gold: bool = False,
) -> SyntheticBenchmark:
    """Load NQ dev as a benchmark with document- and section-level qrels.

    Parameters
    ----------
    shards
        Which dev parquet shards to read (default all 6, ~3.2k usable queries).
    max_queries
        Cap the query count in deterministic file order. The corpus shrinks with
        it -- NQ carries ~1.04 queries per article, so docs ~= queries and there
        are no pure distractors, the same shape as qasper.
    lead_heading
        Heading emitted for the article lead. ``None`` reproduces raw Wikipedia,
        at the cost of dropping the 54.7% of questions answered there.
    drop_table_gold
        Exclude questions whose long answer is a table or list (23.8%). Off by
        default: excluding them silently narrows the task to prose retrieval.
    """
    from localvectordb.section_detection import SectionDetector

    pq = _require_pyarrow()
    paths = download(shards=shards, data_dir=data_dir)
    detector = SectionDetector()

    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    doc_qrels: Dict[str, Dict[str, int]] = {}
    section_qrels: Dict[str, Dict[str, int]] = {}
    passage_qrels: Dict[str, Dict[str, int]] = {}

    n_rows = n_null = n_table = n_unlocated = n_unowned = 0
    done = False
    for path in paths:
        if done:
            break
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            if done:
                break
            for row in pf.read_row_group(rg).to_pylist():
                if max_queries is not None and len(queries) >= max_queries:
                    done = True
                    break
                n_rows += 1
                gold = next(
                    (la for la in row["annotations"]["long_answer"] if la["candidate_index"] >= 0),
                    None,
                )
                if gold is None:
                    n_null += 1
                    continue

                doc = row["document"]
                candidates = row["long_answer_candidates"]
                first = doc["tokens"]["token"][candidates["start_token"][gold["candidate_index"]]]
                is_table = first.startswith(_NON_PROSE)
                if is_table:
                    n_table += 1
                    if drop_table_gold:
                        continue

                # oldid is part of the URL, so two revisions of one article stay
                # distinct docs -- their token arrays differ and section indices
                # must not be shared between them.
                pid = doc["url"]
                text, spans, _ = render_document(doc["tokens"], lead_heading=lead_heading)

                span = _char_span(spans, gold["start_token"], gold["end_token"])
                if span is None:
                    n_unlocated += 1
                    continue
                owner = _owner_section([s for s in detector.detect_sections(text) if s.heading], *span)
                if owner is None:
                    n_unowned += 1
                    continue

                qid = row["id"]
                queries[qid] = row["question"]["text"]
                corpus[pid] = text
                doc_qrels[qid] = {pid: 1}
                section_qrels[qid] = {section_qrel_id(pid, owner.index): 1}
                passage_qrels[qid] = {}

    if not queries:
        raise ValueError("NQ: no questions with a locatable long answer")

    logger.info(
        "NQ dev: %d docs, %d/%d questions kept (%d no long answer, %d unlocated, %d unowned; %d table/list gold)",
        len(corpus),
        len(queries),
        n_rows,
        n_null,
        n_unlocated,
        n_unowned,
        n_table,
    )
    return SyntheticBenchmark(
        name="nq_dev",
        params={
            "source": "natural_questions",
            "split": "validation",
            "shards": list(range(N_SHARDS)) if shards is None else list(shards),
            "lead_heading": lead_heading,
            "drop_table_gold": drop_table_gold,
            "queries": "real (Google search logs)",
            "docs": len(corpus),
        },
        corpus=corpus,
        queries=queries,
        doc_qrels=doc_qrels,
        section_qrels=section_qrels,
        passage_qrels=passage_qrels,
    )


def _self_test() -> int:  # pragma: no cover - manual characterisation entry point
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    from localvectordb.section_detection import SectionDetector

    bench = load_nq(shards=[0], max_queries=400)
    print(bench)

    detector = SectionDetector()
    sec_counts, sec_lens, doc_lens = [], [], []
    for text in bench.corpus.values():
        headed = [s for s in detector.detect_sections(text) if s.heading]
        sec_counts.append(len(headed))
        sec_lens.extend(s.end_pos - s.start_pos for s in headed)
        doc_lens.append(len(text))

    # Alignment: every section qrel must name a section the detector really found.
    for qid, sids in bench.section_qrels.items():
        pid = next(iter(bench.doc_qrels[qid]))
        found = {section_qrel_id(pid, s.index) for s in detector.detect_sections(bench.corpus[pid]) if s.heading}
        assert set(sids) <= found, f"{qid}: section qrel {sids} not among {len(found)} detected sections"

    print(f"sections/doc   med {np.median(sec_counts):.0f}  p90 {np.percentile(sec_counts, 90):.0f}")
    print(f"section chars  med {np.median(sec_lens):,.0f}  p90 {np.percentile(sec_lens, 90):,.0f}")
    print(f"doc chars      med {np.median(doc_lens):,.0f}  p90 {np.percentile(doc_lens, 90):,.0f}")
    print(f"OK: {len(bench.section_qrels)} section qrels align to detected sections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
