"""Qasper: real long-document retrieval with native section-level relevance.

See ``hierarchical-test-plan.md`` §5b. Qasper is ~1.5k NLP papers with their full
text broken into named sections, and questions whose answers are grounded in
specific *evidence paragraphs*. That gives us, off the shelf and with no
synthesis, exactly what the synthetic super-docs approximate: genuinely long,
naturally-sectioned documents with relevance judged at the paragraph/section
level. It is the external-validity check on the synthetic Phase-1 results.

We render each paper to the same Markdown shape the synthetic builder uses
(``##`` section headings) and derive section-level ground truth the same way:
run the real :class:`SectionDetector` over the rendered text and read the
evidence paragraph's section off *that*, so the qrels align to what the database
would compute. Evidence we cannot locate in the body (figure/table references,
paraphrased spans) is dropped rather than guessed; a question with no locatable
textual evidence is skipped.

No ``datasets`` dependency: Qasper is distributed as plain JSON, downloaded and
parsed directly like the BEIR corpora in ``beir_data.py``.

The result is a :class:`benchmarks.superdocs.SyntheticBenchmark` (the name is a
misnomer for real data, but the shape -- corpus / queries / doc+section qrels --
is exactly what ``eval_hierarchical`` consumes, so the same engine runs unchanged).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# When run as a script, ``benchmarks/`` (not the project root) is on sys.path, so
# the ``benchmarks.*`` package imports below would fail. Ensure the root is present
# (a no-op when imported normally, e.g. by eval_hierarchical after its path fix).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.config import DATA_DIR  # noqa: E402
from benchmarks.superdocs import SyntheticBenchmark, _sanitize_passage, section_qrel_id  # noqa: E402

logger = logging.getLogger(__name__)

# AllenAI's official v0.3 release (train + dev in one archive). Direct JSON, no
# ``datasets`` library needed.
QASPER_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
_FLOAT_PREFIX = "FLOAT SELECTED"  # Qasper marks figure/table evidence with this; not body text.


def _safe_extract_tar(archive: tarfile.TarFile, dest: Path) -> None:
    """Extract ``archive`` into ``dest``, refusing any member that escapes it."""
    dest_root = dest.resolve()
    for member in archive.getmembers():
        target = (dest_root / member.name).resolve()
        if not target.is_relative_to(dest_root):
            raise ValueError(f"Refusing tar entry that escapes the destination: {member.name!r}")
    archive.extractall(dest_root)  # noqa: S202 - members validated above


def download(*, data_dir: Optional[Path] = None, force: bool = False) -> Path:
    """Download and extract the Qasper train/dev JSON. Returns the directory."""
    root = ((data_dir or DATA_DIR).resolve()) / "qasper"
    root.mkdir(parents=True, exist_ok=True)
    if any(root.glob("*dev*.json")) and not force:
        logger.info("Qasper already cached at %s", root)
        return root

    import httpx

    archive = root / "qasper-train-dev-v0.3.tgz"
    logger.info("Downloading %s ...", QASPER_URL)
    with httpx.stream("GET", QASPER_URL, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(archive, "wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done * 100 // total}% ({done >> 20}MB / {total >> 20}MB)", end="", flush=True)
    print()
    with tarfile.open(archive) as tf:
        _safe_extract_tar(tf, root)
    archive.unlink()
    logger.info("Extracted Qasper to %s", root)
    return root


def _clean_heading(name: Optional[str], index: int) -> str:
    """A one-line, non-empty section heading the SectionDetector will accept.

    An empty or whitespace ``## `` fails the heading regex (it needs a
    non-newline char after the space), so paragraphs would merge into the
    previous section. Collapse whitespace and fall back to a numbered name.
    """
    cleaned = re.sub(r"\s+", " ", (name or "").strip())
    return cleaned or f"Section {index}"


def _render_paper(paper: dict) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """Render one paper to Markdown; return the text and a paragraph->span map.

    Sections become ``##`` headings (abstract first). Paragraph bodies are
    sanitised so their content cannot forge a section boundary. The returned map
    keys each *original* paragraph string to the ``[start, end)`` char span of
    its rendered (sanitised) body, so evidence strings can be located later.
    """
    parts: List[str] = []
    pos = 0
    para_spans: Dict[str, Tuple[int, int]] = {}

    def emit(s: str) -> None:
        nonlocal pos
        parts.append(s)
        pos += len(s)

    sections: List[Tuple[str, List[str]]] = []
    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        sections.append(("Abstract", [abstract]))
    for entry in paper.get("full_text") or []:
        paras = [p for p in (entry.get("paragraphs") or []) if p and p.strip()]
        if paras:
            sections.append((entry.get("section_name"), paras))

    for i, (name, paras) in enumerate(sections):
        emit(f"## {_clean_heading(name, i)}\n\n")
        for para in paras:
            body = _sanitize_passage(para.strip())
            start = pos
            emit(body)
            para_spans.setdefault(para, (start, pos))  # first occurrence wins
            emit("\n\n")

    return "".join(parts), para_spans


def _owner_section_index(detected, span: Tuple[int, int]) -> Optional[int]:
    """The detected headed-section index that wholly contains ``span``, else None."""
    s0, s1 = span
    owner = next((s for s in detected if s.start_pos <= s0 < s.end_pos), None)
    if owner is None or owner.heading is None or not (owner.start_pos <= s0 and s1 <= owner.end_pos):
        return None
    return owner.index


def _evidence_strings(qa: dict) -> List[str]:
    """Collect body-text evidence strings across a question's answers."""
    out: List[str] = []
    for ans in qa.get("answers") or []:
        answer = ans.get("answer") or {}
        for ev in answer.get("evidence") or []:
            if ev and not ev.startswith(_FLOAT_PREFIX):
                out.append(ev)
    return out


def load_qasper(
    *,
    split: str = "dev",
    data_dir: Optional[Path] = None,
    max_papers: Optional[int] = None,
    seed: int = 0,
) -> SyntheticBenchmark:
    """Load Qasper as a benchmark with document- and section-level qrels.

    Parameters
    ----------
    split
        ``"dev"`` (~280 papers, the default) or ``"train"`` (~880).
    max_papers
        Cap the corpus to the first ``max_papers`` papers (deterministic order);
        all of their answerable questions are kept. ``None`` uses every paper.
    seed
        Unused for selection (order is deterministic); accepted for a uniform
        call signature with the synthetic builder.
    """
    from localvectordb.section_detection import SectionDetector

    root = download(data_dir=data_dir)
    matches = sorted(root.glob(f"*{split}*.json"))
    if not matches:
        raise FileNotFoundError(f"No Qasper {split!r} JSON under {root}")
    data = json.loads(matches[0].read_text(encoding="utf-8"))

    detector = SectionDetector()
    paper_ids = sorted(data)
    if max_papers is not None:
        paper_ids = paper_ids[:max_papers]

    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    doc_qrels: Dict[str, Dict[str, int]] = {}
    section_qrels: Dict[str, Dict[str, int]] = {}
    passage_qrels: Dict[str, Dict[str, int]] = {}

    n_questions = n_kept = n_evidence = n_located = 0
    for pid in paper_ids:
        paper = data[pid]
        text, para_spans = _render_paper(paper)
        detected = detector.detect_sections(text)
        added_paper = False
        for qa in paper.get("qas") or []:
            n_questions += 1
            question = (qa.get("question") or "").strip()
            qid = qa.get("question_id")
            if not question or not qid:
                continue
            evidence = _evidence_strings(qa)
            n_evidence += len(evidence)
            gold_sections = set()
            for ev in evidence:
                span = para_spans.get(ev) or para_spans.get(ev.strip())
                if span is None:
                    continue
                idx = _owner_section_index(detected, span)
                if idx is not None:
                    gold_sections.add(section_qrel_id(pid, idx))
                    n_located += 1
            if not gold_sections:
                continue
            queries[qid] = question
            doc_qrels[qid] = {pid: 1}
            section_qrels[qid] = {sid: 1 for sid in gold_sections}
            passage_qrels[qid] = {}  # paragraph-level ids not needed (engine scores doc+section)
            n_kept += 1
            added_paper = True

        if added_paper:
            corpus[pid] = text

    if not queries:
        raise ValueError(f"Qasper {split!r}: no questions with locatable textual evidence")

    logger.info(
        "Qasper %s: %d papers, %d/%d questions kept, %d/%d evidence spans located",
        split,
        len(corpus),
        n_kept,
        n_questions,
        n_located,
        n_evidence,
    )
    return SyntheticBenchmark(
        name=f"qasper_{split}",
        params={"source": "qasper", "split": split, "papers": len(corpus)},
        corpus=corpus,
        queries=queries,
        doc_qrels=doc_qrels,
        section_qrels=section_qrels,
        passage_qrels=passage_qrels,
    )


def _self_test() -> int:
    import sys

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from localvectordb.section_detection import SectionDetector

    bench = load_qasper(split="dev", max_papers=25)
    print(bench)
    detector = SectionDetector()
    # Alignment: every section qrel names a real detected section of its doc.
    checked = 0
    for qid, sids in bench.section_qrels.items():
        pid = next(iter(bench.doc_qrels[qid]))
        detected = {section_qrel_id(pid, s.index) for s in detector.detect_sections(bench.corpus[pid]) if s.heading}
        assert set(sids) <= detected, f"{qid}: section qrel {sids} not among detected {len(detected)} sections"
        checked += 1
    docs = len(bench.corpus)
    total_sections = sum(len([s for s in detector.detect_sections(t) if s.heading]) for t in bench.corpus.values())
    sec_per_doc = total_sections / max(docs, 1)
    print(f"OK: {checked} queries, section qrels align to detected sections; ~{sec_per_doc:.1f} sections/paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
