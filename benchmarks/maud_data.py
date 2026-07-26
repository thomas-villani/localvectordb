"""MAUD (Atticus merger agreements) loader for the dual-embedding experiment.

    ./.venv/Scripts/python.exe benchmarks/maud_data.py --max-contracts 152

Source: MAUD v1 (Zenodo 7500064, CC-BY-4.0), downloaded to
``benchmarks/data/maud/extracted/data`` (152 contract .txt files + three CSVs).
Frames MAUD as a **per-contract retrieval** benchmark: one query per
(contract, deal-point) pair whose expert-extracted span text can be re-located
in the contract, gold = the contract sections overlapping the located spans.
Format traps this module absorbs (found the hard way — see
``experiments/dual-embedding-plan.md`` §12):

- The "92 questions" are multiple-choice classification variants; the retrieval
  granularity is the 22 ``text_type`` deal points. ``data_type == "main"`` rows
  collapse to exactly one extracted text blob per (contract, deal point).
- Blobs are multi-excerpt, separated by ``<omitted>`` (six typo variants in the
  wild) AND by ``(Page N)`` markers, which act as implicit excerpt breaks.
- There are NO character offsets. Spans re-locate only after canonicalization
  (curly→straight quotes, dash folding, lowercase, strip ALL whitespace); the
  canon→original index map recovers char ranges in the cleaned contract text.
- Contract files embed EDGAR scrape furniture (page-break rules + timestamp +
  sec.gov URL + page fraction) mid-text; it is stripped BEFORE anything else so
  every downstream offset refers to the same cleaned text.
- MAUD's own train/dev/test splits are per question-example (every contract
  appears in all three) — ignored here; one corpus.
- Contracts are flat run-on text. Clause headings come in three families
  ("Section N.N Title", "SECTION N.N", bare "N.N  Title" after a wide gap) and
  are drowned in cross-references ("... pursuant to Section 2.05 ..."). The
  sectionizer keeps candidates that survive (a) a TOC dense-run filter and
  (b) a longest-increasing-subsequence filter on clause numbers — headings
  appear in numeric order, cross-references do not.

Queries embed as ``"{deal point}. {hand-written description}"`` — the deal-point
names alone are flat labels ("Type of Consideration") with too little semantic
surface for a retrieval query.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.superdocs import section_qrel_id  # noqa: E402
from localvectordb.section_detection import SectionBoundary  # noqa: E402

logger = logging.getLogger("benchmarks.maud_data")

DATA_DIR = _ROOT / "benchmarks" / "data" / "maud" / "extracted" / "data"

MIN_PIECE_CANON_CHARS = 15  # canon chars; shorter fragments are marker debris
MIN_PAIR_COVERAGE = 0.8  # keep a (contract, deal point) pair if >=80% of its span chars locate
MIN_GOLD_OVERLAP_CHARS = 30  # a section is gold if it overlaps a located span by this much
TOC_RUN_GAP = 80  # candidate headings closer than this, in runs, are TOC entries
TOC_RUN_LEN = 4  # minimum run length to call it a TOC

# Deal-point descriptions (hand-written 2026-07-24 from the ABA deal-point
# definitions the MAUD paper annotates). The name alone is the label lawyers
# use; the description carries the semantics the embedding needs. Keys must
# match the CSV ``text_type`` values after whitespace collapse.
DEAL_POINT_DESCRIPTIONS: Dict[str, str] = {
    "MAE Definition": (
        'Definition of "Material Adverse Effect" (Company Material Adverse Effect), including '
        "forward-looking language and the carve-out exceptions for general economic, industry, "
        "market, war, or pandemic conditions and disproportionate-impact qualifiers."
    ),
    "Knowledge Definition": (
        'Definition of "Knowledge" of the company — actual versus constructive knowledge after '
        "due inquiry, and whose knowledge (which officers or employees) counts."
    ),
    "Type of Consideration": (
        "The form of merger consideration target shareholders receive per share — all cash, "
        "all stock, mixed cash and stock, or an election between them."
    ),
    "Accuracy of Target R&W Closing Condition": (
        "Closing condition requiring the target's representations and warranties to be true and "
        "correct — the bring-down standard (in all respects, in all material respects, or at a "
        "Material Adverse Effect standard) and the timing (at signing and/or at closing)."
    ),
    "Compliance with Covenant Closing Condition": (
        "Closing condition requiring the target to have performed and complied in all material "
        "respects with its covenants, agreements, and obligations under the merger agreement."
    ),
    "Absence of Litigation Closing Condition": (
        "Closing condition that no litigation, action, suit, or proceeding — governmental or "
        "otherwise, pending or threatened — challenges or seeks to enjoin the transaction."
    ),
    "No-Shop": (
        "No-shop / non-solicitation covenant restricting the target from soliciting, initiating, "
        "or encouraging alternative acquisition proposals or engaging in discussions or "
        "negotiations with other bidders."
    ),
    "Fiduciary exception: Board determination (no-shop)": (
        "Fiduciary exception to the no-shop — the board determination standard (inconsistency "
        "with fiduciary duties, superior proposal or reasonable likelihood thereof) required "
        "before the target may engage with an unsolicited acquisition proposal."
    ),
    "Fiduciary exception to COR covenant": (
        "Fiduciary exception permitting the target board to change, withdraw, or qualify its "
        "recommendation of the merger when required by its fiduciary duties, in response to a "
        "superior proposal or an intervening event."
    ),
    "Agreement provides for matching rights in connection with COR": (
        "Buyer matching rights before a change of recommendation — the notice period and the "
        "buyer's right to negotiate and revise its offer before the target board may change its "
        "recommendation."
    ),
    "Agreement provides for matching rights in connection with FTR": (
        "Buyer matching rights in connection with a fiduciary termination — the notice and match "
        "period before the target may terminate the agreement to accept a superior proposal."
    ),
    "Tail Period & Acquisition Proposal Details": (
        "Termination-fee tail provision — the tail period after termination during which a "
        "consummated or agreed acquisition proposal triggers the termination fee, and how "
        "acquisition proposal is defined for that purpose."
    ),
    "Limitations on FTR Exercise": (
        "Limitations on the target's exercise of its fiduciary termination right to accept a "
        "superior proposal — deadlines, prior notice to the buyer, and termination-fee payment "
        "conditions."
    ),
    "FTR Triggers": (
        "What triggers the target's fiduciary termination right — the conditions, such as receipt "
        "of a superior proposal or a change of board recommendation, under which the target may "
        "terminate the merger agreement."
    ),
    "Breach of No Shop": (
        "Consequences of the target's breach of the no-shop covenant — whether a breach triggers "
        "termination-fee payment or gives the buyer a termination right."
    ),
    "Breach of Meeting Covenant": (
        "Consequences of breaching the covenant to call and hold the shareholder meeting to vote "
        "on adoption of the merger agreement."
    ),
    "Ordinary course covenant": (
        "Interim operating covenant requiring the target to conduct its business in the ordinary "
        "course, consistent with past practice, between signing and closing, and to preserve its "
        "business organization and relationships."
    ),
    "Negative interim operating covenant": (
        "Negative interim operating covenants — specific actions the target may not take between "
        "signing and closing without buyer consent, such as dividends, equity issuances, "
        "acquisitions, divestitures, or incurring debt."
    ),
    "Intervening Event Definition": (
        'Definition of "Intervening Event" — a material development or change unknown at signing '
        "(other than an acquisition proposal) that permits the target board to change its "
        "recommendation."
    ),
    "Superior Offer Definition": (
        'Definition of "Superior Proposal" or "Superior Offer" — the threshold an unsolicited '
        "alternative acquisition proposal must meet (form, financing, likelihood, favorability to "
        "shareholders) to qualify."
    ),
    "General Antitrust Efforts Standard": (
        "Efforts standard for obtaining antitrust and other regulatory approvals — reasonable "
        "best efforts, best efforts, or hell-or-high-water, including any obligation to divest "
        "assets or accept conditions."
    ),
    "Specific Performance": (
        "Specific performance and equitable remedies provision — the parties' right to an "
        "injunction or specific enforcement of the agreement rather than damages alone."
    ),
}

# ---------------------------------------------------------------------------
# Cleaning + canonicalization
# ---------------------------------------------------------------------------

# EDGAR scrape furniture: "____… 3/2/22, 6:46 PM Exhibit 21 https://www.sec.gov/… 2/97".
_FURNITURE = re.compile(
    r"(?:_{4,}\s*)?"
    r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*[AP]\.?M\.?[^\n]{0,120}?\s*"
    r"https?://\S*sec\.gov/\S+(?:\s+\d+/\d+)?",
    re.IGNORECASE,
)

_QUOTE_TRANS = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"}

# <omitted> plus its observed typo family, and "(Page N)" / "(Pages 4-5)" jump markers.
_SPAN_SPLIT = re.compile(r"<\s*omit\w*\s*>|\(Pages?\s+[\d\s,&-]+\)", re.IGNORECASE)


def clean_contract(text: str) -> str:
    """Strip BOM and EDGAR page furniture. Run FIRST: all offsets refer to this text."""
    return _FURNITURE.sub("\n\n", text.lstrip("﻿"))


def _canon_map(s: str) -> Tuple[str, List[int]]:
    """Canonical text + map from canon index back to the original char index."""
    chars: List[str] = []
    pos: List[int] = []
    for i, ch in enumerate(s):
        ch = _QUOTE_TRANS.get(ch, ch)
        if ch.isspace() or ch == "﻿":
            continue
        chars.append(ch.lower())
        pos.append(i)
    return "".join(chars), pos


def locate_spans(span_text: str, canon_doc: str, pos_map: List[int]) -> Tuple[List[Tuple[int, int]], int, int]:
    """Char ranges (in cleaned text) for each locatable excerpt of a span blob.

    Returns (ranges, matched_canon_chars, missed_canon_chars). Multiple
    occurrences resolve to the FIRST (main body precedes any exhibit copy).
    """
    ranges: List[Tuple[int, int]] = []
    matched = missed = 0
    for piece in _SPAN_SPLIT.split(span_text):
        cp, _ = _canon_map(piece)
        if len(cp) < MIN_PIECE_CANON_CHARS:
            continue
        j = canon_doc.find(cp)
        if j >= 0:
            matched += len(cp)
            ranges.append((pos_map[j], pos_map[j + len(cp) - 1] + 1))
            continue
        # Bracket recovery for long pieces broken mid-way by annotator typos or
        # residual furniture: anchor a clean prefix and suffix and take the
        # enclosing range, bounded so a false suffix hit can't claim half the doc.
        if len(cp) >= 200:
            a = canon_doc.find(cp[:80])
            b = canon_doc.find(cp[-80:], a + 80) if a >= 0 else -1
            if a >= 0 and b >= 0 and (b + 80) - a <= int(len(cp) * 1.6):
                matched += len(cp)
                ranges.append((pos_map[a], pos_map[b + 80 - 1] + 1))
                continue
        missed += len(cp)
    return ranges, matched, missed


# ---------------------------------------------------------------------------
# Contract sectionizer
# ---------------------------------------------------------------------------

_HEADING = re.compile(
    r"(?:"
    r"(?:ARTICLE|Article)\s+(?P<anum>[IVXLCDM]+|\d{1,2})(?=[\s.:;,])"
    r"|"
    r"(?:SECTION|Section)\s+(?P<knum>\d{1,2}\.\d{1,2})\b\.?"
    r"|"
    r"(?P<bnum>\d{1,2}\.\d{1,2})\.?\s+(?=[A-Z\"'])"
    r")"
)


def _preceded_by_gap(text: str, pos: int) -> bool:
    """Headings follow formatting gaps (>=2 whitespace or a line start);
    cross-references ("... pursuant to Section 2.05 ...") sit mid-sentence
    after a single space. This one test is most of the precision."""
    if pos == 0 or text[pos - 1] == "\n":
        return True
    return pos >= 2 and text[pos - 1].isspace() and text[pos - 2].isspace()


_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(s: str) -> int:
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN[ch]
        total += -v if i + 1 < len(s) and _ROMAN[s[i + 1]] > v else v
    return total


def _heading_key(m: "re.Match[str]") -> Tuple[int, int]:
    if m.group("anum") is not None:
        a = m.group("anum")
        return (int(a) if a.isdigit() else _roman_to_int(a), 0)
    num = m.group("knum") or m.group("bnum")
    major, minor = num.split(".")
    return (int(major), int(minor))


def _drop_toc_runs(starts: List[int]) -> List[bool]:
    """True for candidates inside a dense run (TOC entries sit back-to-back)."""
    n = len(starts)
    toc = [False] * n
    i = 0
    while i < n - 1:
        j = i
        while j < n - 1 and starts[j + 1] - starts[j] < TOC_RUN_GAP:
            j += 1
        if j - i + 1 >= TOC_RUN_LEN:
            for k in range(i, j + 1):
                toc[k] = True
        i = max(j, i + 1)
    return toc


def _lis_filter(keys: List[Tuple[int, int]]) -> List[int]:
    """Indices of the longest strictly-increasing subsequence of heading keys.

    Real headings run in numeric order through the document; cross-references
    jump around. O(n^2) is fine at the ~600 candidates a contract produces.
    """
    n = len(keys)
    best = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if keys[j] < keys[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                prev[i] = j
    end = max(range(n), key=lambda i: best[i]) if n else -1
    out: List[int] = []
    while end != -1:
        out.append(end)
        end = prev[end]
    return out[::-1]


def detect_contract_sections(text: str) -> List[SectionBoundary]:
    """Clause-level sections for a merger agreement; same shape as _detect_sections.

    The leading region becomes "[front matter]" (title page + TOC) and, when a
    WHEREAS clause precedes the first heading, "[recitals]" — recitals hold real
    deal points (e.g. tender-offer consideration) and must be retrievable, so
    unlike the markdown detector no section here has ``heading=None``.
    """
    matches = [m for m in _HEADING.finditer(text) if _preceded_by_gap(text, m.start())]
    if matches:
        toc = _drop_toc_runs([m.start() for m in matches])
        matches = [m for m, is_toc in zip(matches, toc, strict=True) if not is_toc]
    if matches:
        keep = set(_lis_filter([_heading_key(m) for m in matches]))
        matches = [m for i, m in enumerate(matches) if i in keep]

    boundaries: List[Tuple[int, str]] = []
    first = matches[0].start() if matches else len(text)
    whereas = text.find("WHEREAS", 0, first)
    if first > 0:
        boundaries.append((0, "[front matter]"))
        if 0 < whereas < first:
            boundaries.append((whereas, "[recitals]"))
    for m in matches:
        heading = " ".join(text[m.start() : m.start() + 90].split())
        boundaries.append((m.start(), heading))

    sections: List[SectionBoundary] = []
    for i, (start, heading) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        sections.append(SectionBoundary(index=i, heading=heading, heading_level=1, start_pos=start, end_pos=end))
    return sections


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@dataclass
class MaudBench:
    """Duck-typed like SyntheticBenchmark for the fields eval_dual consumes."""

    corpus: Dict[str, str]
    queries: Dict[str, str]
    doc_qrels: Dict[str, Dict[str, int]]
    section_qrels: Dict[str, Dict[str, int]]
    gold_spans: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    stats: Dict[str, object] = field(default_factory=dict)


def _read_pairs(data_dir: Path) -> Dict[Tuple[str, str], str]:
    csv.field_size_limit(10**9)
    pairs: Dict[Tuple[str, str], str] = {}
    for split in ("train", "dev", "test"):
        with open(data_dir / f"MAUD_{split}.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["data_type"] != "main" or row["contract_name"] == "<RARE_ANSWERS>":
                    continue
                pairs[(row["contract_name"], " ".join(row["text_type"].split()))] = row["text"]
    return pairs


def load_maud(
    max_contracts: Optional[int] = None,
    data_dir: Path = DATA_DIR,
    min_pair_coverage: float = MIN_PAIR_COVERAGE,
) -> MaudBench:
    pairs = _read_pairs(data_dir)
    contract_ids = sorted({cn for cn, _ in pairs}, key=lambda c: int(c.rsplit("_", 1)[1]))
    if max_contracts is not None:
        contract_ids = contract_ids[:max_contracts]
    keep = set(contract_ids)

    unknown_dp = {tt for _, tt in pairs if tt not in DEAL_POINT_DESCRIPTIONS}
    if unknown_dp:
        logger.warning("No description for deal points %s — using the bare name", sorted(unknown_dp))

    bench = MaudBench({}, {}, {}, {}, {}, {})
    dropped_coverage = dropped_nosection = 0
    coverages: List[float] = []
    sections_per_contract: List[int] = []

    for cid in contract_ids:
        text = clean_contract((data_dir / "contracts" / f"{cid}.txt").read_text(encoding="utf-8"))
        bench.corpus[cid] = text
        canon_doc, pos_map = _canon_map(text)
        sections = detect_contract_sections(text)
        sections_per_contract.append(len(sections))

        for (cn, tt), span_text in pairs.items():
            if cn != cid:
                continue
            ranges, matched, missed = locate_spans(span_text, canon_doc, pos_map)
            total = matched + missed
            cov = matched / total if total else 0.0
            coverages.append(cov)
            if cov < min_pair_coverage or not ranges:
                dropped_coverage += 1
                continue
            gold = {
                section_qrel_id(cid, s.index): 1
                for s in sections
                for (a, b) in ranges
                if min(b, s.end_pos) - max(a, s.start_pos) >= min(MIN_GOLD_OVERLAP_CHARS, b - a)
            }
            if not gold:
                dropped_nosection += 1
                continue
            qid = f"{cid}||{tt}"
            desc = DEAL_POINT_DESCRIPTIONS.get(tt)
            bench.queries[qid] = f"{tt}. {desc}" if desc else tt
            bench.doc_qrels[qid] = {cid: 1}
            bench.section_qrels[qid] = gold
            bench.gold_spans[qid] = ranges

    bench.stats = {
        "contracts": len(contract_ids),
        "candidate_pairs": sum(1 for cn, _ in pairs if cn in keep),
        "queries": len(bench.queries),
        "dropped_low_coverage": dropped_coverage,
        "dropped_no_gold_section": dropped_nosection,
        "mean_pair_coverage": round(statistics.mean(coverages), 4) if coverages else 0.0,
        "sections_per_contract_median": statistics.median(sections_per_contract) if sections_per_contract else 0,
    }
    logger.info("MAUD: %s", bench.stats)
    return bench


# ---------------------------------------------------------------------------
# Validation report (run me before trusting the loader on a new tweak)
# ---------------------------------------------------------------------------


def _report(max_contracts: Optional[int]) -> None:
    bench = load_maud(max_contracts=max_contracts)
    est_tokens = lambda s: len(s) // 4  # noqa: E731  (avoid importing the heavy encoder stack)

    print(f"\ncontracts: {len(bench.corpus)}  queries: {len(bench.queries)}")
    print("stats:", bench.stats)

    sec_counts: List[int] = []
    sec_tokens: List[int] = []
    for text in bench.corpus.values():
        secs = detect_contract_sections(text)
        sec_counts.append(len(secs))
        sec_tokens.extend(est_tokens(text[s.start_pos : s.end_pos]) for s in secs)
    sec_counts.sort()
    sec_tokens.sort()
    pct = lambda xs, p: xs[min(len(xs) - 1, int(len(xs) * p))]  # noqa: E731
    print(
        f"sections/contract: min {sec_counts[0]}  p10 {pct(sec_counts, 0.1)}  median {pct(sec_counts, 0.5)}"
        f"  p90 {pct(sec_counts, 0.9)}  max {sec_counts[-1]}"
    )
    print(
        f"section est-tokens: median {pct(sec_tokens, 0.5)}  p90 {pct(sec_tokens, 0.9)}"
        f"  p99 {pct(sec_tokens, 0.99)}  max {sec_tokens[-1]}"
        f"  | >2k: {sum(t > 2048 for t in sec_tokens)}/{len(sec_tokens)}"
        f"  >8k: {sum(t > 8192 for t in sec_tokens)}"
    )
    under = [(cid, len(detect_contract_sections(t))) for cid, t in bench.corpus.items()]
    weak = sorted([x for x in under if x[1] < 20], key=lambda x: x[1])
    print(f"contracts with <20 sections ({len(weak)}):", weak[:10])

    per_dp = Counter(qid.split("||", 1)[1] for qid in bench.queries)
    print("\nqueries per deal point:")
    for dp, n in per_dp.most_common():
        print(f"  {n:5d}  {dp}")
    gold_sizes = [len(v) for v in bench.section_qrels.values()]
    gold_sizes.sort()
    print(f"\ngold sections/query: median {pct(gold_sizes, 0.5)}  p90 {pct(gold_sizes, 0.9)}  max {gold_sizes[-1]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="MAUD loader validation report")
    ap.add_argument("--max-contracts", type=int, default=None)
    args = ap.parse_args()
    _report(args.max_contracts)
