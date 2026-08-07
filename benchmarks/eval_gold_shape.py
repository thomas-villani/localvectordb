"""How section-shaped is a corpus's gold? The control for a chunks-vs-sections claim.

A section-level index gets an **unearned** win on any corpus where the gold span
*is* essentially the whole section: "retrieve the section" and "retrieve the
answer" become the same act, and the comparison stops arbitrating anything.
``span-length-crossover-findings`` §6.14 raised this about qasper and it was
never measured -- so every section-vs-chunk verdict since has rested on an
assumption.

This measures it: for each ``(query, gold section)`` pair, the fraction of the
owning section's characters that the located gold span covers. Read the tail
(``>99%``), not the median -- a section the gold *is* costs the comparison its
meaning, a section the gold sits inside does not.

Measured 2026-08-07 (dev splits, real ``SectionDetector``):

=======================================  =========  =========
statistic                                qasper     NQ
=======================================  =========  =========
(query, gold-section) pairs                 1,230      1,619
median coverage                             0.516      0.331
gold covers >80% of its section              36.1%      19.1%
gold covers >99% of its section              18.0%       3.6%
=======================================  =========  =========

So NQ is the *harder* corpus for a section index by a factor of five at the
tail, and qasper's section win (+0.0333 on vector, ``SYNTHESIS-v2`` §5) is
measured on partly favourable ground: on ~18% of its pairs the gold and the
section are the same text. That does not refute the win -- the effect is far
larger than 18% of the pairs could explain on their own -- but it is the reason
to want the same measurement on a corpus that does not share the flaw.

MAUD is absent by construction: its gold is a clause span located by a bespoke
detector, and ``detect_contract_sections`` returns spans whose relationship to
the annotation is the thing under study rather than an input to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

# Coverage thresholds reported. 0.99 is the one that matters: at that point the
# section carries no text the gold does not already carry.
THRESHOLDS = (0.5, 0.8, 0.9, 0.99)


def _qasper_coverage(max_papers: Optional[int]) -> Tuple[List[float], Dict[str, float]]:
    """Gold-chars / owning-section-chars for every qasper (query, gold-section) pair."""
    from benchmarks.qasper_data import _evidence_strings, _owner_section_index, _render_paper, download
    from localvectordb.section_detection import SectionDetector

    root = download()
    data = json.loads(sorted(root.glob("*dev*.json"))[0].read_text(encoding="utf-8"))
    detector = SectionDetector()

    cover: List[float] = []
    sec_lens: List[int] = []
    doc_lens: List[int] = []
    for pid in sorted(data)[:max_papers]:
        text, para_spans = _render_paper(data[pid])
        detected = detector.detect_sections(text)
        by_index = {s.index: s for s in detected}
        sec_lens.extend(s.end_pos - s.start_pos for s in detected if s.heading)
        doc_lens.append(len(text))

        for qa in data[pid].get("qas") or []:
            # Chars of located evidence per gold section, mirroring load_qasper's
            # attribution exactly -- a different rule here would measure this
            # script rather than the corpus.
            per_section: Dict[int, int] = {}
            for ev in _evidence_strings(qa):
                span = para_spans.get(ev) or para_spans.get(ev.strip())
                if span is None:
                    continue
                idx = _owner_section_index(detected, span)
                if idx is not None:
                    per_section[idx] = per_section.get(idx, 0) + (span[1] - span[0])
            for idx, chars in per_section.items():
                sec = by_index[idx]
                cover.append(chars / max(sec.end_pos - sec.start_pos, 1))

    return cover, {"sections": float(np.median(sec_lens)), "docs": float(np.median(doc_lens))}


def _nq_coverage(max_queries: Optional[int], shards: Optional[Sequence[int]]) -> Tuple[List[float], Dict[str, float]]:
    """The same statistic on Natural Questions, via its long-answer token spans."""
    from benchmarks.nq_data import _char_span, _owner_section, _require_pyarrow, download, render_document
    from localvectordb.section_detection import SectionDetector

    pq = _require_pyarrow()
    detector = SectionDetector()

    cover: List[float] = []
    sec_lens: List[int] = []
    doc_lens: List[int] = []
    for path in download(shards=shards):
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                if max_queries is not None and len(cover) >= max_queries:
                    return cover, {"sections": float(np.median(sec_lens)), "docs": float(np.median(doc_lens))}
                gold = next((la for la in row["annotations"]["long_answer"] if la["candidate_index"] >= 0), None)
                if gold is None:
                    continue
                text, spans, _ = render_document(row["document"]["tokens"])
                headed = [s for s in detector.detect_sections(text) if s.heading]
                sec_lens.extend(s.end_pos - s.start_pos for s in headed)
                doc_lens.append(len(text))
                span = _char_span(spans, gold["start_token"], gold["end_token"])
                if span is None:
                    continue
                owner = _owner_section(headed, *span)
                if owner is None:
                    continue
                cover.append((span[1] - span[0]) / max(owner.end_pos - owner.start_pos, 1))

    return cover, {"sections": float(np.median(sec_lens)), "docs": float(np.median(doc_lens))}


def report(name: str, cover: Sequence[float], meds: Dict[str, float]) -> Dict[str, float]:
    c = np.asarray(cover, dtype=float)
    out = {
        "pairs": int(c.size),
        "median_coverage": float(np.median(c)),
        "mean_coverage": float(c.mean()),
        "median_section_chars": meds["sections"],
        "median_doc_chars": meds["docs"],
    }
    print(f"\n=== {name} ===")
    print(f"  (query, gold-section) pairs   {c.size:,}")
    print(f"  median section chars          {meds['sections']:,.0f}")
    print(f"  median doc chars              {meds['docs']:,.0f}")
    print(f"  coverage  median {np.median(c):.3f}   mean {c.mean():.3f}")
    for t in THRESHOLDS:
        share = float(np.mean(c > t))
        out[f"share_over_{int(t * 100)}"] = share
        flag = "  <-- gold IS the section" if t == 0.99 and share > 0.10 else ""
        print(f"  gold covers >{t:.0%} of its section  {100 * share:5.1f}%{flag}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=("qasper", "nq"), default=["qasper", "nq"])
    p.add_argument("--limit", type=int, default=None, help="cap papers (qasper) / gold pairs (nq)")
    p.add_argument("--shards", type=int, nargs="+", default=None, help="NQ dev shards to read (default all 6)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    results: Dict[str, Dict[str, float]] = {}
    for name in args.datasets:
        if name == "qasper":
            cover, meds = _qasper_coverage(args.limit)
        else:
            cover, meds = _nq_coverage(args.limit, args.shards)
        if not cover:
            raise SystemExit(f"{name}: no (query, gold-section) pairs -- nothing to measure")
        results[name] = report(name, cover, meds)

    if len(results) > 1:
        print("\nRead the >99% row: a corpus high there cannot arbitrate chunks-vs-sections,")
        print("because retrieving the section and retrieving the answer are the same act.")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
