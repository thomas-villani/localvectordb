"""MLDR (en): long-document retrieval with NO detectable section structure.

The third corpus. MAUD (numbered contract clauses) and Qasper (canonical paper
headings) were both chosen *because* they were cleanly sectioned, so every
finding in ``span-length-crossover-findings`` §6.20-§6.27 rests on reliable
heading detection. Measured against the real ``SectionDetector``, MLDR-en has
**0/800 documents with any detected heading** -- its Wikipedia markup is stripped
to flat prose. QuALITY is the same (0/230). Structure that clean is the
exception, not the rule, and this loader exists to find out which findings
survive without it.

What MLDR can and cannot test (§6.28.1):

* **Granularity** -- YES, at DOCUMENT level. The rawspan/centroid ladder needs
  only spans and document-level gold, not headings. And MLDR lands squarely in
  the regime that matters: **55.8% of its documents exceed 8,192 band-tokens**,
  where F2 runs -0.25 to -0.36. Qasper has almost none of that regime and MAUD is
  entirely inside it; MLDR straddles.
* **Structure alignment** and **weighting** -- NO. No headings means no sections,
  and gold is document-level only.

``section_qrels`` is therefore returned **empty**, deliberately. Callers that
need section-level gold must not use this corpus; a fabricated section structure
here would be spans with invented relevance, which is exactly the circularity
§6.14 flagged on Qasper.

The dev split embeds the full text of every positive and negative passage, so a
self-contained 1,585-document / 200-query benchmark builds from a **21 MB**
download -- the 1 GB ``corpus.jsonl.gz`` is not needed.

**Validity caveat, carried in the docstring because it is easy to forget:** MLDR
queries are **LLM-generated**, not user queries. A result here is weaker evidence
than the document count suggests.

No ``datasets`` dependency: plain gzipped JSONL fetched directly, matching
``qasper_data`` and ``beir_data``.

Source: Chen et al., BGE-M3 (``Shitao/MLDR``, CC-BY-3.0 -- Wikipedia-derived).
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.config import DATA_DIR  # noqa: E402
from benchmarks.superdocs import SyntheticBenchmark  # noqa: E402

logger = logging.getLogger(__name__)

_BASE = "https://huggingface.co/datasets/Shitao/MLDR/resolve/main/mldr-v1.0-en"
SPLIT_FILES = {"dev": "dev.jsonl.gz", "test": "test.jsonl.gz"}


def download(split: str = "dev", *, data_dir: Optional[Path] = None, force: bool = False) -> Path:
    """Fetch one MLDR-en split to the benchmark data dir; return the local path."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown MLDR split {split!r}; expected one of {sorted(SPLIT_FILES)}")
    root = (data_dir or DATA_DIR) / "mldr"
    root.mkdir(parents=True, exist_ok=True)
    dest = root / SPLIT_FILES[split]
    if dest.exists() and not force:
        return dest

    import requests

    url = f"{_BASE}/{SPLIT_FILES[split]}"
    logger.info("Downloading MLDR-en %s from %s", split, url)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)  # atomic: a killed download must not look complete
    logger.info("Wrote %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def load_mldr(
    *,
    split: str = "dev",
    data_dir: Optional[Path] = None,
    max_queries: Optional[int] = None,
    include_negatives: bool = True,
) -> SyntheticBenchmark:
    """Load MLDR-en as a document-level retrieval benchmark.

    Parameters
    ----------
    split
        ``"dev"`` (200 queries, 21 MB) or ``"test"`` (82 MB).
    max_queries
        Cap the query count, deterministically (file order). The corpus is built
        from the passages those queries reference, so capping shrinks both.
    include_negatives
        Keep each query's hard negatives in the corpus. **Default True, and
        turning it off makes the task trivially easy** -- the corpus would then
        contain only documents that are gold for some query.
    """
    path = download(split, data_dir=data_dir)

    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    doc_qrels: Dict[str, Dict[str, int]] = {}

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["query_id"]
            positives = row.get("positive_passages") or []
            if not positives:
                continue  # no gold => contributes nothing but noise
            queries[qid] = row["query"]
            doc_qrels[qid] = {}
            for p in positives:
                corpus[p["docid"]] = p["text"]
                doc_qrels[qid][p["docid"]] = 1
            if include_negatives:
                for p in row.get("negative_passages") or []:
                    corpus.setdefault(p["docid"], p["text"])
            if max_queries is not None and len(queries) >= max_queries:
                break

    logger.info(
        "MLDR-en %s: %d docs, %d queries (negatives=%s)",
        split,
        len(corpus),
        len(queries),
        include_negatives,
    )
    return SyntheticBenchmark(
        name=f"mldr-en-{split}",
        params={
            "split": split,
            "include_negatives": include_negatives,
            "structure": "none (0/800 docs have a detected heading)",
            "queries": "LLM-generated",
        },
        corpus=corpus,
        queries=queries,
        doc_qrels=doc_qrels,
        # Deliberately empty: MLDR has no headings and no span-level gold, so any
        # section structure here would be invented. See the module docstring.
        section_qrels={},
        passage_qrels={},
    )


def main() -> int:  # pragma: no cover - manual characterisation entry point
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    import numpy as np

    bench = load_mldr(split="dev")
    lengths = np.array([len(t) for t in bench.corpus.values()])
    tokens = lengths / 3.5
    print(f"docs {len(bench.corpus):,}  queries {len(bench.queries):,}")
    print(f"doc chars   med {np.median(lengths):,.0f}  p90 {np.percentile(lengths, 90):,.0f}  max {lengths.max():,.0f}")
    print(f"est tokens  med {np.median(tokens):,.0f}  >8192: {100 * np.mean(tokens > 8192):.1f}%")
    print(f"gold/query  med {np.median([len(v) for v in bench.doc_qrels.values()]):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
