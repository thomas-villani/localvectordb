"""BEIR dataset download and loading for retrieval evaluation.

Two small, fully-judged BEIR datasets are supported. Both are downloaded as a
zip from the BEIR mirror and cached under ``benchmarks/data/`` (gitignored).

``scifact``
    5,183 abstracts, 300 test queries, binary relevance. The default. Published
    nDCG@10 numbers exist for common sentence-transformers models, which lets us
    sanity-check the harness itself rather than trusting it blind.

``nfcorpus``
    3,633 documents, 323 test queries, *graded* relevance (0/1/2). Useful
    precisely because it exercises the graded-gain path in ``metrics.ndcg_at_k``
    that SciFact's binary qrels leave untested.

``fiqa``
    57,638 passages, 648 test queries, binary relevance. Financial-QA passages;
    many passages per topic, which is what the hierarchical synthetic super-doc
    builder (``benchmarks/superdocs.py``) needs to seed a gold passage and fill
    the rest of a document with genuinely-related distractors.

Only the ``test`` split is loaded. BEIR's ``queries.jsonl`` holds every split's
queries, so the qrels file -- not the queries file -- decides which queries are
evaluated.
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .config import BEIR_BASE_URL, DATA_DIR

logger = logging.getLogger(__name__)

DATASETS = ("scifact", "nfcorpus", "fiqa")


@dataclass(frozen=True)
class BeirDataset:
    """A BEIR test split.

    Attributes
    ----------
    name
        Dataset name, e.g. ``"scifact"``.
    corpus
        ``document id -> document text`` (title and body already joined).
    queries
        ``query id -> query text``, restricted to the queries that ``qrels``
        actually judges.
    qrels
        ``query id -> {document id -> relevance grade}``.
    """

    name: str
    corpus: Dict[str, str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        judged = sum(len(v) for v in self.qrels.values())
        return (
            f"BeirDataset({self.name!r}: {len(self.corpus)} docs, " f"{len(self.queries)} queries, {judged} judgements)"
        )


def _safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    """Extract ``archive`` into ``dest``, refusing entries that escape it.

    ``ZipFile.extractall`` will happily honour ``../`` and absolute paths in
    member names, so a hostile archive can write anywhere the process can. The
    BEIR mirror is not hostile, but this runs against a URL and the check is two
    lines.
    """
    dest_root = dest.resolve()
    for member in archive.namelist():
        target = (dest_root / member).resolve()
        if not target.is_relative_to(dest_root):
            raise ValueError(f"Refusing zip entry that escapes the destination: {member!r}")
    archive.extractall(dest_root)


def download(name: str, *, data_dir: Optional[Path] = None, force: bool = False) -> Path:
    """Download and extract a BEIR dataset. Returns the extracted directory."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}")

    root = (data_dir or DATA_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / name
    if (extracted / "corpus.jsonl").exists() and not force:
        logger.info("%s already cached at %s", name, extracted)
        return extracted

    import httpx

    url = f"{BEIR_BASE_URL}/{name}.zip"
    archive = root / f"{name}.zip"
    logger.info("Downloading %s ...", url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
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

    with zipfile.ZipFile(archive) as zf:
        _safe_extract(zf, root)
    archive.unlink()
    logger.info("Extracted to %s", extracted)
    return extracted


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _join_title_and_body(record: dict) -> str:
    """Join a BEIR corpus record into one document.

    BEIR's own baselines concatenate title and body with a single space. We use a
    blank line instead: this library chunks by sentence before embedding, and a
    bare space glues an unpunctuated title onto the first sentence of the
    abstract. The retrieval effect is small, but it is a deliberate deviation --
    do not "fix" it back to a space and then compare against published tables.
    """
    title = (record.get("title") or "").strip()
    text = (record.get("text") or "").strip()
    return f"{title}\n\n{text}".strip() if title else text


def load(name: str, *, data_dir: Optional[Path] = None, max_docs: Optional[int] = None) -> BeirDataset:
    """Load the ``test`` split of a BEIR dataset, downloading it if needed.

    Parameters
    ----------
    max_docs
        Truncate the corpus for a smoke test. **Never** use this for a baseline:
        shrinking the corpus removes distractors and inflates every metric.
        Documents named in the qrels are kept regardless, so the run still
        produces non-zero scores.
    """
    path = download(name, data_dir=data_dir)

    qrels: Dict[str, Dict[str, int]] = {}
    with open(path / "qrels" / "test.tsv", encoding="utf-8") as fh:
        header = next(fh)
        if not header.lower().startswith("query-id"):
            raise ValueError(f"Unexpected qrels header in {path}: {header!r}")
        for line in fh:
            if not line.strip():
                continue
            query_id, doc_id, grade = line.split("\t")[:3]
            qrels.setdefault(query_id, {})[doc_id] = int(grade)

    corpus = {r["_id"]: _join_title_and_body(r) for r in _read_jsonl(path / "corpus.jsonl")}
    queries = {r["_id"]: r["text"] for r in _read_jsonl(path / "queries.jsonl") if r["_id"] in qrels}

    missing = {d for rel in qrels.values() for d in rel} - set(corpus)
    if missing:
        raise ValueError(f"{name}: {len(missing)} judged documents are absent from the corpus")
    unanswerable = set(qrels) - set(queries)
    if unanswerable:
        raise ValueError(f"{name}: {len(unanswerable)} judged queries are absent from queries.jsonl")

    if max_docs is not None and max_docs < len(corpus):
        judged = {d for rel in qrels.values() for d in rel}
        keep = list(judged) + [d for d in corpus if d not in judged]
        corpus = {d: corpus[d] for d in keep[: max(max_docs, len(judged))]}
        logger.warning("Corpus truncated to %d docs -- results are NOT a valid baseline", len(corpus))

    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)
