"""DIAGNOSTIC: do our two retrieval mechanisms rank identically on identical vectors?

WHY THIS EXISTS. ``eval_dual`` and ``eval_section_finder`` disagree on one number
-- the Qasper chunk arm rolled up to sections, 0.1838 vs 0.2164 -- while agreeing
on the section arm to 0.0015. Forensics eliminated every input as the cause:

    chunks 3,155 == 3,155      sections 3,974 == 3,974
    distinct chunk owners 2,380 == 2,380       gold reachability 73.7% == 73.7%
    aggregation max == max     metric ndcg_at_k@10 == ndcg_at_k@10
    document vectors cos = 1.000000 (bit-identical, correctly single-prefixed)
    query vectors cached under the QUERY prefix, 200/200

That leaves a paradox rather than a suspect. ``eval_dual`` pools **exhaustively**
over exact cosine -- every section is a candidate -- while ``eval_section_finder``
fetches ``k*fetch`` chunks and rolls up only those. Exhaustive must UPPER-BOUND
fetch-limited: a truncated fetch can only miss good sections, never invent better
ones. The exhaustive arm scoring *lower* is therefore not a tuning difference; one
of the two procedures is not doing what its code appears to say.

WHAT THIS FILE DOES. It runs both procedures over the SAME stored vectors and
diffs them **per query**. The mean is deliberately not the output: a 0.033 mean
gap could be a handful of catastrophic queries or a uniform drift, and those imply
completely different bugs. Only the per-query distribution separates them.

WHY IT NEEDS NO ENCODER. Chunk vectors are read straight out of the built FAISS
index; query vectors are read out of ``eval_dual``'s on-disk cache. The provider
is replaced by a stub that serves that cache and REFUSES to embed anything
(``--allow-embed`` is not offered on purpose). So this runs while a build is
occupying Ollama, and -- more importantly -- the query vector is pinned to one
value across both paths, removing the encoder as a variable entirely rather than
assuming it is one.

The stub also answers a question we could not otherwise see: it records which
prefix the text arrived with, so we learn whether ``src/`` prefixes before or
after handing text to a provider. A cache miss here is a FINDING, not an error.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.config import CACHE_DIR  # noqa: E402
from benchmarks.metrics import ndcg_at_k  # noqa: E402

logger = logging.getLogger("eval_equivalence")

K = 10
# EmbeddingGemma's trained templates. Duplicated from src/'s registry ON PURPOSE:
# if src/ ever changes them, this file must fail to find cached vectors rather
# than silently follow the change -- the cache was written under these strings.
EGEMMA_QUERY_PREFIX = "task: search result | query: "
EGEMMA_DOC_PREFIX = "title: none | text: "


class CacheBackedProvider:
    """Serves ``eval_dual``'s vector cache; embeds nothing, ever.

    Wrapping the real provider class would drag in model validation and a live
    base_url. What the query path actually touches is small, so this implements
    that surface directly and raises on anything else -- an unexpected call is a
    finding about the query path, and should stop the run rather than fall back.
    """

    def __init__(self, model: str, cache_dir: Path) -> None:
        self.model = model
        self._dir = cache_dir
        self.prefix_seen: Dict[str, int] = {"as-given": 0, "query-prefixed": 0, "doc-prefixed": 0}
        self.misses: List[str] = []
        self.collisions = 0

    def _path(self, text: str) -> Path:
        h = hashlib.sha256(f"{self.model}\x00{text}".encode("utf-8")).hexdigest()
        return self._dir / f"{h}.npy"

    def lookup(self, text: str) -> Optional[np.ndarray]:
        """Resolve a text to a cached vector, recording which prefixing it arrived with.

        ORDER IS LOAD-BEARING, and finding that out was the first result this
        harness produced. The cache holds a bare AND a query-prefixed vector for
        the same query text under different keys, because the prefix is applied
        before hashing. So "look up the query" has no single answer: a harness
        that applies no prefix silently receives the un-prefixed vector and
        measures a different -- worse -- retrieval system, with no cache miss and
        no warning anywhere. The query-prefixed vector is the correct one for an
        asymmetric model, so it is preferred here, and every collision is counted
        rather than quietly resolved.
        """
        candidates = (
            ("query-prefixed", EGEMMA_QUERY_PREFIX + text),
            ("doc-prefixed", EGEMMA_DOC_PREFIX + text),
            ("as-given", text),
        )
        found = [(label, self._path(c)) for label, c in candidates if self._path(c).exists()]
        if len(found) > 1:
            self.collisions += 1
        if not found:
            self.misses.append(text[:80])
            return None
        label, path = found[0]
        self.prefix_seen[label] += 1
        return np.load(path).astype(np.float32).ravel()

    # --- the surface LocalVectorDB's query path uses -------------------------
    @property
    def provider_name(self) -> str:
        return "cache-stub"

    @property
    def max_batch_size(self) -> int:
        return 64

    def get_dimension(self) -> int:
        return 768

    def validate_model(self) -> bool:
        return True

    def embed_sync(self, texts: Sequence[str], task: str = "query") -> List[List[float]]:
        out = []
        for t in texts:
            v = self.lookup(t)
            if v is None:
                raise KeyError(f"no cached vector for {t[:80]!r}; this harness must not embed")
            out.append(v.tolist())
        return out

    async def embed_async(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed_sync(texts)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed_sync(texts)

    # src/ prefixes inside the provider base class. This stub deliberately does
    # NOT, because lookup() resolves the query-prefixed vector itself -- so the
    # text arriving here is raw and the cached vector is still the prefixed one.
    def apply_prefix(self, texts: Sequence[str], task: str = "query") -> List[str]:
        return list(texts)

    @property
    def uses_prefixes(self) -> bool:
        return False


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def load_chunk_space(db_base: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """(chunk_vectors, section_qrel_id per chunk, valid mask) from a built index.

    Chunks whose ``section_id`` is NULL cannot be rolled up by either procedure
    and are dropped from both, so the comparison is not contaminated by a
    difference in how each side handles them.
    """
    import faiss

    idx = faiss.read_index(f"{db_base}.faiss")
    conn = sqlite3.connect(f"{db_base}.sqlite")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT c.faiss_id AS fid, s.document_id AS doc, s.section_index AS sidx "
        "FROM chunks c JOIN sections s ON c.section_id = s.id "
        "WHERE c.faiss_id IS NOT NULL"
    ).fetchall()
    conn.close()

    vecs = np.vstack([idx.reconstruct(int(r["fid"])) for r in rows]).astype(np.float32)
    owners = [f"{r['doc']}#s{r['sidx']}" for r in rows]
    return _unit(vecs), owners, np.asarray([int(r["fid"]) for r in rows])


def numpy_rollup(
    qvec: np.ndarray,
    chunk_vecs: np.ndarray,
    owners: Sequence[str],
    *,
    fetch: Optional[int],
) -> List[str]:
    """Max-pool chunk scores into sections. ``fetch=None`` is exhaustive (eval_dual).

    ``fetch=n`` keeps only the top ``n`` chunks first, which is the shape of
    ``eval_section_finder``'s call. Running both from one function is the point:
    any divergence is then attributable to fetch depth alone, because nothing
    else about the two paths differs here.
    """
    sims = chunk_vecs @ _unit(qvec)
    order = np.argsort(-sims)
    if fetch is not None:
        order = order[:fetch]
    best: Dict[str, float] = {}
    for j in order:
        sec = owners[j]
        s = float(sims[j])
        if sec not in best or s > best[sec]:
            best[sec] = s
    return [s for s, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="benchmarks/data/db/hiergate__qasper__embeddinggemma-300m__centroid")
    p.add_argument("--model", default="embeddinggemma:300m")
    p.add_argument("--cache", default="ollama__embeddinggemma-300m__ctx2048")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--fetches", type=int, nargs="+", default=[10, 100])
    p.add_argument("--src-path", action="store_true", help="also drive src's db.query with the stub provider")
    args = p.parse_args(argv)

    from benchmarks.qasper_data import load_qasper

    bench = load_qasper(split="dev", max_papers=None)
    stub = CacheBackedProvider(args.model, CACHE_DIR / "hier_embed" / args.cache)

    chunk_vecs, owners, _ = load_chunk_space(args.db)
    logger.info("chunk space: %d chunks over %d distinct sections", len(owners), len(set(owners)))

    qids = list(bench.queries)
    if args.max_queries:
        qids = qids[: args.max_queries]

    src_db = None
    if args.src_path:
        from benchmarks.eval_hier_gate import _section_qrel_id  # noqa: F401
        from localvectordb import LocalVectorDB

        path = Path(args.db)
        # Construct against MOCK, never the real provider. Opening with
        # ``embedding_provider="ollama"`` costs two live embed calls at import
        # time (a validate and a dimension probe) -- which took 118s here, because
        # it forced Ollama to reload the model at a different num_ctx than the
        # build occupying it. This harness must not perturb a running build.
        src_db = LocalVectorDB(
            path.name,
            path.parent,
            embedding_provider="mock",
            embedding_model=args.model,
            embedding_config={"dimension": stub.get_dimension()},
            hierarchical_embeddings=True,
            section_vector_strategy="centroid",
        )
        # Pin the query vector to the SAME one the numpy arm uses. Without this the
        # comparison silently re-introduces the encoder as a variable. The public
        # attribute is read-only, so this reaches past it deliberately.
        src_db._embedding_provider = stub
        logger.info("src path enabled; provider swapped for the cache stub")

    arms: Dict[str, List[float]] = {}
    if src_db is not None:
        for st in ("hybrid", "vector"):
            arms.update({f"src {st} x{f // K}": [] for f in args.fetches})
    arms.update({f"numpy fetch x{f // K}": [] for f in args.fetches})
    arms["numpy exhaustive"] = []
    per_query: Dict[str, Dict[str, float]] = {}
    skipped = 0

    for qid in qids:
        rel = bench.section_qrels.get(qid, {})
        if not any(v > 0 for v in rel.values()):
            continue
        qvec = stub.lookup(bench.queries[qid])
        if qvec is None:
            skipped += 1
            continue
        row: Dict[str, float] = {}
        if src_db is not None:
            for f in args.fetches:
                for stype in ("hybrid", "vector"):
                    hits = src_db.query(
                        bench.queries[qid],
                        k=f,
                        return_type="sections",
                        search_level="chunks",
                        search_type=stype,
                    )
                    ranked = [_section_qrel_id(h.id) for h in hits][:K]
                    row[f"src {stype} x{f // K}"] = ndcg_at_k(ranked, rel, K)
        for f in args.fetches:
            row[f"numpy fetch x{f // K}"] = ndcg_at_k(numpy_rollup(qvec, chunk_vecs, owners, fetch=f), rel, K)
        row["numpy exhaustive"] = ndcg_at_k(numpy_rollup(qvec, chunk_vecs, owners, fetch=None), rel, K)
        for label, v in row.items():
            arms[label].append(v)
        per_query[qid] = row

    print(f"\nscored {len(per_query)} queries; {skipped} skipped (no cached query vector)")
    if stub.misses:
        print(f"  NOTE {len(stub.misses)} cache misses -- e.g. {stub.misses[0]!r}")
    print(f"  prefix resolution: {stub.prefix_seen}")
    if stub.collisions:
        print(
            f"  !! {stub.collisions} texts had MORE THAN ONE cached vector "
            "(bare and prefixed both present).\n"
            "     Which one a harness gets depends on the prefix it applies -- "
            "silently, with no cache miss."
        )

    print(f"\n{'arm':<24}{'ndcg@10':>10}")
    for label, vals in arms.items():
        if vals:
            print(f"{label:<24}{sum(vals) / len(vals):>10.4f}")

    print("\nPublished, for reference:   eval_dual 0.1838    eval_section_finder (fetch x10) 0.2164")

    # The per-query view: a mean gap can hide either a few catastrophes or a
    # uniform drift, and the fix differs completely between those two worlds.
    base = "numpy exhaustive"
    for label in arms:
        if label == base or not arms[label]:
            continue
        d = np.array([per_query[q][label] - per_query[q][base] for q in per_query])
        worse = int((d < -1e-9).sum())
        better = int((d > 1e-9).sum())
        print(
            f"\n{label} vs {base}: mean {d.mean():+.4f}, "
            f"{better} queries better, {worse} worse, {len(d) - better - worse} identical"
        )
        if better:
            print(f"  max gain {d.max():+.4f}   (fetch-limited beating exhaustive is IMPOSSIBLE if both are max-pool)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
