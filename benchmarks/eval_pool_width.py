"""DIAGNOSTIC: does the hybrid pool-width win survive on the real ``db.query()`` path?

WHAT WAS FOUND, AND WHY IT IS NOT YET A RESULT. §19.3 measured the candidate pool
that hybrid fusion scores over and found the shipped width is too narrow: at the
default ``k=10``, ``_hybrid_pool_size`` fetches 40, while 100-200 is worth about
+0.008 nDCG@10. That was measured on **numpy captures** -- a re-implementation of
retrieval, not retrieval. Every cross-level claim in this study that was measured
on an instrument rather than on the product has needed correcting at least once
(§14/§15, the hybrid-default confound, the fetch asymmetry in eval_equivalence),
so a numpy-only finding is a hypothesis about the product, not a measurement of
it. This harness runs the same sweep through ``LocalVectorDB.query()``.

WHY POOL WIDTH IS A SCORING KNOB, NOT A COST KNOB. With
``return_type="documents"`` the chunk->document aggregator runs over the WHOLE
fused pool and only then truncates to ``k``. So the pool is an argument to the
aggregator: "mean of the top 3 children" means something different at 40 than at
400. That is why pool and ``document_scoring_method`` are swept TOGETHER here and
must be decided together -- §19.3's other half is that the shipped
``frequency_boost`` gains nothing by 100 and significantly LOSES by 400, while
``best`` gains and plateaus. Widening the pool alone would cancel its own gain.

WHY THE VECTOR LEG IS NOT SWEPT. On a vector-only leg, widening moved ``max`` by
exactly +0.0000 at every step (§19.3), because the effect is the zero-fill in
``_relative_score_fusion`` -- a keyword-only chunk scores 0.0 on the vector leg
and is indistinguishable from worst-in-pool -- and that mechanism only exists
when there are two legs. A vector arm here would be a control on a null.

THE VALIDATION THAT MAKES IT TRUSTWORTHY. The sweep works by substituting
``_search._hybrid_pool_size``. ``--verify`` runs the shipped width twice, once
through the substitution and once with the real function untouched, and requires
the two to agree **per query, exactly**. If the patch mechanism is not inert at
the shipped width, every number it produces downstream is about the patch.

Zero embedding: the provider is swapped for a cache-backed stub that raises on a
miss, so a text this harness has not seen before stops the run instead of
quietly costing money.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Running any file in `benchmarks/` as a script puts that directory on sys.path,
# where `benchmarks/datasets.py` SHADOWS HuggingFace `datasets`. sentence_transformers
# imports `datasets`, gets the shadow, and dies on its relative import; that
# ImportError is then swallowed by `SentenceTransformerEmbeddings.validate_model`,
# which returns False, and LocalVectorDB reports
# ``ValueError: Embedding model 'all-MiniLM-L6-v2' is not available``. The message
# names the model, so it reads as a missing download or a rate limit -- it is
# neither, and the same run succeeds when imported rather than executed. Dropping
# the script directory is the fix; `benchmarks.*` still resolves from the root.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]

from benchmarks.config import CACHE_DIR, DATA_DIR  # noqa: E402
from benchmarks.eval_equivalence import CacheBackedProvider  # noqa: E402
from benchmarks.eval_section_bm25 import paired  # noqa: E402
from benchmarks.metrics import ndcg_at_k, recall_at_k  # noqa: E402

logger = logging.getLogger("pool_width")

K = 10

# Shipped width at the default k -- `max(k, min(k*4, 100))` with k=10. Every
# comparison below is against this, because it is what a user gets today.
SHIPPED_POOL = 40
POOLS = (40, 100, 200, 400)

# `auto` is excluded on purpose: it resolves to `frequency_boost` for hybrid, so
# sweeping it would silently duplicate an arm under a second name.
METHODS = ("frequency_boost", "best", "average")

# ``hierarchical`` is a property of the BUILD, not a preference: reopening an
# index with the wrong value gives a DB whose section/document FAISS files do not
# match what it thinks it has. It is False for MAUD deliberately -- src/'s
# markdown ``SectionDetector`` cannot express a contract's sections (MAUD's qrels
# come from ``maud_data.detect_contract_sections``), so section vectors would be
# over texts the vector cache has never seen. This harness only searches chunks,
# so it does not need them.
CORPORA: Dict[str, Dict[str, Any]] = {
    "qasper": {
        "db": "hiergate__qasper__embeddinggemma-300m__centroid",
        "model": "embeddinggemma:300m",
        "cache": "ollama__embeddinggemma-300m__ctx2048",
        "dimension": 768,
        "hierarchical": True,
    },
    # Train+dev: 1,088 papers / 13,503 chunks / 2,940 queries, versus dev's
    # 275 / 3,155 / 882. A SEPARATE leg under its own key, never a widened
    # `qasper` -- every qasper number in SYNTHESIS-v2 is dev-only, and regrading
    # them against a 4x corpus under the same name would make that document
    # incomparable to itself. The key follows eval_hier_gate's `hiergate__`
    # convention because the index is built the same way (live local Ollama at
    # ingest, cache stub at query time), so either harness can open it.
    "qasper_full": {
        "db": "hiergate__qasper_full__embeddinggemma-300m__centroid",
        "model": "embeddinggemma:300m",
        "cache": "ollama__embeddinggemma-300m__ctx2048",
        "dimension": 768,
        "hierarchical": True,
        # The encoder runs on this box, so this corpus may be INGESTED live
        # (--embed) instead of from cache. The exact tag matters: the vector
        # cache keys on the model string, so `:latest` would orphan every vector
        # banked under `:300m` while looking like a cold cache.
        "ollama": True,
    },
    "nq": {
        "db": "poolwidth__nq2000__text-embedding-3-small",
        "model": "text-embedding-3-small",
        "cache": "openai__text-embedding-3-small",
        "dimension": 1536,
        "hierarchical": False,
    },
    # NFCorpus is the corpus commit 54a9898 chose for aggregation work precisely
    # because it aggregates (SciFact averages 1.09 chunks/doc and barely does).
    # Its index is already built by the retrieval gate and MiniLM runs locally for
    # free, so this arm needs neither a build nor the cache stub.
    "nfcorpus": {
        "db": "nfcorpus__all-MiniLM-L6-v2__IndexFlatL2__chunk_overlap=1__chunk_size=500__chunking_method=sentences",
        "provider": "sentence_transformers",
        "model": "all-MiniLM-L6-v2",
        "index_type": "IndexFlatL2",
        "chunking": {"chunking_method": "sentences", "chunk_size": 500, "chunk_overlap": 1},
        "hierarchical": False,
    },
    "maud": {
        "db": "poolwidth__maud__text-embedding-3-small",
        "model": "text-embedding-3-small",
        "cache": "openai__text-embedding-3-small",
        "dimension": 1536,
        "hierarchical": False,
    },
}

# MAUD cannot answer this question, and the index builds fine anyway -- which is
# exactly why the refusal is here rather than in a comment. Measured 2026-08-11:
# its 2,752 queries carry only **22 distinct query texts**, each repeated for up
# to 150 different contracts with a different single gold document each time. So
# an unscoped document ranking is not merely hard, it is unrankable: one text has
# 150 mutually exclusive right answers. `eval_aggregation` handles this by scoring
# MAUD on the SECTION target only, with every query scoped to its own contract.
# That option does not exist on the real query path: src/'s markdown
# `SectionDetector` cannot express a contract's sections (MAUD's qrels come from
# `maud_data.detect_contract_sections`), and shipping a narrow contract finder was
# ruled out. The corpus axis has to be widened with a corpus that has a real
# document target -- NQ, which is also where the numpy study's document cells are.
_REFUSED = {
    "maud": (
        "MAUD has 22 distinct query texts over 2,752 queries, each repeated across up to 150 "
        "contracts with a different single gold document -- an unscoped document ranking is "
        "unrankable by construction, and MAUD's real (section) target is not reachable on the "
        "src/ query path. Use --dataset nq for a second document-target corpus."
    )
}


class PoolStub(CacheBackedProvider):
    """``CacheBackedProvider`` widened to what the INGEST path also touches.

    The parent serves the query path only (768-dim, no prefix attributes). Ingest
    calls ``embed_sync(texts, batch_size)`` positionally and ``_core`` reads
    ``document_prefix``/``query_prefix`` when writing metadata, so both are
    supplied here rather than by editing the parent -- eval_equivalence's
    published numbers were produced by that class as it stands.
    """

    document_prefix = ""
    query_prefix = ""

    def __init__(self, model: str, cache_dir: Path, dimension: int) -> None:
        super().__init__(model, cache_dir)
        self._dim = dimension

    def get_dimension(self) -> int:
        return self._dim

    def embed_sync(self, texts: Sequence[str], *args: Any, **kwargs: Any) -> List[List[float]]:
        # Swallows ingest's positional `batch_size`; the parent's second parameter
        # is `task`, and letting a batch size land there would be silent.
        return super().embed_sync(texts)

    def embed_batch(self, texts: Sequence[str], *args: Any, **kwargs: Any) -> List[List[float]]:
        return super().embed_sync(texts)


def _construct(spec: Dict[str, Any], stub: Optional[PoolStub], *, live: bool = False):
    """Open the index, either against a live LOCAL encoder or the cache stub.

    A corpus whose encoder runs on this box (``provider`` set in the spec) is
    opened with the real provider: it costs nothing, it needs no cache, and it
    removes the stub from the trust chain entirely. Everything else is
    constructed against MOCK and has its provider replaced, because opening with
    the live one costs a validate and a dimension probe at construction -- on
    Ollama that forces a model reload, on OpenAI it is a live billed call. After
    the swap no text this harness handles can reach a network encoder; a text
    with no cached vector raises instead.

    ``live`` forces the real provider named by ``spec["ollama"]`` for an INGEST
    that has nothing to read from cache -- a corpus being built for the first
    time. It is only ever set for a local encoder (guarded by the caller), so it
    spends CPU and never money, and it makes the build mechanically identical to
    the one that produced every other ``hiergate__`` index: src/ applies its own
    document prefix through the provider's registry, which is byte-identical to
    the ``EGEMMA_DOC_PREFIX`` the vector cache was keyed with.
    """
    from localvectordb import LocalVectorDB

    if spec.get("provider"):
        return LocalVectorDB(
            spec["db"],
            DATA_DIR / "db",
            embedding_provider=spec["provider"],
            embedding_model=spec["model"],
            faiss_index_type=spec.get("index_type", "IndexFlatL2"),
            **spec.get("chunking", {}),
        )

    if live:
        if not spec.get("ollama"):
            raise ValueError(f"--embed refused for {spec['db']}: no local encoder declared for this corpus")
        kwargs = dict(
            embedding_provider="ollama",
            embedding_model=spec["model"],
            hierarchical_embeddings=spec["hierarchical"],
            # Transport only -- it cannot change a stored vector. 2 rather than
            # the tempting 8: a long-chunk Ollama ingest at high concurrency
            # fails with empty error messages at exactly the 300s timeout.
            embedding_config={"max_concurrent_requests": 2},
        )
        if spec["hierarchical"]:
            kwargs["section_vector_strategy"] = "centroid"
        return LocalVectorDB(spec["db"], DATA_DIR / "db", **kwargs)

    kwargs: Dict[str, Any] = dict(
        embedding_provider="mock",
        embedding_model=spec["model"],
        embedding_config={"dimension": spec["dimension"]},
        hierarchical_embeddings=spec["hierarchical"],
    )
    if spec["hierarchical"]:
        kwargs["section_vector_strategy"] = "centroid"
    db = LocalVectorDB(spec["db"], DATA_DIR / "db", **kwargs)
    db._embedding_provider = stub
    return db


def open_db(spec: Dict[str, Any], stub: PoolStub):
    base = DATA_DIR / "db"
    if not (base / f"{spec['db']}.complete").exists():
        raise FileNotFoundError(f"No completed index at {base / spec['db']} -- run this harness with `--build` first.")
    return _construct(spec, stub)


def build_db(spec: Dict[str, Any], stub: PoolStub, bench, rebuild: bool, *, live: bool = False):
    """Ingest a corpus into a real index using only cached vectors.

    A ``.complete`` sentinel is written last, so an interrupted build is never
    mistaken for a finished one -- an index missing half its documents scores at
    chance while looking like a result (the failure ``eval_hier_gate.build_db``
    documents).
    """
    base = DATA_DIR / "db"
    base.mkdir(parents=True, exist_ok=True)
    sentinel = base / f"{spec['db']}.complete"
    if sentinel.exists() and not rebuild:
        logger.info("Reusing built index %s", spec["db"])
        return _construct(spec, stub)
    # `{db}.*` alone does NOT reach the hierarchical sidecars: they are named
    # `{db}_sections.faiss` / `{db}_documents.faiss`, with an underscore before
    # the dot, so a rebuild would silently inherit the previous build's section
    # and document vectors. The second pattern is anchored on `{db}_` rather than
    # `{db}` so that discarding `...__qasper__...` cannot also delete
    # `...__qasper_full__...`.
    for pattern in (f"{spec['db']}.*", f"{spec['db']}_*.faiss"):
        for path in base.glob(pattern):
            path.unlink(missing_ok=True)

    db = _construct(spec, stub, live=live)
    doc_ids = list(bench.corpus)
    logger.info("building %s: %d documents (live encoder: %s)", spec["db"], len(doc_ids), live)
    started = time.monotonic()
    for start in range(0, len(doc_ids), 10):
        batch = doc_ids[start : start + 10]
        db.upsert([bench.corpus[d] for d in batch], ids=batch)
        done = min(start + 10, len(doc_ids))
        # A live ingest of this size runs for hours, so log a MEASURED rate and
        # ETA rather than leaving the operator to extrapolate from a remembered
        # tokens/sec -- that estimate has been wrong by 6x on this box before.
        elapsed = time.monotonic() - started
        eta = elapsed / done * (len(doc_ids) - done)
        logger.info(
            "  ingested %d/%d  (%.1f docs/min, ETA %.0f min)", done, len(doc_ids), done / elapsed * 60, eta / 60
        )
        if done % 100 == 0:
            db.save()  # checkpoint; the sentinel is still withheld until the end
    db.save()
    if stub is not None and stub.misses:
        raise SystemExit(
            f"ABORT: {len(stub.misses)} chunk texts had no cached vector, e.g. {stub.misses[0]!r}. "
            "The ingest chunker and the one that filled the cache disagree; fix that rather than "
            "embedding, or this index is not the one the numpy results were measured on."
        )
    sentinel.write_text("ok", encoding="utf-8")
    return db


def fill_query_cache(spec: Dict[str, Any], queries: Dict[str, str], qids: Sequence[str]) -> None:
    """Bank the query vectors the stub will read, through the SAME code that wrote the rest.

    A new corpus brings new queries, and the sweep's stub raises on a cache miss
    -- correctly, since a miss means the arm would be scored against a vector
    nobody can account for. This fills the gap by delegating to
    ``eval_hierarchical.CachedEmbedder`` rather than reimplementing its key
    derivation: a hand-rolled ``sha256(model \\x00 text)`` that disagreed with it
    by one byte would not fail, it would write a second, shadow set of vectors
    into the same directory and every harness reading that cache afterwards would
    silently get whichever it happened to hash to.

    The prefix is applied HERE, with the provider's own prefixing off, because
    that is the convention the directory was written under. It is the same string
    src/ applies through its registry, so the banked vector is what a live query
    would have produced -- checked, not assumed (``--verify-cache``).
    """
    from benchmarks.eval_equivalence import EGEMMA_QUERY_PREFIX
    from benchmarks.eval_hierarchical import CachedEmbedder

    want = (CACHE_DIR / "hier_embed" / spec["cache"]).resolve()
    emb = CachedEmbedder("ollama", spec["model"], num_ctx=2048)
    if emb.cache_dir.resolve() != want:
        raise SystemExit(
            f"ABORT: CachedEmbedder would write to {emb.cache_dir}, but the sweep's stub reads "
            f"{want}. Filling the wrong directory leaves the sweep failing on misses while the "
            "vectors sit somewhere nobody looks."
        )
    texts = [EGEMMA_QUERY_PREFIX + queries[q] for q in qids]
    emb.encode(texts)
    logger.info("query cache: %d embedded, %d already present -> %s", emb.n_embedded, emb.n_cached, emb.cache_dir)


def run_arm(db, queries: Dict[str, str], qids: Sequence[str], method: str) -> List[List[str]]:
    """Ranked document ids per query, straight off ``db.query()``."""
    out: List[List[str]] = []
    for qid in qids:
        hits = db.query(
            queries[qid],
            k=K,
            search_level="chunks",
            return_type="documents",
            search_type="hybrid",
            document_scoring_method=method,
        )
        out.append([h.id for h in hits])
    return out


def measure_fanout(db, queries: Dict[str, str], qids: Sequence[str], pool: int) -> Dict[str, float]:
    """Mean children per parent in the fused pool, and how many parents there are.

    READ THIS BEFORE THE nDCG COLUMN. An aggregator only differs from ``max`` when
    a parent owns more than one child in the pool, so at fanout ~1 every method in
    the sweep is the same operator and the whole table is noise. Reading a
    percentile result off a fanout-1.09 corpus is a mistake this study has already
    made once (§19.8), which is why the number is captured here rather than
    reasoned about afterwards.

    ``k=pool`` with the pool patched to ``pool`` returns the fused pool itself:
    ``_hybrid_search``'s chunk branch sorts and truncates to ``k``, and ``k`` is
    the width, so nothing is dropped.
    """
    children: List[int] = []
    parents: List[int] = []
    for qid in qids:
        hits = db.query(queries[qid], k=pool, search_level="chunks", return_type="chunks", search_type="hybrid")
        owners: Dict[str, int] = {}
        for h in hits:
            # `document_id` is a QueryResult field, not metadata -- reading it off
            # `metadata` returns None for every hit and silently reports fanout ==
            # pool size with exactly one parent, which looks like a real number.
            doc = h.document_id
            owners[doc] = owners.get(doc, 0) + 1
        if owners:
            children.append(len(hits))
            parents.append(len(owners))
    if not parents:
        return {"fanout": float("nan"), "pool_chunks": float("nan"), "pool_parents": float("nan")}
    return {
        "fanout": float(np.mean([c / p for c, p in zip(children, parents, strict=True)])),
        "pool_chunks": float(np.mean(children)),
        "pool_parents": float(np.mean(parents)),
    }


def score(ranked: List[List[str]], qids: Sequence[str], qrels: Dict[str, Dict[str, int]]) -> np.ndarray:
    return np.array(
        [ndcg_at_k(r, qrels.get(q, {}), K) for q, r in zip(qids, ranked, strict=True)],
        dtype=np.float64,
    )


def compare(path_a: Path, path_b: Path) -> int:
    """Paired bootstrap between two artifacts, arm by arm, on their shared queries.

    The contrast this exists for is "same queries, bigger haystack": scoring
    qasper's 882 dev queries against the 275-paper index and against the
    1,088-paper one. Those two runs differ in exactly one thing, so the
    difference is the cost of 813 more distractor documents -- which is otherwise
    inseparable from the effect of simply drawing a larger query sample.

    Pairing is by QUERY ID, never by position: the two files' ``qids`` lists are
    built by different runs over different corpora, and lining up two arrays that
    happen to be the same length would silently compare unrelated queries.
    """
    a = json.loads(path_a.read_text(encoding="utf-8"))
    b = json.loads(path_b.read_text(encoding="utf-8"))
    common = [q for q in a["qids"] if q in set(b["qids"])]
    if not common:
        raise SystemExit(f"No shared query ids between {path_a.name} and {path_b.name}")
    ia = {q: i for i, q in enumerate(a["qids"])}
    ib = {q: i for i, q in enumerate(b["qids"])}

    print(f"\nA = {a['config'].get('arm', a['config']['dataset'])}  ({a['config'].get('n_documents', '?')} docs)")
    print(f"B = {b['config'].get('arm', b['config']['dataset'])}  ({b['config'].get('n_documents', '?')} docs)")
    print(f"{len(common)} shared queries\n")
    print(f"{'arm':<30}{'A':>9}{'B':>9}{'B-A':>10}{'95% CI':>22}{'p':>8}")
    for label in a["per_query"]:
        if label not in b["per_query"]:
            continue
        va = np.array([a["per_query"][label][ia[q]] for q in common])
        vb = np.array([b["per_query"][label][ib[q]] for q in common])
        st = paired(vb, va)
        ci = f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
        print(f"{label:<30}{va.mean():>9.4f}{vb.mean():>9.4f}{st['delta']:>+10.4f}{ci:>22}{st['p']:>8.3f}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(CORPORA), default="qasper")
    p.add_argument("--pools", type=int, nargs="+", default=list(POOLS))
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--max-queries", type=int, default=None, help="smoke only; caps the query set")
    p.add_argument("--build", action="store_true", help="ingest the corpus first if no index exists")
    p.add_argument("--nq-queries", type=int, default=2000, help="NQ query cap; 2000 matches agg_nq_v2")
    p.add_argument(
        "--fanout-only",
        action="store_true",
        help="measure children-per-parent in the pool and stop -- the gating variable for "
        "whether any aggregator comparison means anything",
    )
    p.add_argument(
        "--i-know-maud-is-unrankable",
        action="store_true",
        help="override the MAUD refusal (see _REFUSED); the numbers will be noise on an "
        "unrankable target, so this exists for diagnosis only",
    )
    p.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="qasper smoke builds only. Suffixes the index key for the same reason --max-contracts "
        "does: a 12-paper smoke index that got reused by a full run once scored every arm at "
        "chance (nDCG ~= 10/275) while looking like a real result.",
    )
    p.add_argument(
        "--max-contracts",
        type=int,
        default=None,
        help="MAUD smoke builds only. Suffixes the index key, so a 5-contract smoke can never be "
        "reused by the full run -- an index missing most of its gold scores at chance and looks real.",
    )
    p.add_argument("--rebuild", action="store_true", help="discard and re-ingest even if complete")
    p.add_argument("--build-only", action="store_true", help="stop after ingest; do not sweep")
    p.add_argument(
        "--fill-query-cache",
        action="store_true",
        help="embed any query text this corpus needs that is not yet banked, then stop. Local "
        "encoder only. Run this once after a build; the sweep then reads vectors, not the encoder.",
    )
    p.add_argument(
        "--embed",
        action="store_true",
        help="ingest through the REAL local encoder instead of the cache stub. Only legal for a "
        "corpus that declares a local Ollama model, so it spends CPU-hours and never money; "
        "required to build a corpus whose chunks have never been embedded.",
    )
    p.add_argument(
        "--query-subset",
        choices=("all", "dev"),
        default="all",
        help="qasper_full only. `dev` scores the 882 dev queries against the FULL index, which is "
        "what separates the effect of 4x more distractors from the effect of a different query "
        "sample -- dev is a verified strict subset of full (same ids, same rendered text).",
    )
    p.add_argument("--no-verify", action="store_true", help="skip the patch-inertness check (do not)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("A.json", "B.json"),
        help="paired bootstrap between two finished artifacts on their shared query ids, and exit",
    )
    args = p.parse_args(argv)

    if args.compare:
        return compare(*args.compare)

    if args.dataset in _REFUSED and not args.i_know_maud_is_unrankable:
        raise SystemExit(f"REFUSING --dataset {args.dataset}: {_REFUSED[args.dataset]}")

    spec = dict(CORPORA[args.dataset])
    stub = (
        None
        if spec.get("provider")
        else PoolStub(spec["model"], CACHE_DIR / "hier_embed" / spec["cache"], spec["dimension"])
    )

    if args.dataset == "nfcorpus":
        from benchmarks.beir_data import load

        ds = load("nfcorpus")

        class _Bench:
            corpus = ds.corpus
            queries = ds.queries
            doc_qrels = ds.qrels

        bench = _Bench()
    elif args.dataset in ("qasper", "qasper_full"):
        from benchmarks.qasper_data import load_qasper

        bench = load_qasper(split="dev" if args.dataset == "qasper" else "full", max_papers=args.max_papers)
        if args.max_papers:
            spec["db"] += f"__max{args.max_papers}"
    elif args.dataset == "nq":
        from benchmarks.nq_data import load_nq

        # 2,000 matches the numpy run this is confirming (`agg_nq_v2.json`); a
        # different cap would compare two different corpora, not two paths.
        bench = load_nq(max_queries=args.nq_queries)
        if args.nq_queries != 2000:
            spec["db"] = spec["db"].replace("nq2000", f"nq{args.nq_queries}")
    else:
        from benchmarks.maud_data import load_maud

        bench = load_maud(max_contracts=args.max_contracts)
        if args.max_contracts:
            spec["db"] += f"__max{args.max_contracts}"

    if args.fill_query_cache:
        if not spec.get("ollama"):
            raise SystemExit(f"--fill-query-cache refused for {args.dataset}: no local encoder declared")
        # Every scored query, not just the arm being swept: a subset arm reads
        # from the same directory, and a half-filled cache fails later, not here.
        scored = [q for q in bench.queries if any(v > 0 for v in bench.doc_qrels.get(q, {}).values())]
        fill_query_cache(spec, bench.queries, scored)
        return 0

    if args.build or args.rebuild:
        db = build_db(spec, stub, bench, args.rebuild, live=args.embed)
        if args.build_only:
            # Deliberately a separate invocation from the sweep. A `--embed` build
            # leaves the LIVE encoder attached, and the sweep re-embeds every query
            # once per arm -- 14 passes over 2,940 queries at ~0.5 s each is days,
            # against seconds once those vectors are in the cache the stub reads.
            logger.info("build complete: %s -- rerun without --build to sweep", spec["db"])
            return 0
    else:
        db = open_db(spec, stub)

    qrels = bench.doc_qrels
    qids = [q for q in bench.queries if any(v > 0 for v in qrels.get(q, {}).values())]

    if args.query_subset == "dev":
        if args.dataset != "qasper_full":
            raise SystemExit("--query-subset dev is only meaningful for --dataset qasper_full")
        from benchmarks.qasper_data import load_qasper

        keep = set(load_qasper(split="dev").queries)
        subset = [q for q in qids if q in keep]
        # If dev ever stops being a subset of full, this arm silently becomes a
        # different query set and the distractor contrast is no longer paired.
        if len(subset) != len(keep):
            raise SystemExit(f"ABORT: {len(keep) - len(subset)} dev queries are absent from qasper_full")
        qids = subset

    if args.max_queries:
        qids = qids[: args.max_queries]
    # Artifacts are named for the ARM, not the dataset: `qasper_full` scored over
    # dev's queries is a different measurement from `qasper_full` over all of
    # them, and writing both to one filename would let the second silently
    # overwrite the first.
    tag = args.dataset if args.query_subset == "all" else f"{args.dataset}_devq"
    # A capped smoke must not write to the full run's filename either -- the same
    # collision the index key guards against, one level up in the artifacts.
    for cap, name in ((args.max_papers, "papers"), (args.max_contracts, "contracts"), (args.max_queries, "q")):
        if cap:
            tag += f"__max{name}{cap}"
    logger.info("%s: %d scored queries over %d documents", tag, len(qids), len(bench.corpus))

    from localvectordb.database import _search

    shipped_fn = _search._hybrid_pool_size

    def patch(pool: int) -> None:
        _search._hybrid_pool_size = lambda k, _p=pool: max(k, _p)

    results: Dict[str, Dict[str, Any]] = {}
    per_query: Dict[str, List[float]] = {}
    fanout: Dict[str, Dict[str, float]] = {}

    if args.fanout_only:
        try:
            for pool in args.pools:
                patch(pool)
                fanout[f"pool={pool}"] = measure_fanout(db, bench.queries, qids, pool)
                logger.info("pool=%-5d %s", pool, fanout[f"pool={pool}"])
        finally:
            _search._hybrid_pool_size = shipped_fn
        out = args.out or (Path("experiments") / f"poolwidth_{tag}_fanout.json")
        out.write_text(json.dumps({"dataset": args.dataset, "fanout": fanout}, indent=2), encoding="utf-8")
        logger.info("wrote %s", out)
        return 0

    # Inertness check first: whatever the sweep says afterwards is only about
    # pool width if substituting the function changes nothing at the shipped one.
    if not args.no_verify:
        _search._hybrid_pool_size = shipped_fn
        untouched = score(run_arm(db, bench.queries, qids, "frequency_boost"), qids, qrels)
        patch(SHIPPED_POOL)
        patched = score(run_arm(db, bench.queries, qids, "frequency_boost"), qids, qrels)
        if not np.array_equal(untouched, patched):
            n = int((untouched != patched).sum())
            raise SystemExit(
                f"ABORT: the pool patch is not inert at the shipped width -- {n}/{len(qids)} queries "
                f"differ (mean {untouched.mean():.4f} vs {patched.mean():.4f}). Every number this "
                "harness would print measures the patch, not the pool."
            )
        logger.info("verify OK: patched pool=%d reproduces the shipped path per query", SHIPPED_POOL)

    try:
        for pool in args.pools:
            patch(pool)
            for method in args.methods:
                ranked = run_arm(db, bench.queries, qids, method)
                vals = score(ranked, qids, qrels)
                label = f"pool={pool}|{method}"
                per_query[label] = vals.tolist()
                results[label] = {
                    "ndcg@10": float(vals.mean()),
                    "recall@10": float(
                        np.mean([recall_at_k(r, qrels.get(q, {}), K) for q, r in zip(qids, ranked, strict=True)])
                    ),
                }
                logger.info("%-28s ndcg@10 %.4f", label, vals.mean())
    finally:
        _search._hybrid_pool_size = shipped_fn

    # Everything is read against what a user gets today: the shipped width with
    # the shipped hybrid aggregator.
    base_label = f"pool={SHIPPED_POOL}|frequency_boost"
    print(f"\n{tag}: {len(qids)} queries over {len(bench.corpus)} docs, baseline = {base_label}\n")
    print(f"{'arm':<30}{'ndcg@10':>10}{'delta':>10}{'95% CI':>22}{'p':>8}")
    stats: Dict[str, Dict[str, float]] = {}
    if base_label in per_query:
        base = np.asarray(per_query[base_label])
        for label, vals in per_query.items():
            arr = np.asarray(vals)
            if label == base_label:
                print(f"{label:<30}{arr.mean():>10.4f}{'--':>10}{'':>22}{'':>8}")
                continue
            st = paired(arr, base)
            stats[label] = st
            ci = f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
            print(f"{label:<30}{arr.mean():>10.4f}{st['delta']:>+10.4f}{ci:>22}{st['p']:>8.3f}")

    out = args.out or (Path("experiments") / f"poolwidth_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": {
                    "dataset": args.dataset,
                    "arm": tag,
                    "query_subset": args.query_subset,
                    "n_documents": len(bench.corpus),
                    "db": spec["db"],
                    "model": spec["model"],
                    "pools": args.pools,
                    "methods": args.methods,
                    "k": K,
                    "search_type": "hybrid",
                    "n_queries": len(qids),
                    "verified": not args.no_verify,
                },
                "qids": qids,
                "grid": results,
                "vs_shipped": stats,
                "per_query": per_query,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", out)
    if stub is not None and stub.misses:
        logger.warning("%d cache misses (should be 0): e.g. %r", len(stub.misses), stub.misses[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
