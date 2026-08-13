"""Does cross-encoder reranking wash out first-stage tuning? (PAPER-OUTLINE §6.2)

The single largest unmeasured lever in this study. `reranking.py` ships pluggable
cross-encoders and `_resolve_rerank_k` correctly over-fetches ``5*k``, but
reranking appears in **no** retrieval measurement here -- it was excluded from the
committed baseline as "slow on CPU", and that objection died when hosted
rerankers became free.

Why it may reframe everything: every effect in SYNTHESIS-v2 lives at ~+0.008,
while the keyword leg was worth +0.084 to +0.131. If a cross-encoder over a wide
pool is another order-of-magnitude lever AND it erases the differences between
pool widths and aggregators, then the honest headline becomes "do not tune the
first stage, rerank" -- a finding, but a different paper. We cannot publish
without knowing which.

DESIGN -- capture once per WIDTH, rerank once, aggregate offline. Aggregators
(``METHODS``) are derived offline from a single rerank call, which is the
§17.1/§18.2 capture-once pattern and is valid: they re-rank a fixed candidate
set. Pool width is NOT: see ``--pools``. Each width gets its own retrieval.

    WARNING -- this file originally did derive pool widths by slicing one
    POOL_MAX=200 capture, on the false assumption that a narrow pool is a prefix
    of a wide one. It is not: width changes the hybrid fusion itself. Any
    artifact written before that fix has invalid pool-width axes. The tell is
    ``best`` scoring IDENTICALLY at every width, because truncation cannot change
    a max -- `experiments/logs/rerank_qasper_nemotron.log` shows 0.4724 at all
    three widths and must not be read as a pool-width result.

Rerank responses are cached on disk keyed by (model, query, candidate ids), so
re-analysis is free and a killed run resumes.

Zero embedding: rides the same cache-backed stub as the pool-width sweeps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.config import CACHE_DIR  # noqa: E402

logger = logging.getLogger(__name__)

RERANK_URL = "https://openrouter.ai/api/v1/rerank"
# Confirmed live 2026-08-12. The `:free` suffix is load-bearing on the nvidia
# model -- without it the endpoint reports "No endpoints found", which reads as a
# wrong model id rather than a missing tier.
# All six verified live 2026-08-12. Price is $/M tokens; context matters because
# a reranker scores (query + candidate) pairs, and cohere v3.5's 4k window is the
# only one that a long chunk could plausibly overflow.
MODELS = {
    "nemotron": "nvidia/llama-nemotron-rerank-vl-1b-v2:free",  # free,     10,240 ctx
    "cohere35": "cohere/rerank-v3.5",  # $0.001,    4,096 ctx
    "cohere4fast": "cohere/rerank-4-fast",  # $0.002,   32,768 ctx
    "cohere4pro": "cohere/rerank-4-pro",  # $0.0025,  32,768 ctx
    "voyage25lite": "voyageai/rerank-2.5-lite",  # $0.02,    32,000 ctx
    "voyage25": "voyageai/rerank-2.5",  # $0.05,    32,000 ctx
}

# Local cross-encoders, run through the class `src/` already ships
# (`SentenceTransformersReranker`). These matter for more than cost: the library's
# whole pitch is offline and one-dependency, so a lever that only exists behind an
# API would compromise it.
#
# "Slow on CPU" is why reranking was excluded from the committed baseline and
# therefore why the largest effect in this study went unmeasured for its whole
# duration. The objection turned out to be RIGHT, and an earlier version of this
# comment claimed the opposite on a bad benchmark:
#
#   WITHDRAWN -- "285 pairs/s, qasper in 2.1 min, ~7x faster than the hosted
#   free tier". That was measured on 80 SHORT SENTENCES. Real candidates are
#   ~500-token chunks and attention is quadratic in length, so on the actual
#   workload bge-reranker-base runs at ~1.7 pairs/s -- roughly 170x slower, and
#   ~3.5 h per corpus against the hosted tier's ~14 min for all 882 queries.
#
# Local reranking is therefore a LATENCY trade, not a free win: bge-base is
# statistically tied with a hosted model charging $0.35/run, at ~15x the wall
# clock. Benchmark the operator on the inputs it will actually see -- the same
# error as `_CHARS_PER_TOKEN=3.5`.
#
# NOTE Ollama has reranker models on its site but **no rerank route** (checked
# 0.32.5: /api/rerank and /v1/rerank both 404). A cross-encoder needs a
# (query, documents) -> scores endpoint; /api/embed would return bi-encoder
# vectors, which is a different operator. Do not route local reranking there.
# The pool is chosen to SEPARATE the two live explanations for MiniLM-L6's null
# on qasper. Holding training data fixed and raising capacity (L6 -> L12) isolates
# capacity; holding capacity roughly fixed and changing training data
# (ms-marco -> bge/mxbai) isolates domain. Testing only bigger models would
# confound the two, which is the mistake this study keeps having to undo.
# TOTAL PARAMETER COUNT IS THE WRONG SIZE AXIS HERE, for both capacity and
# runtime. Most of these models' parameters are the vocabulary embedding, which
# is a lookup table: it costs disk, not compute, and does not make the model a
# better ranker. Measured from each config (vocab x hidden):
#
#   model                 type          L    H    emb params   backbone
#   ms-marco-MiniLM-L-6   bert           6  384       11.7M    small
#   ms-marco-MiniLM-L-12  bert          12  384       11.7M    narrow
#   ms-marco-electra-base electra       12  768       23.4M    full
#   bge-reranker-base     xlm-roberta   12  768      192.0M    full   (278M total!)
#   mxbai-rerank-base-v1  deberta-v2    12  768       98.4M    full   (184M total)
#
# So bge (278M) and mxbai (184M) have the SAME backbone geometry; the entire
# apparent size gap is XLM-R's 250k-token vocabulary. Compare L and H, not totals.
# Runtime follows the backbone too, plus DeBERTa's disentangled attention, which
# roughly doubles attention cost -- which is why mxbai (the "smaller" model) runs
# ~2.3x slower per pair than bge. Estimating either from total params is the same
# error as `_CHARS_PER_TOKEN`: measure the operator, don't scale a proxy.
LOCAL_MODELS = {
    # MS MARCO family -- web-search passages, the suspected domain mismatch.
    "local-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 22M
    "local-minilm12": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # 33M, capacity control
    # THE DOMAIN CONTROL. Same backbone as bge/mxbai (L=12, H=768) but MS MARCO
    # training and a BERT-family attention, so it isolates training data at fixed
    # capacity -- the cell `local-bge` vs `local-minilm` cannot separate, because
    # those differ in BOTH. BERT-family means no DeBERTa tax, so expect ~bge's
    # wall clock, not mxbai's.
    "local-electra": "cross-encoder/ms-marco-electra-base",  # L=12 H=768, MS MARCO
    # Broader retrieval training data, at comparable and larger sizes.
    "local-bge": "BAAI/bge-reranker-base",  # 278M
    "local-bge-large": "BAAI/bge-reranker-large",  # 560M
    "local-bge-m3": "BAAI/bge-reranker-v2-m3",  # 568M, multilingual
    "local-mxbai": "mixedbread-ai/mxbai-rerank-base-v1",  # 184M
}

# A relevance head either works or it does not, and a broken one is INVISIBLE
# from the outside: llama.cpp's Qwen3-Reranker GGUFs emit ~4.5e-23 for every
# pair, which is well-formed JSON, plausible floats, and an arbitrary ranking.
# `local-minilm` scored a clean 0.0000 on qasper and I reported it as a finding
# before checking that its head worked at all -- these two look identical in the
# results table. Three pairs, seconds, run before every model.
SANITY = (
    "What dataset was used to train the model?",
    [
        ("relevant", "We train our model on the SQuAD 2.0 dataset, using the standard train split."),
        ("related", "The model architecture uses 12 transformer layers with 768 hidden units."),
        ("offtopic", "Preheat the oven to 200C and butter a 9-inch springform cake tin."),
    ],
)


def sanity_check(client) -> None:
    """Refuse to run a reranker that cannot rank three obvious documents."""
    query, docs = SANITY
    scores = client.score(query, [f"sanity-{lab}" for lab, _ in docs], [t for _, t in docs])
    logger.info("  sanity: " + "  ".join(f"{lab}={s:+.4f}" for (lab, _), s in zip(docs, scores, strict=False)))
    if not (scores[0] > scores[2]):
        raise SystemExit(
            f"ABORT: reranker {client.model} scores an off-topic document at least as high as a "
            f"relevant one ({scores[0]:+.4f} vs {scores[2]:+.4f}). Its relevance head is not "
            "working; a null result from this model would be a wiring bug, not a finding."
        )


POOL_MAX = 200
POOLS = (40, 100, 200)
METHODS = ("frequency_boost", "best")
K = 10


class RerankClient:
    """OpenRouter rerank with a persistent per-(model, query, pool) disk cache.

    Deliberately NOT a ``localvectordb.reranking.Reranker`` subclass yet. Adding
    a provider to ``src/`` obliges tests and both retrieval gates; this is a
    measurement, and if the result justifies shipping it, promoting it is a
    separate change with its own gates.
    """

    def __init__(self, model_key: str, timeout: int = 120) -> None:
        self.model = MODELS[model_key]
        self.timeout = timeout
        self.dir = CACHE_DIR / "rerank" / model_key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n_called = self.n_cached = 0
        self._key = os.getenv("OPENROUTER_API_KEY")
        if not self._key:
            raise SystemExit("OPENROUTER_API_KEY missing -- experiments/.env is not loaded")

    def _path(self, query: str, ids: Sequence[str]) -> Path:
        # The candidate IDS go in the key, not the texts: same ids => same texts
        # for a fixed index, and hashing 200 chunk bodies per query is pure cost.
        h = hashlib.sha256(("\x00".join([self.model, query, *ids])).encode("utf-8")).hexdigest()
        return self.dir / f"{h}.json"

    def score(self, query: str, ids: Sequence[str], texts: Sequence[str]) -> List[float]:
        """Relevance score per candidate, in the order given."""
        path = self._path(query, ids)
        if path.exists():
            self.n_cached += 1
            return json.loads(path.read_text(encoding="utf-8"))

        import requests

        payload = {"model": self.model, "query": query, "documents": list(texts)}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        for attempt in range(6):
            r = requests.post(RERANK_URL, headers=headers, json=payload, timeout=self.timeout)
            if r.status_code == 200:
                break
            # 429 is expected on a free tier and is not an error condition; back
            # off rather than dying two hours into a run.
            if r.status_code in (429, 502, 503, 529):
                wait = 2**attempt
                logger.warning("  rerank %s, retry in %ds", r.status_code, wait)
                time.sleep(wait)
                continue
            raise SystemExit(f"rerank failed {r.status_code}: {r.text[:300]}")
        else:
            raise SystemExit("rerank: exhausted retries")

        out = [0.0] * len(texts)
        for row in r.json()["results"]:
            out[row["index"]] = float(row["relevance_score"])
        path.write_text(json.dumps(out), encoding="utf-8")
        self.n_called += 1
        return out


class LocalRerankClient:
    """A CPU cross-encoder with the same interface as ``RerankClient``.

    No disk cache: at 285 pairs/s the whole corpus re-scores in about the time a
    cache lookup would take to write, and a cache is one more thing that can
    silently serve a stale model's scores.
    """

    def __init__(self, model_key: str, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self.model = LOCAL_MODELS[model_key]
        self.n_called = self.n_cached = 0
        # 512 is the conventional default and is NOT safe here: a chunk built at
        # `chunk_size=500` TOKENS, plus the query and specials, exceeds it -- so
        # the candidate is clipped before the cross-encoder ever sees its tail.
        # That is S5's coverage problem reappearing on the reranking side, in a
        # place the study never looked. Sweep it before blaming a null on domain
        # mismatch or model capacity.
        self.max_length = max_length
        self._ce = CrossEncoder(self.model, max_length=max_length)

    def score(self, query: str, ids: Sequence[str], texts: Sequence[str]) -> List[float]:
        self.n_called += 1
        return [float(s) for s in self._ce.predict([(query, t) for t in texts], batch_size=32, show_progress_bar=False)]


def ndcg_at_k(ranked: Sequence[str], rels: Dict[str, int], k: int) -> float:
    gains = [rels.get(d, 0) for d in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def aggregate(pairs: Sequence[Tuple[str, float]], method: str) -> List[str]:
    """Chunk (doc_id, score) pairs -> ranked document ids, mirroring src/."""
    by_doc: Dict[str, List[float]] = {}
    for doc, s in pairs:
        by_doc.setdefault(doc, []).append(s)
    scored: Dict[str, float] = {}
    for doc, ss in by_doc.items():
        best = max(ss)
        if method == "best":
            scored[doc] = best
        elif method == "frequency_boost":
            # src/'s formula, clamp included -- §18.4 shows the clamp is
            # load-bearing, so dropping it would measure a different operator.
            scored[doc] = min(1.0, best * (1 + (np.log2(2 + len(ss)) - 1) * 0.3))
        else:
            raise ValueError(method)
    return [d for d, _ in sorted(scored.items(), key=lambda kv: -kv[1])]


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.ERROR)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="qasper")
    p.add_argument("--nq-queries", type=int, default=2000, help="NQ query cap; 2000 matches the built index")
    p.add_argument("--model", choices=sorted(MODELS) + sorted(LOCAL_MODELS), default="nemotron")
    p.add_argument(
        "--search-type",
        choices=("vector", "keyword", "hybrid"),
        default="hybrid",
        help="FIRST-STAGE retrieval. The interesting question is whether reranking a vector-only "
        "first stage lands where hybrid+rerank lands: if it does, BM25 is doing work the reranker "
        "would have done anyway, and the two big levers are substitutes rather than additive.",
    )
    p.add_argument(
        "--pools",
        type=int,
        nargs="+",
        default=[40],
        help="first-stage widths, each RETRIEVED SEPARATELY. Do not go back to slicing one wide "
        "capture: pool width changes the hybrid fusion itself (min-max denominators and the "
        "zero-fill for chunks only one leg found, i.e. the whole S19.3 mechanism), so a slice of a "
        "pool-200 fusion is not a pool-40 retrieval. The tell that this was wrong the first time: "
        "`best` scored identically at every width, because truncation cannot change a max.",
    )
    p.add_argument(
        "--max-length", type=int, default=512, help="local rerankers only; token budget per (query, chunk) pair"
    )
    p.add_argument("--max-queries", type=int, default=None, help="smoke; also suffixes the artifact")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    from benchmarks.eval_dual import _load_experiments_env

    _load_experiments_env()

    from benchmarks.eval_pool_width import CORPORA, PoolStub, open_db
    from localvectordb.database import _search

    spec = dict(CORPORA[args.dataset])
    stub = (
        None
        if spec.get("provider")
        else PoolStub(spec["model"], CACHE_DIR / "hier_embed" / spec["cache"], spec["dimension"])
    )

    from benchmarks.eval_pool_width import _REFUSED

    if args.dataset in _REFUSED:
        raise SystemExit(f"REFUSING --dataset {args.dataset}: {_REFUSED[args.dataset]}")

    if args.dataset in ("qasper", "qasper_full"):
        from benchmarks.qasper_data import load_qasper

        bench = load_qasper(split="dev" if args.dataset == "qasper" else "full")
    elif spec.get("mldr"):
        from benchmarks.mldr_data import load_mldr

        bench = load_mldr(**spec["mldr"])
    elif args.dataset == "nfcorpus":
        from benchmarks.beir_data import load

        ds = load("nfcorpus")

        class _Bench:
            corpus = ds.corpus
            queries = ds.queries
            doc_qrels = ds.qrels

        bench = _Bench()
    elif args.dataset == "nq":
        from benchmarks.nq_data import load_nq

        # 2,000 is the cap the built index carries in its name; any other value
        # is a DIFFERENT corpus and must not reuse `poolwidth__nq2000`.
        bench = load_nq(max_queries=args.nq_queries)
        if args.nq_queries != 2000:
            spec["db"] = spec["db"].replace("nq2000", f"nq{args.nq_queries}")
    else:
        raise SystemExit(f"--dataset {args.dataset} not wired here yet")

    qids = [q for q in sorted(bench.queries) if any(v > 0 for v in bench.doc_qrels.get(q, {}).values())]
    if args.max_queries:
        qids = qids[: args.max_queries]
    db = open_db(spec, stub)
    client = (
        LocalRerankClient(args.model, max_length=args.max_length)
        if args.model in LOCAL_MODELS
        else RerankClient(args.model)
    )

    logger.info(
        "%s: %d queries, first stage=%s pools=%s, reranker=%s",
        args.dataset,
        len(qids),
        args.search_type,
        args.pools,
        client.model,
    )
    sanity_check(client)

    per_query: Dict[str, List[float]] = {}
    started = time.monotonic()
    for n, qid in enumerate(qids, 1):
        rels = bench.doc_qrels[qid]
        for pool in args.pools:
            # A genuine retrieval per width, so the fusion is recomputed rather
            # than truncated.
            _search._hybrid_pool_size = lambda k, _p=pool: max(k, _p)
            hits = db.query(
                bench.queries[qid], k=pool, search_level="chunks", return_type="chunks", search_type=args.search_type
            )
            if not hits:  # a keyword first stage can legitimately return nothing
                for method in METHODS:
                    for stage in ("first", "rerank"):
                        per_query.setdefault(f"pool={pool}|{method}|{stage}", []).append(0.0)
                continue
            docs = [h.document_id for h in hits]
            first = [float(h.score) for h in hits]
            rr = client.score(bench.queries[qid], [h.id for h in hits], [h.content for h in hits])
            order = np.argsort(-np.asarray(rr))
            for method in METHODS:
                per_query.setdefault(f"pool={pool}|{method}|first", []).append(
                    ndcg_at_k(aggregate(list(zip(docs, first, strict=False)), method), rels, K)
                )
                per_query.setdefault(f"pool={pool}|{method}|rerank", []).append(
                    ndcg_at_k(aggregate([(docs[i], rr[i]) for i in order], method), rels, K)
                )
        if n % 50 == 0 or n == len(qids):
            el = time.monotonic() - started
            logger.info(
                "  %d/%d  (%d api, %d cached, ETA %.0f min)",
                n,
                len(qids),
                client.n_called,
                client.n_cached,
                el / n * (len(qids) - n) / 60,
            )

    from benchmarks.eval_section_bm25 import paired

    print(f"\n{args.dataset}: {len(qids)} queries, first stage={args.search_type}, reranker={client.model}\n")
    print(f"{'arm':<28}{'first':>9}{'rerank':>9}{'delta':>10}{'95% CI':>22}{'p':>8}")
    results = {}
    for pool in args.pools:
        for method in METHODS:
            a = np.asarray(per_query[f"pool={pool}|{method}|first"])
            b = np.asarray(per_query[f"pool={pool}|{method}|rerank"])
            st = paired(b, a)
            ci = f"[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}]"
            print(
                f"{f'pool={pool}|{method}':<28}{a.mean():>9.4f}{b.mean():>9.4f}"
                f"{st['delta']:>+10.4f}{ci:>22}{st['p']:>8.3f}"
            )
            results[f"pool={pool}|{method}"] = {"first": a.mean(), "rerank": b.mean(), **st}

    # The paper question: does reranking COMPRESS the spread between first-stage
    # configurations? If the spread collapses, first-stage tuning stops mattering.
    if len(args.pools) * len(METHODS) > 1:
        for stage in ("first", "rerank"):
            vals = [np.asarray(per_query[f"pool={p}|{m}|{stage}"]).mean() for p in args.pools for m in METHODS]
            print(
                f"  spread across the {len(vals)} first-stage configs, {stage:>6}: "
                f"{max(vals) - min(vals):.4f}  (min {min(vals):.4f}, max {max(vals):.4f})"
            )

    # `max_length` changes the scores, so it MUST key the artifact: without it a
    # 1024-token run silently overwrites the 512-token one and the comparison
    # they exist for becomes unreadable. Same rule as the query caps.
    tag = f"{args.dataset}_{args.search_type}"
    if args.model in LOCAL_MODELS and args.max_length != 512:
        tag += f"__len{args.max_length}"
    if args.max_queries:
        tag += f"__max{args.max_queries}"
    out = args.out or Path("experiments") / f"rerank_{tag}_{args.model}.json"
    out.write_text(
        json.dumps(
            {
                "config": {
                    "dataset": args.dataset,
                    "model": client.model,
                    "search_type": args.search_type,
                    "pools": args.pools,
                    "n": len(qids),
                },
                "qids": qids,
                "results": results,
                "per_query": per_query,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
