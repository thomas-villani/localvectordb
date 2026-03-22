"""Tier 1: ANN Recall Benchmarks using SIFT-128.

Measures recall@k and QPS for each FAISS index type,
comparing raw FAISS baseline vs. through-localvectordb.
"""

import logging
import tempfile
from typing import Any, Dict, List

import faiss
import numpy as np

from localvectordb.database import LocalVectorDB
from localvectordb.embeddings import EmbeddingRegistry

from .config import INDEX_TYPES, RECALL_K_VALUES, SIFT_DIMENSION
from .datasets import compute_ground_truth, compute_recall, download_sift, load_sift
from .providers import PrecomputedEmbeddings
from .utils import LatencyStats, measure_memory_mb, timer

logger = logging.getLogger(__name__)


def _build_raw_faiss_index(
    vectors: np.ndarray,
    index_type: str,
    dimension: int = SIFT_DIMENSION,
) -> faiss.IndexIDMap2:
    """Build a FAISS index matching the construction in _core.py:423-438."""
    if index_type == "IndexFlatL2":
        base = faiss.IndexFlatL2(dimension)
    elif index_type == "IndexFlatIP":
        base = faiss.IndexFlatIP(dimension)
    elif index_type == "IndexHNSWFlat":
        base = faiss.IndexHNSWFlat(dimension, 16)
    elif index_type == "IndexLSH":
        base = faiss.IndexLSH(dimension, dimension * 2)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index = faiss.IndexIDMap2(base)
    ids = np.arange(vectors.shape[0], dtype=np.int64)
    index.add_with_ids(vectors, ids)
    return index


def _maybe_normalize(vectors: np.ndarray, index_type: str) -> np.ndarray:
    """L2-normalize vectors if using inner-product index."""
    if index_type == "IndexFlatIP":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return (vectors / norms).astype(np.float32)
    return vectors


def run_raw_faiss(
    train: np.ndarray,
    test: np.ndarray,
    ground_truth: np.ndarray,
    scale: int,
    index_type: str,
) -> Dict[str, Any]:
    """Path A: Raw FAISS baseline benchmark."""
    logger.info("  [Raw FAISS] index=%s, scale=%d", index_type, scale)
    subset = _maybe_normalize(train[:scale].copy(), index_type)
    queries = _maybe_normalize(test.copy(), index_type)
    max_k = max(RECALL_K_VALUES)

    # Build index
    with timer("build") as t_build:
        index = _build_raw_faiss_index(subset, index_type)
    logger.info("    Build time: %.3fs", t_build.elapsed_seconds)

    # Search (batch all queries)
    with timer("search") as t_search:
        _, pred_ids = index.search(queries, max_k)
    t_search.iterations = queries.shape[0]
    qps = t_search.ops_per_second

    # Compute recall at each k
    recalls = {}
    for k in RECALL_K_VALUES:
        if k <= max_k:
            recalls[f"recall@{k}"] = round(compute_recall(pred_ids, ground_truth, k), 4)

    mem = measure_memory_mb()

    return {
        "path": "raw_faiss",
        "index_type": index_type,
        "scale": scale,
        "build_time_s": round(t_build.elapsed_seconds, 4),
        "search_time_s": round(t_search.elapsed_seconds, 4),
        "qps": round(qps, 1),
        "memory_mb": round(mem, 1),
        **recalls,
    }


def run_through_lvdb(
    train: np.ndarray,
    test: np.ndarray,
    ground_truth: np.ndarray,
    scale: int,
    index_type: str,
) -> Dict[str, Any]:
    """Path B: Through-localvectordb benchmark."""
    logger.info("  [LVDB] index=%s, scale=%d", index_type, scale)
    subset = _maybe_normalize(train[:scale].copy(), index_type)
    queries = _maybe_normalize(test.copy(), index_type)
    max_k = max(RECALL_K_VALUES)

    # Register the precomputed provider
    PrecomputedEmbeddings(vectors=subset, dimension=SIFT_DIMENSION)
    EmbeddingRegistry.register("precomputed", PrecomputedEmbeddings)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build DB and upsert
        with timer("build") as t_build:
            db = LocalVectorDB(
                name="bench",
                base_path=tmpdir,
                embedding_provider="precomputed",
                embedding_model="precomputed",
                embedding_config={"vectors": subset, "dimension": SIFT_DIMENSION},
                faiss_index_type=index_type,
                chunk_size=10000,
                chunk_overlap=0,
                enable_fts=False,
            )
            docs = [f"sift_vec_{i}" for i in range(scale)]
            db.upsert(docs, batch_size=1000)
        logger.info("    Build+upsert time: %.3fs", t_build.elapsed_seconds)

        # Query each test vector
        latencies = LatencyStats()
        all_pred_ids = []

        with timer("search") as t_search:
            for q_idx in range(queries.shape[0]):
                q_text = f"query_{q_idx}"
                db.embedding_provider.register_query(q_text, queries[q_idx])

                import time as _time

                t0 = _time.perf_counter()
                results = db.query(q_text, k=max_k, search_type="vector")
                elapsed_ms = (_time.perf_counter() - t0) * 1000
                latencies.timings_ms.append(elapsed_ms)

                # Parse document IDs to get vector indices
                pred = []
                for r in results:
                    doc_id = r.id
                    if doc_id.startswith("sift_vec_"):
                        pred.append(int(doc_id.split("_")[-1]))
                    elif doc_id.startswith("doc_"):
                        # doc_{idx} pattern — the idx is 1-based
                        pred.append(int(doc_id.split("_")[-1]) - 1)
                # Pad if fewer results than max_k
                while len(pred) < max_k:
                    pred.append(-1)
                all_pred_ids.append(pred[:max_k])

        t_search.iterations = queries.shape[0]
        qps = t_search.ops_per_second

        pred_array = np.array(all_pred_ids, dtype=np.int64)
        recalls = {}
        for k in RECALL_K_VALUES:
            if k <= max_k:
                recalls[f"recall@{k}"] = round(compute_recall(pred_array, ground_truth, k), 4)

        mem = measure_memory_mb()
        db.close()

    return {
        "path": "lvdb",
        "index_type": index_type,
        "scale": scale,
        "build_time_s": round(t_build.elapsed_seconds, 4),
        "search_time_s": round(t_search.elapsed_seconds, 4),
        "qps": round(qps, 1),
        "latency": latencies.to_dict(),
        "memory_mb": round(mem, 1),
        **recalls,
    }


def run_tier1(
    scales: List[int] | None = None,
    index_types: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Run all Tier 1 ANN benchmarks.

    Returns a list of result dicts.
    """
    from .config import ANN_SCALES

    scales = scales or ANN_SCALES
    index_types = index_types or INDEX_TYPES
    max_k = max(RECALL_K_VALUES)

    # Download and load dataset
    sift_path = download_sift()
    train, test = load_sift(sift_path)

    results = []
    for scale in scales:
        logger.info("=== Scale: %d ===", scale)
        subset = train[:scale]

        # Recompute ground truth for this subset
        # For IP index, we need ground truth on normalized vectors
        gt_l2 = compute_ground_truth(subset, test, max_k)

        for index_type in index_types:
            # For IP, recompute ground truth on normalized vectors
            if index_type == "IndexFlatIP":
                norm_subset = _maybe_normalize(subset.copy(), index_type)
                norm_test = _maybe_normalize(test.copy(), index_type)
                gt = compute_ground_truth(norm_subset, norm_test, max_k)
            else:
                gt = gt_l2

            # Path A: Raw FAISS
            try:
                r = run_raw_faiss(train, test, gt, scale, index_type)
                results.append(r)
                logger.info("    Raw FAISS QPS=%.0f, recall@10=%.4f", r["qps"], r.get("recall@10", 0))
            except Exception as e:
                logger.error("    Raw FAISS failed for %s/%d: %s", index_type, scale, e)

            # Path B: Through LVDB
            try:
                r = run_through_lvdb(train, test, gt, scale, index_type)
                results.append(r)
                logger.info("    LVDB QPS=%.0f, recall@10=%.4f", r["qps"], r.get("recall@10", 0))
            except Exception as e:
                logger.error("    LVDB failed for %s/%d: %s", index_type, scale, e)

    return results
