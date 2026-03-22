"""Tier 2: Full-Stack Benchmarks.

Measures real-world performance through the complete LocalVectorDB API
with realistic document workloads.
"""

import gc
import logging
import random
import tempfile
import time
from typing import Any, Dict, List

from .config import (
    FULLSTACK_SCALES,
    INSERT_BATCH_SIZE,
    METADATA_CATEGORIES,
    METADATA_PRIORITY_RANGE,
    MOCK_DIMENSION,
    MULTI_DB_COUNT,
    MULTI_DB_DOCS_PER_DB,
    NUM_QUERY_ITERATIONS,
    QUERY_K,
    WORDS_PER_DOC,
)
from .utils import LatencyStats, measure_memory_mb, timer

logger = logging.getLogger(__name__)

# Word pool for generating realistic-ish documents
_WORD_POOL: List[str] = []


def _get_word_pool() -> List[str]:
    """Lazily build a pool of pseudo-random words."""
    global _WORD_POOL
    if not _WORD_POOL:
        rng = random.Random(42)  # nosec B311
        consonants = "bcdfghjklmnpqrstvwxyz"
        vowels = "aeiou"
        for _ in range(2000):
            length = rng.randint(3, 10)
            word = ""
            for j in range(length):
                word += rng.choice(vowels if j % 2 else consonants)
            _WORD_POOL.append(word)
    return _WORD_POOL


def _generate_document(rng: random.Random, n_words: int = WORDS_PER_DOC) -> str:
    """Generate a realistic-ish text document."""
    pool = _get_word_pool()
    words = [rng.choice(pool) for _ in range(n_words)]
    # Group into sentences of 8-15 words
    doc = []
    i = 0
    while i < len(words):
        sentence_len = rng.randint(8, 15)
        sentence = " ".join(words[i : i + sentence_len])
        sentence = sentence.capitalize() + "."
        doc.append(sentence)
        i += sentence_len
    return " ".join(doc)


def _generate_documents(n: int, seed: int = 42) -> List[str]:
    """Generate n text documents."""
    rng = random.Random(seed)  # nosec B311
    return [_generate_document(rng) for _ in range(n)]


def _generate_metadata(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate metadata dicts for n documents."""
    rng = random.Random(seed)  # nosec B311
    return [
        {
            "category": rng.choice(METADATA_CATEGORIES),
            "priority": rng.randint(*METADATA_PRIORITY_RANGE),
        }
        for _ in range(n)
    ]


def _create_db(tmpdir: str, name: str = "bench", enable_fts: bool = False, metadata_schema: dict | None = None):
    """Create a LocalVectorDB with mock embeddings."""
    from localvectordb.database import LocalVectorDB

    kwargs: Dict[str, Any] = {
        "name": name,
        "base_path": tmpdir,
        "embedding_provider": "mock",
        "embedding_model": "mock",
        "embedding_config": {"dimension": MOCK_DIMENSION},
        "chunk_size": 500,
        "chunk_overlap": 1,
        "enable_fts": enable_fts,
    }
    if metadata_schema:
        kwargs["metadata_schema"] = metadata_schema
    return LocalVectorDB(**kwargs)


def bench_insert_throughput(scales: List[int] | None = None) -> List[Dict[str, Any]]:
    """Benchmark: Insert throughput (docs/sec) at each scale."""
    scales = scales or FULLSTACK_SCALES
    results = []
    for scale in scales:
        logger.info("  Insert throughput: scale=%d", scale)
        docs = _generate_documents(scale)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _create_db(tmpdir)
            mem_before = measure_memory_mb()

            with timer("insert") as t:
                db.upsert(docs, batch_size=INSERT_BATCH_SIZE)
            t.iterations = scale

            mem_after = measure_memory_mb()
            db.close()

        results.append(
            {
                "benchmark": "insert_throughput",
                "scale": scale,
                "elapsed_s": round(t.elapsed_seconds, 3),
                "docs_per_sec": round(t.ops_per_second, 1),
                "memory_delta_mb": round(mem_after - mem_before, 1),
            }
        )
        logger.info("    %.1f docs/sec (%.2fs)", t.ops_per_second, t.elapsed_seconds)

    return results


def bench_query_latency(scales: List[int] | None = None) -> List[Dict[str, Any]]:
    """Benchmark: Vector query latency (p50/p95/p99) at each scale."""
    scales = scales or FULLSTACK_SCALES
    results = []
    for scale in scales:
        logger.info("  Query latency: scale=%d", scale)
        docs = _generate_documents(scale)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _create_db(tmpdir)
            db.upsert(docs, batch_size=INSERT_BATCH_SIZE)

            # Generate query texts
            rng = random.Random(99)  # nosec B311  # nosec B311
            query_texts = [_generate_document(rng, n_words=10) for _ in range(NUM_QUERY_ITERATIONS)]

            latencies = LatencyStats()
            for qt in query_texts:
                t0 = time.perf_counter()
                db.query(qt, k=QUERY_K, search_type="vector")
                latencies.timings_ms.append((time.perf_counter() - t0) * 1000)

            db.close()

        results.append(
            {
                "benchmark": "query_latency",
                "scale": scale,
                "search_type": "vector",
                **latencies.to_dict(),
            }
        )
        logger.info("    p50=%.2fms, p95=%.2fms, p99=%.2fms", latencies.p50, latencies.p95, latencies.p99)

    return results


def bench_hybrid_search(scales: List[int] | None = None) -> List[Dict[str, Any]]:
    """Benchmark: Hybrid search latency (vector + FTS5)."""
    scales = scales or FULLSTACK_SCALES
    results = []
    for scale in scales:
        logger.info("  Hybrid search: scale=%d", scale)
        docs = _generate_documents(scale)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _create_db(tmpdir, enable_fts=True)
            db.upsert(docs, batch_size=INSERT_BATCH_SIZE)

            rng = random.Random(99)  # nosec B311  # nosec B311
            query_texts = [_generate_document(rng, n_words=10) for _ in range(NUM_QUERY_ITERATIONS)]

            latencies = LatencyStats()
            for qt in query_texts:
                t0 = time.perf_counter()
                db.query(qt, k=QUERY_K, search_type="hybrid")
                latencies.timings_ms.append((time.perf_counter() - t0) * 1000)

            db.close()

        results.append(
            {
                "benchmark": "hybrid_search",
                "scale": scale,
                "search_type": "hybrid",
                **latencies.to_dict(),
            }
        )
        logger.info("    p50=%.2fms, p95=%.2fms, p99=%.2fms", latencies.p50, latencies.p95, latencies.p99)

    return results


def bench_filtered_search(scales: List[int] | None = None) -> List[Dict[str, Any]]:
    """Benchmark: Vector search with metadata filtering."""
    from localvectordb.core import MetadataField, MetadataFieldType

    scales = scales or FULLSTACK_SCALES
    results = []

    schema = {
        "category": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        "priority": MetadataField(type=MetadataFieldType.INTEGER, indexed=True),
    }

    for scale in scales:
        logger.info("  Filtered search: scale=%d", scale)
        docs = _generate_documents(scale)
        meta = _generate_metadata(scale)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _create_db(tmpdir, metadata_schema=schema)
            db.upsert(docs, metadata=meta, batch_size=INSERT_BATCH_SIZE)

            rng = random.Random(99)  # nosec B311  # nosec B311
            query_texts = [_generate_document(rng, n_words=10) for _ in range(NUM_QUERY_ITERATIONS)]

            latencies = LatencyStats()
            for qt in query_texts:
                t0 = time.perf_counter()
                db.query(qt, k=QUERY_K, search_type="vector", filters={"category": "A"})
                latencies.timings_ms.append((time.perf_counter() - t0) * 1000)

            db.close()

        results.append(
            {
                "benchmark": "filtered_search",
                "scale": scale,
                **latencies.to_dict(),
            }
        )
        logger.info("    p50=%.2fms, p95=%.2fms, p99=%.2fms", latencies.p50, latencies.p95, latencies.p99)

    return results


def bench_multi_database() -> Dict[str, Any]:
    """Benchmark: 10 independent DBs of 5K docs each, simulating agent multi-KB use case."""
    logger.info("  Multi-database scenario: %d DBs x %d docs", MULTI_DB_COUNT, MULTI_DB_DOCS_PER_DB)

    docs = _generate_documents(MULTI_DB_DOCS_PER_DB, seed=42)
    mem_before = measure_memory_mb()

    tmpdir = tempfile.mkdtemp()
    dbs = []
    insert_times = []

    for i in range(MULTI_DB_COUNT):
        db = _create_db(tmpdir, name=f"agent_db_{i}")
        with timer("insert") as t:
            db.upsert(docs, batch_size=INSERT_BATCH_SIZE)
        insert_times.append(t.elapsed_seconds)
        dbs.append(db)

    mem_after = measure_memory_mb()

    # Query round-robin
    rng = random.Random(99)  # nosec B311
    query_texts = [_generate_document(rng, n_words=10) for _ in range(NUM_QUERY_ITERATIONS)]

    latencies = LatencyStats()
    for i, qt in enumerate(query_texts):
        db = dbs[i % MULTI_DB_COUNT]
        t0 = time.perf_counter()
        db.query(qt, k=QUERY_K, search_type="vector")
        latencies.timings_ms.append((time.perf_counter() - t0) * 1000)

    for db in dbs:
        db.close()

    # Clean up
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)

    result = {
        "benchmark": "multi_database",
        "num_dbs": MULTI_DB_COUNT,
        "docs_per_db": MULTI_DB_DOCS_PER_DB,
        "total_docs": MULTI_DB_COUNT * MULTI_DB_DOCS_PER_DB,
        "total_insert_time_s": round(sum(insert_times), 3),
        "avg_insert_time_per_db_s": round(sum(insert_times) / MULTI_DB_COUNT, 3),
        "memory_total_mb": round(mem_after, 1),
        "memory_delta_mb": round(mem_after - mem_before, 1),
        "memory_per_db_mb": round((mem_after - mem_before) / MULTI_DB_COUNT, 1),
        "query_latency": latencies.to_dict(),
    }
    logger.info(
        "    Total memory: %.1fMB (%.1fMB/db), query p50=%.2fms",
        mem_after - mem_before,
        (mem_after - mem_before) / MULTI_DB_COUNT,
        latencies.p50,
    )
    return result


def bench_memory_footprint(scales: List[int] | None = None) -> List[Dict[str, Any]]:
    """Benchmark: Memory footprint (RSS) at each scale."""
    scales = scales or FULLSTACK_SCALES
    results = []
    for scale in scales:
        logger.info("  Memory footprint: scale=%d", scale)
        docs = _generate_documents(scale)

        gc.collect()
        mem_before = measure_memory_mb()

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _create_db(tmpdir)
            db.upsert(docs, batch_size=INSERT_BATCH_SIZE)
            gc.collect()
            mem_after = measure_memory_mb()
            db.close()

        results.append(
            {
                "benchmark": "memory_footprint",
                "scale": scale,
                "rss_mb": round(mem_after, 1),
                "delta_mb": round(mem_after - mem_before, 1),
            }
        )
        logger.info("    RSS=%.1fMB (delta=%.1fMB)", mem_after, mem_after - mem_before)

    return results


def run_tier2(scales: List[int] | None = None) -> Dict[str, Any]:
    """Run all Tier 2 full-stack benchmarks.

    Returns a dict keyed by benchmark name.
    """
    scales = scales or FULLSTACK_SCALES

    results: Dict[str, Any] = {}

    logger.info("--- Insert Throughput ---")
    results["insert_throughput"] = bench_insert_throughput(scales)

    logger.info("--- Query Latency ---")
    results["query_latency"] = bench_query_latency(scales)

    logger.info("--- Hybrid Search ---")
    results["hybrid_search"] = bench_hybrid_search(scales)

    logger.info("--- Filtered Search ---")
    results["filtered_search"] = bench_filtered_search(scales)

    logger.info("--- Multi-Database ---")
    results["multi_database"] = bench_multi_database()

    logger.info("--- Memory Footprint ---")
    results["memory_footprint"] = bench_memory_footprint(scales)

    return results
