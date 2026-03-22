"""Benchmark configuration constants."""

from pathlib import Path

# Paths
BENCHMARK_DIR = Path(__file__).parent
CACHE_DIR = BENCHMARK_DIR / ".cache"
RESULTS_DIR = BENCHMARK_DIR / "results"

# SIFT-128 dataset
SIFT_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
SIFT_FILENAME = "sift-128-euclidean.hdf5"
SIFT_DIMENSION = 128

# Tier 1: ANN benchmark scales
ANN_SCALES = [10_000, 25_000, 50_000]

# Tier 1: FAISS index types to benchmark
INDEX_TYPES = ["IndexFlatL2", "IndexFlatIP", "IndexHNSWFlat", "IndexLSH"]

# Tier 1: Recall@k values
RECALL_K_VALUES = [1, 10, 100]

# Tier 1: Number of test queries to use
NUM_TEST_QUERIES = 10_000

# Tier 2: Full-stack benchmark scales
FULLSTACK_SCALES = [1_000, 5_000, 10_000, 25_000, 50_000]

# Tier 2: Query benchmark parameters
NUM_QUERY_ITERATIONS = 200
QUERY_K = 10
INSERT_BATCH_SIZE = 100

# Tier 2: Multi-database scenario
MULTI_DB_COUNT = 10
MULTI_DB_DOCS_PER_DB = 5_000

# Tier 2: Metadata categories for filtered search
METADATA_CATEGORIES = ["A", "B", "C", "D", "E"]
METADATA_PRIORITY_RANGE = (1, 10)

# Mock embedding dimension (matches SIFT for consistency)
MOCK_DIMENSION = 128

# Document generation
WORDS_PER_DOC = 100
