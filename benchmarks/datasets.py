"""SIFT-128 dataset download, loading, and ground truth recomputation."""

import logging
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np

from .config import CACHE_DIR, NUM_TEST_QUERIES, RECALL_K_VALUES, SIFT_DIMENSION, SIFT_FILENAME, SIFT_URL

logger = logging.getLogger(__name__)


def ensure_cache_dir() -> Path:
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def download_sift(force: bool = False) -> Path:
    """Download the SIFT-128 HDF5 dataset if not already cached.

    Uses httpx (already a project dependency) for the download.
    """
    dest = ensure_cache_dir() / SIFT_FILENAME
    if dest.exists() and not force:
        logger.info("SIFT dataset already cached at %s", dest)
        return dest

    import httpx

    logger.info("Downloading SIFT-128 dataset from %s ...", SIFT_URL)
    with httpx.stream("GET", SIFT_URL, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(
                        f"\r  Download progress: {pct}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)",
                        end="",
                        flush=True,
                    )
    print()
    logger.info("Download complete: %s", dest)
    return dest


def load_sift(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load train vectors and test queries from the SIFT HDF5 file.

    Returns
    -------
    train : np.ndarray, shape (N, 128), float32
    test : np.ndarray, shape (Q, 128), float32
    """
    import h5py

    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"][:NUM_TEST_QUERIES], dtype=np.float32)
    logger.info("Loaded SIFT: train=%s, test=%s", train.shape, test.shape)
    return train, test


def compute_ground_truth(train_subset: np.ndarray, test: np.ndarray, max_k: int | None = None) -> np.ndarray:
    """Recompute exact ground truth for a subset of training vectors.

    The HDF5 ground truth is against the full 1M vectors; when we benchmark
    with 10K/25K/50K subsets we need fresh ground truth.

    Parameters
    ----------
    train_subset : np.ndarray, shape (n, d)
    test : np.ndarray, shape (q, d)
    max_k : int, optional
        Maximum k to compute neighbors for. Defaults to max of RECALL_K_VALUES.

    Returns
    -------
    neighbors : np.ndarray, shape (q, max_k), int64 — ground truth IDs
    """
    if max_k is None:
        max_k = max(RECALL_K_VALUES)
    logger.info(
        "Computing ground truth for %d vectors, %d queries, k=%d ...",
        train_subset.shape[0],
        test.shape[0],
        max_k,
    )
    index = faiss.IndexFlatL2(SIFT_DIMENSION)
    index.add(train_subset)
    _, gt_ids = index.search(test, max_k)
    return gt_ids


def compute_recall(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k.

    Parameters
    ----------
    predicted : np.ndarray, shape (q, k_pred) — predicted neighbor IDs
    ground_truth : np.ndarray, shape (q, k_gt) — ground truth IDs (must have >= k columns)
    k : int

    Returns
    -------
    float — mean recall@k across all queries
    """
    assert ground_truth.shape[1] >= k, f"Ground truth has {ground_truth.shape[1]} cols, need >= {k}"
    gt_k = ground_truth[:, :k]
    pred_k = predicted[:, :k]

    recalls = []
    for i in range(gt_k.shape[0]):
        gt_set = set(gt_k[i].tolist())
        pred_set = set(pred_k[i].tolist())
        if len(gt_set) > 0:
            recalls.append(len(gt_set & pred_set) / len(gt_set))
    return float(np.mean(recalls))
