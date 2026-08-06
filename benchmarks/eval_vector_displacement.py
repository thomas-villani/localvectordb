"""DIAGNOSTIC: how far did truncation actually move the stored vectors?

WHY THIS EXISTS. Every "coverage" number in this project is a CHARACTER proxy for
a TOKEN quantity, and it was answering the wrong question anyway. Two builds of
the same corpus that differ only in ``num_ctx`` give the exact answer directly:
compare the stored vectors. No estimate, no tokenizer, no constant.

WHAT IT FOUND (egemma, fiqa-superdocs density_g4, native 2048 vs num_ctx=512):

    c      vectors CHANGED   mean cos   delta nDCG@10
    219      0 / 10,421      1.000000     0.0000
    500  2,077 /  4,478      0.9972      -0.0000
    1000 1,713 /  2,478      0.9360      -0.0571
    1750   981 /  1,577      0.8967      -0.0396

The c=500 row is the point: **46% of chunks truncated, zero movement in nDCG at
four decimal places.** Truncation that clips a little off a chunk barely rotates
its vector, and retrieval does not notice. "Was it truncated" is the wrong
question; "how far did the vector move" is the right one. Damage on this leg
needs mean 1-cos above ~0.01.

The c=219 row does double duty: 10,421 vectors, zero drift at 1e-6, so this
encoder is deterministic ACROSS PROCESSES AND BUILDS. Without that control every
other row could be numerical noise, and the whole comparison would be vacuous.

WHY A CHAR CAP CANNOT REPLACE THIS. Chars per token is not constant across
chunks -- on this leg it spans at least 2.1 to 5.3 -- so where the length
distribution sits ON the cap, no char threshold reproduces which vectors moved.
Fitting the best single cap to the c=500 rung still misclassifies 31% of chunks;
on c=1000, where chunks sit far above the cap, the same fit is exact to 0.1%.
A char proxy is reliable precisely where it is not needed.

Zero embedding: both indexes are already built, and this only reads them.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import DATA_DIR  # noqa: E402

logger = logging.getLogger("displacement")

# Below this the two vectors are the same vector; fp reduction order alone moves
# a normalised 768-dim embedding by ~2e-8, four orders of magnitude under this.
SAME = 1e-6


def load(key: str) -> Tuple[np.ndarray, np.ndarray]:
    """(unit-normalised vectors, chunk char lengths) for one built index."""
    import faiss

    base = Path(DATA_DIR) / "db"
    index = faiss.read_index(str(base / f"{key}.faiss"))
    vecs = index.reconstruct_n(0, index.ntotal)
    conn = sqlite3.connect(f"file:{base / (key + '.sqlite')}?mode=ro", uri=True)
    try:
        lens = dict(conn.execute("SELECT faiss_id, LENGTH(content) FROM chunks").fetchall())
    finally:
        conn.close()
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms), np.array([lens.get(i, 0) for i in range(len(vecs))])


def compare(key_a: str, key_b: str) -> Dict[str, float]:
    va, lens = load(key_a)
    vb, _ = load(key_b)
    if va.shape != vb.shape:
        raise SystemExit(
            f"{key_a} has {va.shape} vectors, {key_b} has {vb.shape}. Different corpora or chunkers -- "
            "these are not paired builds and comparing them position-by-position is meaningless."
        )
    cos = np.einsum("ij,ij->i", va, vb)
    moved = (1.0 - cos) > SAME
    out = {
        "n": int(len(cos)),
        "n_moved": int(moved.sum()),
        "frac_moved": float(moved.mean()),
        "mean_cos": float(cos.mean()),
        "mean_displacement": float((1.0 - cos).mean()),
        "p99_displacement": float(np.percentile(1.0 - cos, 99)),
        "min_cos": float(cos.min()),
    }
    if moved.any():
        out["shortest_moved_chars"] = int(lens[moved].min())
        out["longest_unmoved_chars"] = int(lens[~moved].max()) if (~moved).any() else 0
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="baseline index key (no .faiss/.sqlite suffix)")
    p.add_argument("--b", required=True, help="index key to compare against it")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    r = compare(args.a, args.b)
    print(f"\n{args.a}\n  vs {args.b}\n")
    print(f"  vectors            {r['n']}")
    print(f"  moved (>1e-6)      {r['n_moved']}  ({100 * r['frac_moved']:.1f}%)")
    print(f"  mean cos           {r['mean_cos']:.6f}")
    print(f"  mean displacement  {r['mean_displacement']:.2e}   p99 {r['p99_displacement']:.2e}")
    print(f"  worst cos          {r['min_cos']:.6f}")
    if "shortest_moved_chars" in r:
        print(f"  shortest MOVED     {r['shortest_moved_chars']} chars")
        print(f"  longest UNMOVED    {r['longest_unmoved_chars']} chars")
        if r["longest_unmoved_chars"] > r["shortest_moved_chars"]:
            print(
                "  -> the two overlap, so NO character threshold separates truncated from intact here.\n"
                "     Chars/token varies across chunks; a char-based coverage number is guessing."
            )
    if r["mean_displacement"] < 0.01 and r["frac_moved"] > 0.1:
        print("\n  Many vectors moved, but all by very little. Expect no measurable retrieval change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
