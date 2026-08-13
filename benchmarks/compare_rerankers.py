"""Rank rerankers against EACH OTHER, not each against its own baseline.

Why this exists: eval_rerank.py reports every model as a delta over the first
stage, each with its own CI. Reading a ranking off those is not a paired test --
two models whose CIs overlap may still differ significantly per query, and two
whose deltas differ may not. Cohere v3.5 (+0.0458) and 4-fast (+0.0448) are
exactly that case: a 0.0010 gap inside both intervals, which says nothing.

Every artifact scores the SAME queries under the SAME first stage, so the
reranked per-query vectors pair directly. That comparison is free -- no API
calls, no embedding -- and it is the only honest way to say model A beats B.

FIRST-STAGE IDENTITY CHECK: because the first stage is held fixed across models,
the `first` column must be identical in every artifact. If it is not, retrieval
was nondeterministic and NO cross-model claim survives. That check runs before
any ranking is printed.

Usage:
    python benchmarks/compare_rerankers.py [--dataset qasper] [--method frequency_boost]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_section_bm25 import paired  # noqa: E402


def load(dataset: str, search_type: str) -> Dict[str, dict]:
    pattern = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "experiments",
        f"rerank_{dataset}_{search_type}_*.json",
    )
    out = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            d = json.load(fh)
        cfg = d["config"]
        # Key on the model PLUS any knob that makes a run distinct. Keying on
        # config["model"] alone silently dropped an arm: the `--max-length 256`
        # control and the 512 run it controls carry the same model string, and
        # older artifacts do not record max_length at all (only the filename
        # does), so one overwrote the other depending on glob order.
        label = cfg["model"]
        ml = cfg.get("max_length")
        if ml is None and "__len" in os.path.basename(path):  # pre-fix artifact
            ml = int(os.path.basename(path).split("__len")[1].split("_")[0])
        if ml is not None and ml != 512:
            label += f" @len{ml}"
        if cfg.get("max_queries"):
            label += f" @max{cfg['max_queries']}"
        if label in out:
            raise SystemExit(f"duplicate artifact label {label!r} ({path}) -- refusing to silently drop an arm")
        out[label] = d
    return out


def check_first_stage(arts: Dict[str, dict], key: str) -> bool:
    """The first stage is held fixed; its per-query scores must match exactly."""
    ref_name, ref = next(iter(arts.items()))
    ref_qids, ref_first = ref["qids"], np.asarray(ref["per_query"][f"{key}|first"])
    ok = True
    for name, art in arts.items():
        if art["qids"] != ref_qids:
            print(f"  !! {name}: qid ORDER differs from {ref_name} -- pairing would be wrong")
            ok = False
            continue
        first = np.asarray(art["per_query"][f"{key}|first"])
        if not np.allclose(first, ref_first, atol=0, rtol=0):
            drift = float(np.abs(first - ref_first).max())
            print(f"  !! {name}: first stage differs from {ref_name}, max |drift| {drift:.6f}")
            ok = False
    print(f"  first-stage identity across {len(arts)} artifacts: {'OK' if ok else 'VIOLATED'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="qasper")
    p.add_argument("--search-type", default="hybrid")
    p.add_argument("--method", default="frequency_boost", choices=["frequency_boost", "best"])
    p.add_argument("--pool", type=int, default=40)
    args = p.parse_args()

    arts = load(args.dataset, args.search_type)
    if len(arts) < 2:
        print(f"need >=2 artifacts, found {len(arts)}")
        return 1
    key = f"pool={args.pool}|{args.method}"

    print(f"\n{args.dataset} / {args.search_type} / {key}  ({len(arts)} models)\n")
    check_first_stage(arts, key)

    scores = {m: np.asarray(a["per_query"][f"{key}|rerank"]) for m, a in arts.items()}
    order: List[str] = sorted(scores, key=lambda m: -scores[m].mean())

    print(f"\n{'model':<44}{'nDCG@10':>9}")
    for m in order:
        print(f"{m:<44}{scores[m].mean():>9.4f}")

    print("\npairwise (row - col), * = 95% CI excludes zero\n")
    print(f"{'':<44}" + "".join(f"{m.split('/')[-1][:14]:>16}" for m in order))
    for a in order:
        cells = ""
        for b in order:
            if a == b:
                cells += f"{'--':>16}"
                continue
            st = paired(scores[a], scores[b])
            star = "*" if (st["ci_lo"] > 0 or st["ci_hi"] < 0) else " "
            cells += f"{st['delta']:>+15.4f}{star}"
        print(f"{a:<44}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
