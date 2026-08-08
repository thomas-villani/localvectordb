"""Cross-corpus read of the aggregation sweeps: is the aggregator a property of
the TARGET UNIT rather than of the corpus?

Every prior knob in this study came out corpus-dependent (`section_weight`,
`vector_weight`, chunk size), which makes "another per-corpus argmax table" the
default expectation and therefore the thing to test against. The sweeps store
per-query nDCG for every cell, so every contrast below is a paired re-read of
saved vectors -- no re-run, no re-embedding.

Two contrasts, both pre-specified:

* `sum@2` vs the shipped operator -- the summing aggregator, hypothesised to help
  on document targets and hurt on section targets.
* `freq@0.3` vs `max` on SECTION targets specifically -- sections ship with a hard
  max and no knob, and `freq@0.3` was the post-hoc argmax on all three corpora.
  A post-hoc argmax needs its own CI before it means anything, which the sweep's
  own report deliberately does not give it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.eval_section_bm25 import paired  # noqa: E402

SHIPPED_BY_TARGET = {"doc": "freq@0.3", "section": "max"}


def verdict(stat: Dict[str, float]) -> str:
    if stat["ci_lo"] > 0:
        return "WIN "
    if stat["ci_hi"] < 0:
        return "LOSS"
    return "null"


def contrast(swept: Dict, pool: int, arm: str, base: str) -> Dict[str, float]:
    a = np.asarray(swept["per_query"][f"pool={pool}|{arm}"])
    b = np.asarray(swept["per_query"][f"pool={pool}|{base}"])
    return paired(a, b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", type=Path, nargs="+", required=True)
    ap.add_argument("--pools", type=int, nargs="+", default=[40, 400])
    args = ap.parse_args()

    loaded = []
    for path in args.files:
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded.append((data["config"]["dataset"], data))

    for pool in args.pools:
        print(f"\n{'='*96}\n=== sum@2 vs SHIPPED, pool={pool} -- does the sign track the TARGET UNIT? ===")
        print(f"  {'corpus':<8} {'target':<8} {'leg':<7} {'shipped':<10} {'delta':>9}  {'95% CI':>22}  verdict")
        rows: Dict[str, List[str]] = {"doc": [], "section": []}
        for name, data in loaded:
            for key, swept in data["arms"].items():
                tname, leg = key.split("|")
                base = SHIPPED_BY_TARGET[tname]
                if f"pool={pool}|sum@2" not in swept["per_query"]:
                    continue
                stat = contrast(swept, pool, "sum@2", base)
                rows[tname].append(
                    f"  {name:<8} {tname:<8} {leg:<7} {base:<10} {stat['delta']:>+9.4f}  "
                    f"[{stat['ci_lo']:>+8.4f},{stat['ci_hi']:>+8.4f}]  {verdict(stat)}"
                )
        for tname in ("doc", "section"):
            for row in sorted(rows[tname]):
                print(row)
            if rows[tname]:
                print()

        print(f"=== freq@0.3 vs max on SECTION targets, pool={pool} -- sections ship a hard max ===")
        print(f"  {'corpus':<8} {'leg':<7} {'delta':>9}  {'95% CI':>22}  {'p':>7}  verdict")
        for name, data in loaded:
            for key, swept in data["arms"].items():
                tname, leg = key.split("|")
                if tname != "section" or f"pool={pool}|freq@0.3" not in swept["per_query"]:
                    continue
                stat = contrast(swept, pool, "freq@0.3", "max")
                print(
                    f"  {name:<8} {leg:<7} {stat['delta']:>+9.4f}  "
                    f"[{stat['ci_lo']:>+8.4f},{stat['ci_hi']:>+8.4f}]  {stat['p']:>7.3f}  {verdict(stat)}"
                )

        # The one concrete src candidate: is document_scoring_method="best" the
        # better DEFAULT for search_type="vector"? Split by leg, because the whole
        # point is that the answer differs between them -- a single averaged
        # verdict would hide exactly the effect under test.
        print(f"\n=== max vs freq@0.3 on DOC targets, pool={pool} -- should 'vector' default to 'best'? ===")
        print(f"  {'corpus':<8} {'leg':<7} {'delta':>9}  {'95% CI':>22}  {'p':>7}  verdict")
        for name, data in loaded:
            for key, swept in data["arms"].items():
                tname, leg = key.split("|")
                if tname != "doc" or f"pool={pool}|freq@0.3" not in swept["per_query"]:
                    continue
                stat = contrast(swept, pool, "max", "freq@0.3")
                print(
                    f"  {name:<8} {leg:<7} {stat['delta']:>+9.4f}  "
                    f"[{stat['ci_lo']:>+8.4f},{stat['ci_hi']:>+8.4f}]  {stat['p']:>7.3f}  {verdict(stat)}"
                )


if __name__ == "__main__":
    main()
