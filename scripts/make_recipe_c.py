"""Build Recipe C negatives (pair-preserving block shuffle), one per real promoter.

Reads:  configs/config.yaml, splits/human_split_v1.csv, data/raw/hg38.fa
        (sanity check against data/processed/human_positive_split_v1.csv if present)
Writes: data/processed/recipe_c_v1.csv      (neg_id, source_promoter, split, seq)
        data/processed/recipe_c_report.json
Each negative inherits the split of its source promoter.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from pyfaidx import Fasta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipes.features import window_features  # noqa: E402
from recipes.recipe_c import block_bounds, make_negative  # noqa: E402
from recipes.windows import get_window  # noqa: E402

def pair_counts(s):
    s = s.upper()
    return Counter(s[i:i + 2] for i in range(len(s) - 1))

def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    seed = cfg["project"]["base_seed"]
    block_size = cfg["recipe_c"]["block_size"]
    genome = Fasta(str(ROOT / cfg["paths"]["hg38_fasta"]))
    sp = pd.read_csv(ROOT / "splits" / "human_split_v1.csv")

    ref_path = ROOT / "data" / "processed" / "human_positive_split_v1.csv"
    ref = pd.read_csv(ref_path, usecols=["epd_name", "seq"]) if ref_path.exists() else None
    if ref is not None:
        assert len(ref) == len(sp)
        assert (ref["epd_name"].to_numpy() == sp["epd_name"].to_numpy()).all()

    rows, agg = [], Counter()
    d_gc, d_cpg, d_z1, d_z2, d_z3 = [], [], [], [], []
    for i, r in enumerate(sp.itertuples(index=False)):
        w = get_window(genome, r.chrom, int(r.tss0), r.strand)
        assert w is not None and len(w) == 601, f"bad window at row {i}"
        if ref is not None:
            assert w.upper() == ref["seq"].iat[i], f"window mismatch at row {i}"
        rng = random.Random(f"{seed}|C|{r.epd_name}|{r.tss0}")
        neg = make_negative(w, rng, block_size)

        agg["identical_to_source"] += int(neg.upper() == w.upper())
        agg["whole_window_pairs_equal"] += int(pair_counts(neg) == pair_counts(w))
        agg["block_pairs_equal"] += int(
            all(pair_counts(neg[a:b]) == pair_counts(w[a:b]) for a, b in block_bounds(601, block_size))
        )
        agg["mask_equal"] += int(all(x.islower() == y.islower() for x, y in zip(neg, w)))
        fw, fn = window_features(w), window_features(neg)
        d_gc.append(abs(fw["gc"] - fn["gc"]))
        d_cpg.append(abs(fw["cpg"] - fn["cpg"]))
        d_z1.append(abs(fw["gc_zone1"] - fn["gc_zone1"]))
        d_z2.append(abs(fw["gc_zone2"] - fn["gc_zone2"]))
        d_z3.append(abs(fw["gc_zone3"] - fn["gc_zone3"]))
        rows.append((f"C{i:05d}", r.epd_name, r.split, neg))

    n = len(rows)
    out = pd.DataFrame(rows, columns=["neg_id", "source_promoter", "split", "seq"])
    out.to_csv(ROOT / "data" / "processed" / "recipe_c_v1.csv", index=False)
    report = {
        "seed": int(seed),
        "block_size": int(block_size),
        "block_bounds": block_bounds(601, block_size),
        "n_negatives": n,
        "identical_to_source": agg["identical_to_source"],
        "whole_window_pairs_equal": agg["whole_window_pairs_equal"],
        "block_pairs_equal": agg["block_pairs_equal"],
        "mask_equal": agg["mask_equal"],
        "max_abs_gc_diff": round(max(d_gc), 6),
        "max_abs_cpg_count_diff": int(max(d_cpg)),
        "mean_abs_gc_diff_zone1": round(sum(d_z1) / n, 5),
        "mean_abs_gc_diff_zone2": round(sum(d_z2) / n, 5),
        "mean_abs_gc_diff_zone3": round(sum(d_z3) / n, 5),
        "max_abs_gc_diff_zone2": round(max(d_z2), 4),
        "max_abs_gc_diff_zone3": round(max(d_z3), 4),
        "split_counts": {k: int(v) for k, v in out["split"].value_counts().items()},
    }
    with open(ROOT / "data" / "processed" / "recipe_c_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    if not (
        agg["identical_to_source"] == 0
        and agg["whole_window_pairs_equal"] == n
        and agg["block_pairs_equal"] == n
        and agg["mask_equal"] == n
    ):
        raise SystemExit("Recipe C verification FAILED, paste the report to the planner.")

if __name__ == "__main__":
    main()
