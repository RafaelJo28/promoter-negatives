"""Check 1 report: how well does a negative set match the real promoters?

Usage: python scripts/check_matching.py --neg data/processed/recipe_c_v1.csv --name C
Reads:  data/processed/human_positive_split_v1.csv and the negative CSV
        (columns neg_id, split, seq)
Writes: data/processed/check1_<name>.json
Thresholds are the DRAFT values in recipes/metrics.py (not frozen until approved).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recipes import metrics as M  # noqa: E402
from recipes.windows import load_real_windows  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    real = load_real_windows(ROOT)[["epd_name", "split", "seq"]]
    assert real["seq"].str.contains("[acgt]").any(), "real windows lost their case"
    neg = pd.read_csv(ROOT / args.neg, usecols=["neg_id", "split", "seq"])
    print(f"real promoters: {len(real)}   negatives ({args.name}): {len(neg)}")

    fr = M.feature_table(real["seq"])
    fn = M.feature_table(neg["seq"])
    matching = M.compare_features(fr, fn)
    worst_smd = max(matching.items(), key=lambda kv: abs(kv[1]["smd"]))
    worst_ks = max(matching.items(), key=lambda kv: kv[1]["ks"])
    match_pass = all(abs(v["smd"]) <= M.SMD_MAX and v["ks"] <= M.KS_MAX for v in matching.values())

    y = np.r_[np.ones(len(real)), np.zeros(len(neg))]
    split = np.r_[real["split"].to_numpy(), neg["split"].to_numpy()]
    seqs = list(real["seq"]) + list(neg["seq"])
    X_gc = np.vstack([fr[["gc", "cpg"]].to_numpy(), fn[["gc", "cpg"]].to_numpy()])

    controls = {"gc_cpg": M.lr_auroc(X_gc, y, split)}
    controls["pairs16"] = M.lr_auroc(M.kmer_table(seqs, M.DINUCS), y, split)
    controls["trinuc64_extra"] = M.lr_auroc(M.kmer_table(seqs, M.TRINUCS), y, split)

    gc_ok = controls["gc_cpg"]["test"] <= M.AUROC_MAX
    pairs_ok = controls["pairs16"]["test"] <= M.AUROC_MAX
    check1_pass = bool(match_pass and gc_ok and (pairs_ok if args.name == "C" else True))

    result = {
        "name": args.name,
        "n_real": int(len(real)),
        "n_neg": int(len(neg)),
        "thresholds": {"smd_max": M.SMD_MAX, "ks_max": M.KS_MAX, "auroc_max": M.AUROC_MAX},
        "matching": matching,
        "match_pass": bool(match_pass),
        "controls_auroc": controls,
        "check1_pass_draft": check1_pass,
    }
    out_path = ROOT / "data" / "processed" / f"check1_{args.name}.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\nmatching: {len(matching)} comparisons, pass={match_pass}")
    print(f"  worst |SMD| = {abs(worst_smd[1]['smd']):.4f}  ({worst_smd[0]})")
    print(f"  worst KS    = {worst_ks[1]['ks']:.4f}  ({worst_ks[0]})")
    print("\ncontrol AUROCs (logistic regression, real vs negatives):")
    for name, v in controls.items():
        print(f"  {name:16s} train {v['train']:.3f}  val {v['val']:.3f}  test {v['test']:.3f}")
    print(f"\nCHECK 1 (draft thresholds): {'PASS' if check1_pass else 'FAIL'}")
    print(f"full details: {out_path}")

if __name__ == "__main__":
    main()
