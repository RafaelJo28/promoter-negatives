"""Build our own leak-aware train/val/test split of the EPDnew human promoters.

Reads:  data/processed/epd_human_positives.csv, configs/config.yaml
Writes: data/processed/human_positive_split_v1.csv   (with sequences, git-ignored)
        splits/human_split_v1.csv                    (no sequences, tracked in git)
        data/processed/split_report.json
Rules:  drop windows containing N; group windows that are identical or overlap
        on the same chromosome; assign whole groups to splits, stratified by TATA.
"""
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
TRACKED = ROOT / "splits"
RATIOS = {"train": 0.64, "val": 0.16, "test": 0.20}
MAX_TSS_DIST = 600  # two 601-letter windows overlap iff their TSSs are <= 600 apart

def find(p, i):
    while p[i] != i:
        p[i] = p[p[i]]
        i = p[i]
    return i

def union(p, a, b):
    ra, rb = find(p, a), find(p, b)
    if ra != rb:
        p[max(ra, rb)] = min(ra, rb)

def main():
    seed = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))["project"]["base_seed"]
    TRACKED.mkdir(exist_ok=True)
    df = pd.read_csv(OUT / "epd_human_positives.csv")
    n_total = len(df)
    df = df[~df["seq"].str.contains("N")].reset_index(drop=True)
    df["is_tata"] = df["in_tata_files"].astype(bool)

    # 1) union-find grouping
    p = list(range(len(df)))
    order = df.sort_values(["chrom", "tss0"]).index.to_list()
    for a, b in zip(order[:-1], order[1:]):
        if df.at[a, "chrom"] == df.at[b, "chrom"] and df.at[b, "tss0"] - df.at[a, "tss0"] <= MAX_TSS_DIST:
            union(p, a, b)
    first = {}
    for i, s in enumerate(df["seq"]):
        if s in first:
            union(p, first[s], i)
        else:
            first[s] = i
    df["group_id"] = [find(p, i) for i in range(len(df))]

    # 2) assign whole groups to splits, separately for TATA / non-TATA
    groups = df.groupby("group_id").agg(n=("seq", "size"), is_tata=("is_tata", "max")).reset_index()
    assignment = {}
    for flag in (False, True):
        g = groups[groups["is_tata"] == flag].sample(frac=1.0, random_state=seed)
        total = int(g["n"].sum())
        target = {s: r * total for s, r in RATIOS.items()}
        filled = {s: 0 for s in RATIOS}
        for gid, n in zip(g["group_id"], g["n"]):
            s = max(RATIOS, key=lambda k: target[k] - filled[k])
            assignment[gid] = s
            filled[s] += int(n)
    df["split"] = df["group_id"].map(assignment)

    # 3) leakage checks (fail loudly)
    assert df.groupby("group_id")["split"].nunique().max() == 1
    assert df.groupby("seq")["split"].nunique().max() == 1
    bad = 0
    for _, sub in df.sort_values(["chrom", "tss0"]).groupby("chrom"):
        t, s = sub["tss0"].to_numpy(), sub["split"].to_numpy()
        bad += int(((t[1:] - t[:-1] <= MAX_TSS_DIST) & (s[1:] != s[:-1])).sum())
    assert bad == 0, f"{bad} overlapping neighbours sit in different splits"

    # 4) write outputs
    df.to_csv(OUT / "human_positive_split_v1.csv", index=False)
    df[["epd_name", "chrom", "strand", "tss0", "group_id", "is_tata", "split"]].to_csv(
        TRACKED / "human_split_v1.csv", index=False
    )
    sizes = df.groupby("group_id").size()
    report = {
        "seed": int(seed),
        "windows_in": int(n_total),
        "dropped_for_N": int(n_total - len(df)),
        "windows_out": int(len(df)),
        "n_groups": int(len(sizes)),
        "groups_with_more_than_one_window": int((sizes > 1).sum()),
        "largest_group": int(sizes.max()),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().items()},
        "split_fractions": {k: round(float(v), 4) for k, v in df["split"].value_counts(normalize=True).items()},
        "tata_per_split": {k: int(v) for k, v in df[df["is_tata"]]["split"].value_counts().items()},
        "leakage_checks": "passed",
    }
    with open(OUT / "split_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
