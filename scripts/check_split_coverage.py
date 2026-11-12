"""Check how the published Hadad et al. split covers the human EPDnew promoters.

Reads:  data/raw/hadad/positive_data.csv and the six *_substitution.csv files
Writes: data/processed/human_positive_split.csv
        data/processed/split_coverage_report.json
Matching is done on the 601-letter sequence; the ID is only cross-checked.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hadad"
OUT = ROOT / "data" / "processed"

EXPECTED_HUMAN = 29598
EXPECTED_TATA = 3065

SPLIT_FILES = [
    ("train", False, "train_non_tata_substitution.csv"),
    ("train", True, "train_tata_substitution.csv"),
    ("val", False, "val_non_tata_substitution.csv"),
    ("val", True, "val_tata_substitution.csv"),
    ("test", False, "test_non_tata_substitution.csv"),
    ("test", True, "test_tata_substitution.csv"),
]

def n_items(v):
    return len(v) if isinstance(v, list) else 0

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    pos = pd.read_csv(RAW / "positive_data.csv")
    pos = pos[pos["organism"] == "human"].copy()
    pos["seq"] = pos["seq"].str.upper()
    print(f"human promoters in positive_data.csv: {len(pos)}")

    parts = []
    for split, is_tata, fname in SPLIT_FILES:
        df = pd.read_csv(RAW / fname, usecols=["organism", "ID", "seq", "True_label"])
        hp = df[(df["organism"] == "human") & (df["True_label"].astype(int) == 1)].copy()
        print(f"{fname}: {len(df)} rows, {len(hp)} human positives")
        if len(hp) == 0:
            print("   organisms present:", sorted(df["organism"].unique()))
        hp["split"] = split
        hp["is_tata"] = is_tata
        parts.append(hp)
    sp = pd.concat(parts, ignore_index=True)
    sp["seq"] = sp["seq"].str.upper()

    per_seq = sp.groupby("seq").agg(
        n_rows=("split", "size"),
        splits=("split", lambda s: sorted(set(s))),
        tata_flags=("is_tata", lambda s: sorted(set(s))),
        ids=("ID", lambda s: sorted(set(s))),
    )
    pos = pos.join(per_seq, on="seq")
    covered = pos["n_rows"].notna()

    multi_split = pos["splits"].apply(n_items) > 1
    multi_tata = pos["tata_flags"].apply(n_items) > 1
    id_mismatch = covered & ~pos.apply(
        lambda r: isinstance(r["ids"], list) and r["ID"] in r["ids"], axis=1
    )
    clean = covered & ~multi_split

    pos["split"] = "missing"
    pos.loc[clean, "split"] = pos.loc[clean, "splits"].apply(lambda v: v[0])
    pos.loc[multi_split, "split"] = "conflict"
    pos["is_tata"] = None
    ok_tata = clean & ~multi_tata
    pos.loc[ok_tata, "is_tata"] = pos.loc[ok_tata, "tata_flags"].apply(lambda v: v[0])

    extra = set(per_seq.index) - set(pos["seq"])
    split_counts = pos["split"].value_counts().to_dict()

    report = {
        "n_human_positives": int(len(pos)),
        "duplicate_ids_in_human_positives": int(pos["ID"].duplicated().sum()),
        "duplicate_seqs_in_human_positives": int(pos["seq"].duplicated().sum()),
        "covered_by_split_files": int(covered.sum()),
        "missing_from_split_files": int((~covered).sum()),
        "in_more_than_one_split": int(multi_split.sum()),
        "in_both_tata_and_non_tata": int(multi_tata.sum()),
        "id_mismatch_on_sequence_match": int(id_mismatch.sum()),
        "split_positives_not_in_positive_data": len(extra),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "n_tata_assigned": int(pos["is_tata"].eq(True).sum()),
        "expected_human": EXPECTED_HUMAN,
        "expected_tata": EXPECTED_TATA,
    }
    full = (
        len(pos) == EXPECTED_HUMAN
        and report["missing_from_split_files"] == 0
        and report["in_more_than_one_split"] == 0
    )
    report["verdict"] = "FULL COVERAGE" if full else "NOT FULL COVERAGE - build own split"

    pos[["ID", "split", "is_tata", "seq"]].to_csv(OUT / "human_positive_split.csv", index=False)
    with open(OUT / "split_coverage_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
