"""Rebuild the 601-letter human promoter windows from EPDnew BED + hg38 and
compare them with the Hadad et al. positives.

Reads:  data/raw/epdnew/Hs_EPDnew_006_hg38.bed, data/raw/hg38.fa,
        data/raw/hadad/positive_data.csv, data/raw/hadad/*_substitution.csv
Writes: data/processed/epd_human_positives.csv
        data/processed/hg38_validation_report.json
"""
import json
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
HALF = 300  # 300 before + TSS + 300 after = 601 letters
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
SPLIT_FILES = [
    "train_non_tata_substitution.csv", "train_tata_substitution.csv",
    "val_non_tata_substitution.csv", "val_tata_substitution.csv",
    "test_non_tata_substitution.csv", "test_tata_substitution.csv",
]

def revcomp(s):
    return s.translate(COMP)[::-1]

def window(genome, chrom, tss0, strand):
    a, b = tss0 - HALF, tss0 + HALF + 1
    if chrom not in genome or a < 0 or b > len(genome[chrom]):
        return None
    s = str(genome[chrom][a:b])
    return revcomp(s) if strand == "-" else s

def tss_from_row(r, off):
    # 0-based TSS; off > 0 shifts downstream (in the direction of transcription)
    return r.start + 49 + off if r.strand == "+" else r.end - 50 - off

def load_hadad():
    pos = pd.read_csv(RAW / "hadad" / "positive_data.csv", usecols=["ID", "seq", "organism"])
    pos = pos[pos["organism"] == "human"].copy()
    pos["seq"] = pos["seq"].str.upper()
    split_seqs, tata_seqs = set(), set()
    for name in SPLIT_FILES:
        df = pd.read_csv(RAW / "hadad" / name, usecols=["organism", "seq", "True_label"])
        df = df[(df["organism"] == "human") & (df["True_label"].astype(int) == 1)]
        s = set(df["seq"].str.upper())
        split_seqs |= s
        if "non_tata" not in name:
            tata_seqs |= s
    return pos, set(pos["seq"]), split_seqs, tata_seqs

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bed = pd.read_csv(
        RAW / "epdnew" / "Hs_EPDnew_006_hg38.bed", sep=r"\s+", header=None,
        names=["chrom", "start", "end", "epd_name", "score", "strand", "thick_start", "thick_end"],
    )
    print("BED rows:", len(bed))
    pos, pos_seqs, split_seqs, tata_seqs = load_hadad()
    genome = Fasta(str(RAW / "hg38.fa"))

    # 1) find the TSS offset that reproduces the Hadad sequences
    sample = bed.sample(1500, random_state=0)
    offsets = {}
    for off in range(-2, 3):
        hits = 0
        for r in sample.itertuples():
            w = window(genome, r.chrom, tss_from_row(r, off), r.strand)
            hits += int(w is not None and w.upper() in pos_seqs)
        offsets[off] = hits
        print(f"offset {off:+d}: {hits}/{len(sample)} windows found in Hadad positives")
    best = max(offsets, key=offsets.get)
    if offsets[best] < 0.5 * len(sample):
        raise SystemExit(
            "Fewer than half of the sampled windows match any Hadad positive at every "
            "offset. Stop and paste this output to the planner."
        )
    print("using offset", best)

    # 2) build all windows
    rows, skipped = [], 0
    for r in bed.itertuples():
        t = tss_from_row(r, best)
        w = window(genome, r.chrom, t, r.strand)
        if w is None:
            skipped += 1
            continue
        rows.append((r.epd_name, r.chrom, r.strand, int(t),
                     sum(c.islower() for c in w) / len(w), w.upper()))
    epd = pd.DataFrame(rows, columns=["epd_name", "chrom", "strand", "tss0", "repeat_frac", "seq"])
    epd["in_positive_data"] = epd["seq"].isin(pos_seqs)
    epd["in_split_files"] = epd["seq"].isin(split_seqs)
    epd["in_tata_files"] = epd["seq"].isin(tata_seqs)
    epd_seqs = set(epd["seq"])

    report = {
        "bed_rows": int(len(bed)),
        "windows_built": int(len(epd)),
        "skipped_off_chromosome_or_missing": int(skipped),
        "best_offset": int(best),
        "offset_hits_in_sample_of_1500": {str(k): int(v) for k, v in offsets.items()},
        "unique_windows": len(epd_seqs),
        "windows_containing_N": int(epd["seq"].str.contains("N").sum()),
        "mean_repeat_frac": round(float(epd["repeat_frac"].mean()), 4),
        "epd_windows_in_positive_data": int(epd["in_positive_data"].sum()),
        "epd_windows_in_split_files": int(epd["in_split_files"].sum()),
        "epd_windows_in_tata_files": int(epd["in_tata_files"].sum()),
        "hadad_positive_rows": int(len(pos)),
        "hadad_positive_unique_seqs": len(pos_seqs),
        "hadad_positive_rows_not_matching_any_epd_window": int((~pos["seq"].isin(epd_seqs)).sum()),
        "split_file_seqs_not_matching_any_epd_window": len(split_seqs - epd_seqs),
    }
    epd.to_csv(OUT / "epd_human_positives.csv", index=False)
    with open(OUT / "hg38_validation_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
