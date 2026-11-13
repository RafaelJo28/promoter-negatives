"""Cut case-preserving 601-letter windows from hg38 (lowercase = soft-masked repeat)."""

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
HALF = 300  # 300 before + TSS + 300 after = 601 letters

def get_window(genome, chrom, tss0, strand, half=HALF):
    """tss0 is the 0-based TSS position. Returns None if the window leaves the chromosome."""
    a, b = tss0 - half, tss0 + half + 1
    if chrom not in genome or a < 0 or b > len(genome[chrom]):
        return None
    s = str(genome[chrom][a:b])
    return s.translate(COMP)[::-1] if strand == "-" else s

def load_real_windows(root):
    """Real promoter windows with case preserved (lowercase = soft-masked repeat).

    Returns the rows of splits/human_split_v1.csv plus a 'seq' column, rebuilt from hg38.
    """
    import pandas as pd
    import yaml
    from pyfaidx import Fasta

    cfg = yaml.safe_load(open(root / "configs" / "config.yaml"))
    genome = Fasta(str(root / cfg["paths"]["hg38_fasta"]))
    sp = pd.read_csv(root / "splits" / "human_split_v1.csv")
    seqs = [get_window(genome, r.chrom, int(r.tss0), r.strand) for r in sp.itertuples(index=False)]
    assert all(s is not None and len(s) == 601 for s in seqs)
    sp = sp.copy()
    sp["seq"] = seqs
    return sp
