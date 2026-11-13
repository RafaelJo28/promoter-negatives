"""Matching metrics and shortcut controls (Check 1).

Compares real promoters with a negative set (Recipe C or D):
  - standardized mean difference (SMD) and KS distance for GC, CpG and repeat share,
    overall, per zone, and per promoter-GC quintile bin
  - logistic-regression controls, trained on the train split and scored on val/test
THRESHOLDS BELOW ARE DRAFT VALUES (docs/decision_rules.md). Not frozen until approved.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from recipes.features import ZONES, window_features

SMD_MAX = 0.05
KS_MAX = 0.05
AUROC_MAX = 0.55
N_BINS = 5

DINUCS = ["".join(p) for p in itertools.product("ACGT", repeat=2)]
TRINUCS = ["".join(p) for p in itertools.product("ACGT", repeat=3)]

def feature_table(seqs):
    """gc, cpg, repeat_frac, plus per-zone gc, cpg and repeat share."""
    rows = []
    for s in seqs:
        f = window_features(s)
        u = s.upper()
        for k, (a, b) in enumerate(ZONES, start=1):
            f[f"cpg_zone{k}"] = u[a:b].count("CG")
            f[f"repeat_zone{k}"] = sum(c.islower() for c in s[a:b]) / (b - a)
        rows.append(f)
    return pd.DataFrame(rows)

def smd(a, b):
    """Standardized mean difference with pooled SD."""
    sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    diff = float(np.mean(a) - np.mean(b))
    if sd == 0:
        return 0.0 if diff == 0 else float("inf")
    return diff / float(sd)

def compare_features(fr, fn, n_bins=N_BINS, min_n=30):
    """SMD and KS for every feature column, and per real-promoter GC quintile bin."""
    out = {}
    for c in fr.columns:
        out[c] = {
            "smd": smd(fr[c].to_numpy(), fn[c].to_numpy()),
            "ks": float(ks_2samp(fr[c], fn[c]).statistic),
        }
    edges = np.quantile(fr["gc"], np.linspace(0, 1, n_bins + 1)[1:-1])
    br = np.digitize(fr["gc"], edges)
    bn = np.digitize(fn["gc"], edges)
    for b in range(n_bins):
        for c in ("gc", "cpg", "repeat_frac"):
            a = fr.loc[br == b, c].to_numpy()
            d = fn.loc[bn == b, c].to_numpy()
            if len(a) < min_n or len(d) < min_n:
                continue
            out[f"{c}@gcbin{b + 1}"] = {"smd": smd(a, d), "ks": float(ks_2samp(a, d).statistic)}
    return out

def kmer_table(seqs, kmers):
    """Per-window k-mer frequencies (case-insensitive) for the given k-mer list."""
    k = len(kmers[0])
    idx = {m: i for i, m in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)))
    for r, s in enumerate(seqs):
        u = s.upper()
        for i in range(len(u) - k + 1):
            j = idx.get(u[i:i + k])
            if j is not None:
                X[r, j] += 1
        X[r] /= max(1, len(u) - k + 1)
    return X

def lr_auroc(X, y, split):
    """Logistic regression fitted on train; AUROC on train, val and test."""
    split = np.asarray(split)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    tr = split == "train"
    model.fit(X[tr], y[tr])
    out = {}
    for name in ("train", "val", "test"):
        m = split == name
        out[name] = float(roc_auc_score(y[m], model.predict_proba(X[m])[:, 1]))
    return out
