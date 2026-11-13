"""Agreement statistics for the recipe effect (Step 7).

Input: an n x n symmetric similarity matrix S between explained runs (Jaccard of
motif-family sets, or Spearman correlation of ISM maps). Runs 0..n_c-1 were trained on
Recipe C, runs n_c..n-1 on Recipe D.

G = (mean within-C + mean within-D) / 2 - mean between (C vs D)
Different : exact permutation p < alpha AND lower end of the bootstrap CI of G > 0
Same      : bootstrap CI (TOST level) of G lies inside +/- delta
delta     : factor x SD of the pooled within-recipe pairwise values, clipped to [lo, hi],
            computed from within-recipe pairs only (before any between-recipe value is used)
THRESHOLDS AND FACTORS ARE DRAFT VALUES (docs/decision_rules.md), not frozen until approved.
"""
import itertools
import math
from functools import lru_cache

import numpy as np
from scipy.stats import rankdata




# ---------- noise floor, G, delta ----------
def _check(S, n_c, n_d):
    S = np.asarray(S, dtype=float)
    if n_c < 3 or n_d < 3:
        raise ValueError("need at least 3 runs per recipe")
    if S.shape != (n_c + n_d, n_c + n_d):
        raise ValueError(f"S has shape {S.shape}, expected {(n_c + n_d, n_c + n_d)}")
    if not np.allclose(S, S.T, equal_nan=True):
        raise ValueError("S must be symmetric")
    return S


def _groups(n_c, n_d):
    return np.arange(n_c), np.arange(n_c, n_c + n_d)


def pair_values(S, idx):
    """Off-diagonal values of S for the runs in idx, each unordered pair once."""
    idx = np.asarray(idx)
    sub = np.asarray(S)[np.ix_(idx, idx)]
    return sub[np.triu_indices(len(idx), k=1)]


def pooled_within_values(S, n_c, n_d):
    c_idx, d_idx = _groups(n_c, n_d)
    return np.concatenate([pair_values(S, c_idx), pair_values(S, d_idx)])


def gap_G(S, c_idx, d_idx):
    S = np.asarray(S)
    wc = pair_values(S, c_idx).mean()
    wd = pair_values(S, d_idx).mean()
    between = S[np.ix_(c_idx, d_idx)].mean()
    return float(0.5 * (wc + wd) - between)


# ---------- exact permutation test ----------
def n_partitions(n_c, n_d):
    """Distinct ways to split the runs into groups of the given sizes.
    For equal sizes the two group labels are interchangeable (G is symmetric)."""
    n = n_c + n_d
    return math.comb(n - 1, n_c - 1) if n_c == n_d else math.comb(n, n_c)


@lru_cache(maxsize=8)
def _partition_matrix(n_c, n_d):
    n = n_c + n_d
    if n_c == n_d:
        combos = [(0,) + c for c in itertools.combinations(range(1, n), n_c - 1)]
    else:
        combos = list(itertools.combinations(range(n), n_c))
    arr = np.array(combos, dtype=np.int64)
    X = np.zeros((len(arr), n))
    X[np.arange(len(arr))[:, None], arr] = 1.0
    return X


def _G_for_partitions(S, X, n_c, n_d):
    """G for every row of X (1 = group A of size n_c, 0 = group B of size n_d)."""
    S0 = np.array(S, dtype=float)
    np.fill_diagonal(S0, 0.0)
    Y = 1.0 - X
    XS, YS = X @ S0, Y @ S0
    within_a = (XS * X).sum(axis=1) / (n_c * (n_c - 1))
    within_b = (YS * Y).sum(axis=1) / (n_d * (n_d - 1))
    between = (XS * Y).sum(axis=1) / (n_c * n_d)
    return 0.5 * (within_a + within_b) - between




# ---------- bootstrap over runs ----------
def bootstrap_G(S, n_c, n_d, n_boot=10000, rng=None):
    """Resample runs with replacement inside each recipe. Pairs formed from two draws of the
    same run are excluded from the within-recipe means (they would otherwise count as 1.0)."""
    S = np.asarray(S, dtype=float)
    rng = np.random.default_rng(rng)
    ic = rng.integers(0, n_c, size=(n_boot, n_c))
    jd = n_c + rng.integers(0, n_d, size=(n_boot, n_d))

    def within(idx, n):
        a, b = idx[:, :, None], idx[:, None, :]
        mask = (a != b) & np.triu(np.ones((n, n), dtype=bool), 1)[None, :, :]
        cnt = mask.sum(axis=(1, 2))
        tot = np.where(mask, S[a, b], 0.0).sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, tot / cnt, np.nan)

    between = S[ic[:, :, None], jd[:, None, :]].mean(axis=(1, 2))
    g = 0.5 * (within(ic, n_c) + within(jd, n_d)) - between
    return g[~np.isnan(g)]


# ---------- decision ----------
def decide(p_value, ci_diff, ci_tost, delta, alpha=0.05):
    different = bool(p_value < alpha and ci_diff[0] > 0)
    same = bool(ci_tost[0] > -delta and ci_tost[1] < delta)
    if different and same:
        return "different_within_margin"
    if different:
        return "different"
    if same:
        return "same"
    return "inconclusive"


def analyze_core(S, n_c, n_d, n_boot=10000, rng=None, diff_level=0.95, tost_level=0.90):
    S = _check(S, n_c, n_d)
    rng = np.random.default_rng(rng)
    g_obs, p, info = permutation_pvalue(S, n_c, n_d, rng=rng)
    boot = bootstrap_G(S, n_c, n_d, n_boot=n_boot, rng=rng)
    return {
        "G": g_obs, "p_perm": p, "perm_info": info,
        "ci_diff": percentile_ci(boot, diff_level), "ci_tost": percentile_ci(boot, tost_level),
        "boot_sd": float(np.std(boot, ddof=1)), "n_boot_used": int(len(boot)),
    }


def analyze(S, n_c, n_d, delta=None, delta_factor=0.5, delta_lo=0.03, delta_hi=0.10,
            alpha=0.05, diff_level=0.95, tost_level=0.90, n_boot=10000, rng=None):
    S = _check(S, n_c, n_d)
    floor = noise_floor(S, n_c, n_d)  # within-recipe values only
    if delta is None:
        delta, _ = compute_delta(pooled_within_values(S, n_c, n_d), delta_factor, delta_lo, delta_hi)
    core = analyze_core(S, n_c, n_d, n_boot=n_boot, rng=rng,
                        diff_level=diff_level, tost_level=tost_level)
    outcome = decide(core["p_perm"], core["ci_diff"], core["ci_tost"], delta, alpha)
    return {**core, "delta": float(delta), "noise_floor": floor, "outcome": outcome}
    return g[~np.isnan(g)]


def percentile_ci(values, level):
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(values, [a, 1.0 - a])
    return float(lo), float(hi)
def permutation_pvalue(S, n_c, n_d, max_exact=2_000_000, n_mc=200_000, rng=None, tol=1e-12):
    """One-sided p-value: share of relabelings with G >= the observed G.
    Exact enumeration when feasible (the observed split is one of the splits counted)."""
    S = np.asarray(S, dtype=float)
    c_idx, d_idx = _groups(n_c, n_d)
    g_obs = gap_G(S, c_idx, d_idx)
    if n_partitions(n_c, n_d) <= max_exact:
        g = _G_for_partitions(S, _partition_matrix(n_c, n_d), n_c, n_d)
        return g_obs, float(np.mean(g >= g_obs - tol)), {"method": "exact", "n_partitions": int(len(g))}
    rng = np.random.default_rng(rng)
    n, count, done = n_c + n_d, 0, 0
    while done < n_mc:
        b = min(20_000, n_mc - done)
        order = np.argsort(rng.random((b, n)), axis=1)[:, :n_c]
        X = np.zeros((b, n))
        X[np.arange(b)[:, None], order] = 1.0
        count += int((_G_for_partitions(S, X, n_c, n_d) >= g_obs - tol).sum())
        done += b
    return g_obs, (1 + count) / (1 + n_mc), {"method": "monte_carlo", "n_mc": int(n_mc)}
    between = S[np.ix_(c_idx, d_idx)].mean()
    return float(0.5 * (wc + wd) - between)


def noise_floor(S, n_c, n_d):
    c_idx, d_idx = _groups(n_c, n_d)
    out = {}
    for name, idx in (("C", c_idx), ("D", d_idx)):
        v = pair_values(S, idx)
        out[name] = {"n_pairs": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    pooled = pooled_within_values(S, n_c, n_d)
    out["pooled_within"] = {"n_pairs": int(len(pooled)), "mean": float(pooled.mean()),
                            "sd": float(pooled.std(ddof=1))}
    return out


def compute_delta(within_values, factor=0.5, lo=0.03, hi=0.10):
    sd = float(np.std(np.asarray(within_values, dtype=float), ddof=1))
    return float(min(hi, max(lo, factor * sd))), sd
# ---------- similarity between runs ----------
def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def similarity_matrix(sets):
    n = len(sets)
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = jaccard(sets[i], sets[j])
    return S


def spearman_matrix(maps):
    """Spearman correlation between runs; maps has one row per run."""
    return np.corrcoef(rankdata(np.asarray(maps, dtype=float), axis=1))