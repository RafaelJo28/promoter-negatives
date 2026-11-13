import math

import numpy as np
import pytest

from stats import agreement as A


def block_matrix(n_c, n_d, within, between):
    n = n_c + n_d
    S = np.full((n, n), between, dtype=float)
    S[:n_c, :n_c] = within
    S[n_c:, n_c:] = within
    np.fill_diagonal(S, 1.0)
    return S


def make_runs(rng, n, p):
    return [frozenset(np.flatnonzero(rng.random(len(p)) < p)) for _ in range(n)]


def random_S(rng, n):
    M = rng.random((n, n))
    S = (M + M.T) / 2
    np.fill_diagonal(S, 1.0)
    return S


def test_jaccard():
    a = {"TATA", "SP/KLF", "ETS", "CCAAT"}
    b = {"TATA", "SP/KLF", "NRF", "CCAAT"}
    assert A.jaccard(a, b) == pytest.approx(0.6)
    assert A.jaccard(a, a) == 1.0
    assert A.jaccard(set(), set()) == 1.0
    assert A.jaccard(a, set()) == 0.0


def test_similarity_matrix():
    S = A.similarity_matrix([{1, 2}, {2, 3}, {1, 2, 3}])
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 1.0)
    assert S[0, 1] == pytest.approx(1 / 3)


def test_gap_known_value():
    S = block_matrix(4, 4, 0.8, 0.5)
    assert A.gap_G(S, np.arange(4), np.arange(4, 8)) == pytest.approx(0.3)


def test_noise_floor_block():
    f = A.noise_floor(block_matrix(5, 5, 0.8, 0.5), 5, 5)
    assert f["C"]["n_pairs"] == 10 and f["D"]["n_pairs"] == 10
    assert f["C"]["mean"] == pytest.approx(0.8)
    assert f["pooled_within"]["sd"] == pytest.approx(0.0)


def test_n_partitions():
    assert A.n_partitions(10, 10) == 92378
    assert A.n_partitions(8, 10) == math.comb(18, 8)


def test_partition_matrix():
    X = A._partition_matrix(4, 4)
    assert X.shape == (35, 8)
    assert (X.sum(axis=1) == 4).all()
    assert (X[:, 0] == 1).all()
    assert len({tuple(r) for r in X}) == 35


def test_vectorised_G_matches_bruteforce():
    rng = np.random.default_rng(0)
    for n_c, n_d in ((4, 4), (3, 5)):
        S = random_S(rng, n_c + n_d)
        X = A._partition_matrix(n_c, n_d)
        g = A._G_for_partitions(S, X, n_c, n_d)
        for row, gv in zip(X, g):
            a, b = np.flatnonzero(row == 1), np.flatnonzero(row == 0)
            assert gv == pytest.approx(A.gap_G(S, a, b))


def test_perm_perfect_separation():
    S = block_matrix(10, 10, 1.0, 0.0)
    g_obs, p, info = A.permutation_pvalue(S, 10, 10)
    assert g_obs == pytest.approx(1.0)
    assert info["n_partitions"] == 92378
    assert p == pytest.approx(1 / 92378)


def test_perm_constant_matrix():
    S = block_matrix(10, 10, 0.5, 0.5)
    assert A.permutation_pvalue(S, 10, 10)[1] == 1.0


def test_perm_monte_carlo_branch():
    flat = block_matrix(10, 10, 0.5, 0.5)
    assert A.permutation_pvalue(flat, 10, 10, max_exact=1000, n_mc=20000, rng=0)[1] == 1.0
    sep = block_matrix(10, 10, 1.0, 0.0)
    assert A.permutation_pvalue(sep, 10, 10, max_exact=1000, n_mc=20000, rng=0)[1] < 0.001


def test_perm_type1_rate():
    rng = np.random.default_rng(123)
    p = np.r_[np.full(6, 0.9), np.full(8, 0.5), np.full(10, 0.15)]
    hits = 0
    for _ in range(300):
        S = A.similarity_matrix(make_runs(rng, 16, p))
        hits += A.permutation_pvalue(S, 8, 8)[1] < 0.05
    assert hits / 300 <= 0.10


def test_analyze_detects_big_effect():
    rng = np.random.default_rng(5)
    pc = np.r_[np.full(10, 0.9), np.full(10, 0.1), np.full(10, 0.5)]
    pd_ = np.r_[np.full(10, 0.1), np.full(10, 0.9), np.full(10, 0.5)]
    S = A.similarity_matrix(make_runs(rng, 10, pc) + make_runs(rng, 10, pd_))
    r = A.analyze(S, 10, 10, n_boot=2000, rng=1)
    assert r["G"] > 0.2
    assert r["p_perm"] < 0.001
    assert r["ci_diff"][0] > 0
    assert r["outcome"] == "different"


def test_bootstrap_excludes_self_pairs():
    S = block_matrix(10, 10, 0.5, 0.5)
    g = A.bootstrap_G(S, 10, 10, n_boot=2000, rng=0)
    assert len(g) > 1990
    assert np.allclose(g, 0.0, atol=1e-12)


def test_percentile_ci():
    lo, hi = A.percentile_ci(np.linspace(0, 1, 1001), 0.90)
    assert lo == pytest.approx(0.05) and hi == pytest.approx(0.95)


def test_delta_clipping():
    d, sd = A.compute_delta(np.array([0.5, 0.5, 0.5]))
    assert d == 0.03 and sd == 0.0
    assert A.compute_delta(np.array([0.0, 1.0]))[0] == 0.10
    d, sd = A.compute_delta(np.array([0.4, 0.6]))
    assert d == pytest.approx(0.5 * sd) and 0.03 < d < 0.10


def test_decide_rules():
    assert A.decide(0.001, (0.1, 0.3), (0.12, 0.28), 0.05) == "different"
    assert A.decide(0.5, (-0.05, 0.06), (-0.02, 0.03), 0.05) == "same"
    assert A.decide(0.001, (0.01, 0.05), (0.02, 0.04), 0.05) == "different_within_margin"
    assert A.decide(0.2, (-0.1, 0.2), (-0.08, 0.15), 0.05) == "inconclusive"
    assert A.decide(0.001, (-0.01, 0.3), (0.0, 0.28), 0.05) == "inconclusive"


def test_spearman_matrix():
    R = A.spearman_matrix([[1, 2, 3, 4], [10, 20, 30, 40], [4, 3, 2, 1]])
    assert R[0, 1] == pytest.approx(1.0)
    assert R[0, 2] == pytest.approx(-1.0)


def test_analyze_rejects_bad_input():
    with pytest.raises(ValueError):
        A.analyze(np.eye(5), 3, 3)
    with pytest.raises(ValueError):
        A.analyze(block_matrix(2, 2, 0.8, 0.5), 2, 2)
    bad = block_matrix(3, 3, 0.8, 0.5)
    bad[0, 1] = 0.1
    with pytest.raises(ValueError):
        A.analyze(bad, 3, 3)