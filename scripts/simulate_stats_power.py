"""Simulation on SYNTHETIC runs: how often does the draft decision rule give each outcome?

No real data is used. Run this before freezing delta, because it shows whether the
'Same' outcome is reachable at 10 runs per recipe.
Usage: python scripts/simulate_stats_power.py [--sims 150] [--boot 2000]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stats import agreement as A  # noqa: E402

OUTCOMES = ("different", "different_within_margin", "same", "inconclusive")
FACTORS = (0.5, 1.0, 1.5)


def probs(p_core, k, recipe):
    """30 families: 8 core, 10 middle (0.5), 12 rare (0.1). Recipe D flips 2k middle families."""
    p = np.r_[np.full(8, p_core), np.full(10, 0.5), np.full(12, 0.1)]
    if recipe == "D" and k > 0:
        p[8:8 + k] = 0.9
        p[8 + k:8 + 2 * k] = 0.1
    return p


def make_runs(rng, n, p):
    return [frozenset(np.flatnonzero(rng.random(len(p)) < p)) for _ in range(n)]


def run_scenario(p_core, k, n_sims, n_boot, seed):
    rng = np.random.default_rng(seed)
    pc, pd_ = probs(p_core, k, "C"), probs(p_core, k, "D")
    recs = []
    for _ in range(n_sims):
        S = A.similarity_matrix(make_runs(rng, 10, pc) + make_runs(rng, 10, pd_))
        core = A.analyze_core(S, 10, 10, n_boot=n_boot, rng=rng)
        within = A.pooled_within_values(S, 10, 10)
        rec = {"G": core["G"], "se": core["boot_sd"],
               "half90": (core["ci_tost"][1] - core["ci_tost"][0]) / 2,
               "sd_within": float(np.std(within, ddof=1))}
        for f in FACTORS:
            delta, _ = A.compute_delta(within, factor=f)
            rec[f"delta_{f}"] = delta
            rec[f"out_{f}"] = A.decide(core["p_perm"], core["ci_diff"], core["ci_tost"], delta)
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    seed = 2026
    for p_core in (0.95, 0.75):
        for k, label in ((0, "no recipe effect"), (1, "small effect"), (3, "large effect")):
            recs = run_scenario(p_core, k, args.sims, args.boot, seed)
            seed += 1
            m = lambda key: float(np.mean([r[key] for r in recs]))  # noqa: E731
            print(f"\n[core detection {p_core}] {label} (k={k}), {args.sims} simulated experiments")
            print(f"  mean G {m('G'):+.3f} | bootstrap SE of G {m('se'):.3f} | "
                  f"90% CI half-width {m('half90'):.3f} | within-recipe SD {m('sd_within'):.3f}")
            for f in FACTORS:
                outs = [r[f"out_{f}"] for r in recs]
                share = "  ".join(f"{o}={outs.count(o) / len(outs):.2f}" for o in OUTCOMES)
                print(f"  delta = {f} x SD (mean delta {m(f'delta_{f}'):.3f}): {share}")
    print(f"\nfinished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()