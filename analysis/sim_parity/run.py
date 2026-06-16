"""Setting-1 parity simulations (paper Section 4.1, Figures 1 and 3).

Two outputs, each matching how the paper builds its figure:

  parity_fig1.csv  -- threshold parity over a dense log-spaced sample-size grid,
                      ONE simulated dataset per size (estimate + Wald SE). The
                      many single-draw points form the funnel in Figure 1.

  parity_fig3.csv  -- probabilistic parity over a small linear grid; per size we
                      average the TL variance and the naive difference-in-means
                      variance over several replicates (Figure 3).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from tlfair.simulations import parity_ground_truth, parity_sim, coverage_sim_parity


def run_fig1(truth, rng, lo=1.5, hi=4.0, step=0.01):
    """Dense log grid, one draw per size: estimate + SE (threshold parity)."""
    sizes = np.unique(np.round(10 ** np.arange(lo, hi, step)).astype(int))
    rows = []
    started = time.perf_counter()
    for size in sizes:
        _, est, std, _, _ = coverage_sim_parity(
            ground_truth=truth, parity='threshold', n_sim=1, n_samples=int(size), rng=rng,
        )
        rows.append({'sample_size': int(size), 'estimate': float(est[0]),
                     'std': float(std[0]), 'truth': truth})
    print(f"fig1: {len(sizes)} sizes in {time.perf_counter() - started:.2f}s", flush=True)
    return pd.DataFrame(rows)


def run_fig3(rng, sizes, reps):
    """Small linear grid: mean TL var vs mean naive var (probabilistic parity)."""
    rows = []
    started = time.perf_counter()
    for size in sizes:
        tl_vars, naive_vars = [], []
        for _ in range(reps):
            _, var, _, naive_var = parity_sim(n=int(size), parity='prob', rng=rng)
            tl_vars.append(var)
            naive_vars.append(naive_var)
        rows.append({'sample_size': int(size),
                     'tl_var': float(np.mean(tl_vars)),
                     'naive_var': float(np.mean(naive_vars))})
    print(f"fig3: {len(sizes)} sizes x {reps} reps in {time.perf_counter() - started:.2f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth-n', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--fig3-sizes', nargs='+', type=int, default=[50, 100, 250, 500, 1000])
    parser.add_argument('--fig3-reps', type=int, default=50)
    parser.add_argument('--fig1-output', default='data/generated/parity_fig1.csv')
    parser.add_argument('--fig3-output', default='data/generated/parity_fig3.csv')
    args = parser.parse_args()

    # Single Generator threaded through all phases: reproducible from --seed and
    # non-overlapping draws across ground truth, fig1, and fig3.
    rng = np.random.default_rng(args.seed)
    threshold_truth, _ = parity_ground_truth(n=args.truth_n, rng=rng)
    print(f"threshold ground truth: {threshold_truth:.5f}", flush=True)

    run_fig1(threshold_truth, rng).to_csv(args.fig1_output, index=False)
    print(f'Wrote {args.fig1_output}', flush=True)
    run_fig3(rng, args.fig3_sizes, args.fig3_reps).to_csv(args.fig3_output, index=False)
    print(f'Wrote {args.fig3_output}', flush=True)


if __name__ == '__main__':
    main()
