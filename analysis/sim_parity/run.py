"""Setting-1 parity simulation sweep (paper Section 4.1, Figures 1 and 3).

For each sample size we run ``n_sim`` replicates and record:
  - the mean TL estimate and mean standard error (Figure 1: estimate + CI vs n)
  - the mean TL variance and mean naive difference-in-means variance (Figure 3)
relative to the Monte Carlo ground truth.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from tlfair.simulations import (
    parity_ground_truth,
    parity_sim,
    coverage_sim_parity,
)


def run_experiment(sample_sizes, parity, n_sim, truth_n, seed):
    rng = np.random.default_rng(seed)
    threshold_truth, prob_truth = parity_ground_truth(n=truth_n, rng=rng)
    truth = threshold_truth if parity == 'threshold' else prob_truth
    print(f"ground truth ({parity}): {truth:.5f}", flush=True)

    rows = []
    for size in sample_sizes:
        started = time.perf_counter()
        # Figure 1: estimate + CI across replicates.
        _, estimates, std, upper, lower = coverage_sim_parity(
            ground_truth=truth, parity=parity, n_sim=n_sim, n_samples=size, rng=rng,
        )
        # Figure 3: TL vs naive variance across replicates.
        tl_vars, naive_vars = [], []
        for _ in range(n_sim):
            _, var, _, naive_var = parity_sim(n=size // 2, parity=parity, rng=rng)
            tl_vars.append(var)
            naive_vars.append(naive_var)

        rows.append({
            'sample_size': size,
            'parity': parity,
            'truth': truth,
            'estimate': float(np.mean(estimates)),
            'std': float(np.mean(std)),
            'ci_lower': float(np.mean(lower)),
            'ci_upper': float(np.mean(upper)),
            'coverage': float(np.mean((upper >= truth) & (lower <= truth))),
            'tl_var': float(np.mean(tl_vars)),
            'naive_var': float(np.mean(naive_vars)),
        })
        print(f"n={size}: {time.perf_counter() - started:.2f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-sizes', nargs='+', type=int,
                        default=[50, 100, 250, 500, 1000, 2500, 5000, 10000])
    parser.add_argument('--parity', choices=['threshold', 'prob'], default='threshold')
    parser.add_argument('--n-sim', type=int, default=100)
    parser.add_argument('--truth-n', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', default='data/generated/parity_sim.csv')
    args = parser.parse_args()

    res = run_experiment(
        sample_sizes=args.sample_sizes,
        parity=args.parity,
        n_sim=args.n_sim,
        truth_n=args.truth_n,
        seed=args.seed,
    )
    res.to_csv(args.output, index=False)
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
