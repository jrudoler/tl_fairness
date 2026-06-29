"""Entrypoint: one-step vs TMLE vs CV-TMLE coverage simulation (Setting 1)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tlfair.tmle_sim import coverage_sim_tmle, setting1_prob_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", nargs="+", type=int,
                        default=[100, 250, 500, 1000, 2500])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--truth-n", type=int, default=2_000_000)
    parser.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="reserved for parity with other rules; the heavy "
                             "lifting is vectorised via the backend.")
    parser.add_argument("--output", default="data/generated/tmle_coverage.csv")
    args = parser.parse_args()

    truth = setting1_prob_truth(n=args.truth_n)
    print(f"Setting-1 probabilistic-parity truth = {truth:.5f}", flush=True)

    res = coverage_sim_tmle(
        truth,
        sample_sizes=tuple(args.sample_sizes),
        n_sim=args.reps,
        n_folds=args.n_folds,
        backend=args.backend,
        rng=np.random.default_rng(args.seed),
        bernoulli=True,  # match the manuscript: Y ~ Bernoulli(P(Y=1|X))
    )
    res.to_csv(args.output, index=False)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
