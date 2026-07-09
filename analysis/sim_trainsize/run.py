"""Entrypoint: training-size vs. naive-baseline coverage simulation (Setting 3).

Thin pipeline wrapper around ``experiments.exp5_trainsize_coverage``: it owns the
compute (nuisance training size swept against a fixed evaluation set; TL one-step
vs. three naive fixed-model CLT intervals) and writes the aggregated CSV. Plotting
lives in the companion ``analysis/fig_trainsize`` rule. See the experiment module
docstring for the full rationale.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp5_trainsize_coverage import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sizes", nargs="+", type=int,
                        default=[100, 250, 500, 1000, 2500, 5000, 10000])
    parser.add_argument("--test-size", type=int, default=2000,
                        help="evaluation-set size, held FIXED while train size sweeps")
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--truth-n", type=int, default=10_000_000)
    parser.add_argument("--deterministic", action="store_true",
                        help="use the deterministic Bayes-decision outcome instead of "
                             "the canonical Bernoulli draw (matches the manuscript).")
    parser.add_argument("--output", default="data/generated/trainsize_coverage.csv")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = run(args.train_sizes, args.test_size, args.reps, args.seed, args.n_jobs,
             args.truth_n, bernoulli=not args.deterministic)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
