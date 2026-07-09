"""Entrypoint: conditioning-set vs. data-fairness verdict simulation (CMI).

Thin pipeline wrapper around ``experiments.exp4_feature_selection``: a confounded
DGP in which ``Y ⊥ G | Z`` holds exactly, sweeping how many confounders are in the
conditioning set and applying the one-sided TL CMI Wald test. This rule owns the
compute and writes the aggregated CSV; plotting lives in ``analysis/fig_condset``.
See the experiment module docstring for the full rationale.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp4_feature_selection import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=30, help="total number of features")
    parser.add_argument("--n-confounders", type=int, default=5,
                        help="confounders driving both G and Y (the needed adjustment set)")
    parser.add_argument("--n-outcome", type=int, default=8,
                        help="outcome-only predictors (drive Y, independent of G)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[2500, 5000])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--signal", type=float, default=1.5,
                        help="coefficient scale (larger -> stronger dependence)")
    parser.add_argument("--coef-seed", type=int, default=0,
                        help="seed for the (fixed) DGP coefficients")
    parser.add_argument("--seed", type=int, default=123, help="seed for the data draws")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", default="data/generated/condset.csv")
    args = parser.parse_args()

    if args.n_confounders + args.n_outcome > args.p:
        parser.error("--n-confounders + --n-outcome must not exceed --p")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = run(args.p, args.n_confounders, args.n_outcome, args.sizes, args.reps,
             args.signal, args.coef_seed, args.seed, args.n_jobs)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
