"""CMI coverage comparison: one-step vs TMLE iteration vs boundary CI.

Tests whether TMLE iteration repairs the poor small-c coverage of the one-step
CMI estimator that the manuscript reports in Figure 4. Reuses the paper DGP and
MC truth from cmi_sim; for each (c, sample size) it estimates the coverage and
signed error of four estimators on shared draws/fits. Output is a long CSV with
columns: c, sample_size, estimator, coverage, error.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from tlfair.cmi_sim import (
    cmi_ground_truth, cmi_tmle_coverage_sim, CMI_TMLE_ESTIMATORS,
)


def run_experiment(weights, sizes, *, n_truth=1_000_000, sims=100, seed=123,
                   n_jobs=1):
    rng = np.random.default_rng(seed)
    rows = []
    truth_dict = {}
    for c in weights:
        # Two benchmarks: the unconditional MI I(X;Y) (what the paper's table and
        # Fig 4 use) and the conditional CMI I(X;Y|Z) (the estimand the estimators
        # actually target). They differ most at small c; see cmi_sim notes.
        truths = {
            "uncond": cmi_ground_truth(c=c, d=3, n=n_truth, rng=rng),
            "cond": cmi_ground_truth(c=c, d=3, n=n_truth, rng=rng,
                                     conditional=True),
        }
        truth_dict[c] = truths
        print(f"truth c={c}: uncond={truths['uncond']:.4f} "
              f"cond={truths['cond']:.4f}", flush=True)
        for s in sizes:
            t = time.perf_counter()
            res = cmi_tmle_coverage_sim(n=s, c=c, ground_truths=truths,
                                        sims=sims, rng=rng, n_jobs=n_jobs)
            for name in CMI_TMLE_ESTIMATORS:
                for tt in ("uncond", "cond"):
                    rows.append({"c": c, "sample_size": s, "estimator": name,
                                 "truth_type": tt,
                                 "coverage": res[name][tt]["coverage"],
                                 "error": res[name][tt]["error"]})
            cov = ", ".join(f"{n_}={res[n_]['cond']['coverage']:.2f}"
                            for n_ in CMI_TMLE_ESTIMATORS)
            print(f"  c={c} n={s} [{time.perf_counter()-t:.0f}s] cond cov: {cov}",
                  flush=True)
    return pd.DataFrame(rows), truth_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=float,
                        default=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    parser.add_argument('--sizes', nargs='+', type=int,
                        default=[500, 750, 1000, 1750, 2500, 3750, 5000, 7500, 10000])
    parser.add_argument('--n-truth', type=int, default=1_000_000)
    parser.add_argument('--sims', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--output', default='data/generated/cmi_tmle_coverage.csv')
    parser.add_argument('--truth-output', default='data/generated/cmi_tmle_truth.pkl')
    args = parser.parse_args()

    df, truth = run_experiment(
        weights=args.weights, sizes=args.sizes, n_truth=args.n_truth,
        sims=args.sims, seed=args.seed, n_jobs=args.n_jobs,
    )
    df.to_csv(args.output, index=False)
    with open(args.truth_output, 'wb') as f:
        pickle.dump(truth, f)
    print(f"Wrote {args.output}", flush=True)


if __name__ == '__main__':
    main()
