import argparse
import sys
import time
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from tlfair.metrics import *
from tlfair.superlearner import *
from tlfair.knncmi import *
from tlfair.cmi_sim import *

def _safe_compare_jobs(n, n_jobs, mem_gb):
    """Cap parallel workers so concurrent knncmi distance arrays fit in memory.

    knncmi builds an O(p * n^2) float64 array (p=5 here -> ~4 GB at n=10000),
    so running too many at once on a large n exhausts RAM. The coverage step has
    no such footprint and always uses the full n_jobs.
    """
    per_worker_gb = 5 * (n ** 2) * 8 / 1e9
    if per_worker_gb <= 0:
        return n_jobs
    return max(1, min(n_jobs, int(mem_gb / per_worker_gb)))


def run_experiment(
    weights,
    sizes,
    n_truth=1000000,
    sims=100,
    compare_repeats=10,
    seed=123,
    conditional_truth=True,
    n_jobs=1,
    compare_mem_gb=16.0,
):
    rng = np.random.default_rng(seed)
    results = []
    truth_dict = {}
    timings = {}

    for c in weights:
        started = time.perf_counter()
        truth = cmi_ground_truth(
            c=c,
            d=3,
            n=n_truth,
            rng=rng,
            conditional=conditional_truth,
        )
        truth_dict[c] = truth
        timings[f'truth_{c}'] = time.perf_counter() - started
        print(f"truth c={c}: {truth:.4f} in {timings[f'truth_{c}']:.2f}s", flush=True)
        for s in sizes:
            started = time.perf_counter()
            coverage, error = cmi_coverage_sim(
                n=s,
                c=c,
                ground_truth=truth,
                rng=rng,
                sims=sims,
                n_jobs=n_jobs,
            )
            timings[f'coverage_c={c}_n={s}'] = time.perf_counter() - started
            results.append({
                "sample_size": s,
                "c": c,
                "error": error,
                "coverage": coverage,
            })
            print(
                f"coverage c={c}, n={s}: {timings[f'coverage_c={c}_n={s}']:.2f}s",
                flush=True,
            )

    compare_results = []
    for s in sizes:
        started = time.perf_counter()
        compare_jobs = _safe_compare_jobs(s, n_jobs, compare_mem_gb)
        compare_results.append(cmi_compare(
            n=s,
            params=weights,
            repeats=compare_repeats,
            rng=rng,
            n_jobs=compare_jobs,
        ))
        timings[f'compare_n={s}'] = time.perf_counter() - started
        print(
            f"compare n={s} ({compare_jobs} workers): "
            f"{timings[f'compare_n={s}']:.2f}s",
            flush=True,
        )

    return (
        pd.DataFrame(results),
        pd.concat(compare_results, ignore_index=True),
        truth_dict,
        timings,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=float, default=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    parser.add_argument('--sizes', nargs='+', type=int, default=[500, 750, 1000, 1750, 2500, 3750, 5000, 7500, 10000])
    parser.add_argument('--n-truth', type=int, default=1000000)
    parser.add_argument('--sims', type=int, default=100)
    parser.add_argument('--compare-repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--unconditional-truth', action='store_true',
                        help='(legacy) score coverage/error against the '
                             'unconditional MI I(X;Y) instead of the conditional '
                             'CMI I(X;Y|Z) that the estimator actually targets')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='parallel workers for coverage sims and the '
                             'comparison (deterministic via per-task RNG spawn)')
    parser.add_argument('--compare-mem-gb', type=float, default=16.0,
                        help='memory budget for concurrent knncmi distance '
                             'arrays; caps compare workers at large n')
    parser.add_argument('--coverage-output', default='data/generated/cmi_coverage.csv')
    parser.add_argument('--compare-output', default='data/generated/cmi_compare.csv')
    parser.add_argument('--truth-output', default='data/generated/truth_dict.pkl')
    parser.add_argument('--timing-output', default='data/generated/cmi_timing.pkl')
    args = parser.parse_args()

    coverage, compare, truth, timings = run_experiment(
        weights=args.weights,
        sizes=args.sizes,
        n_truth=args.n_truth,
        sims=args.sims,
        compare_repeats=args.compare_repeats,
        seed=args.seed,
        conditional_truth=not args.unconditional_truth,
        n_jobs=args.n_jobs,
        compare_mem_gb=args.compare_mem_gb,
    )
    coverage.to_csv(args.coverage_output, index=False)
    compare.to_csv(args.compare_output, index=False)
    with open(args.truth_output, 'wb') as f:
        pickle.dump(truth, f)
    with open(args.timing_output, 'wb') as f:
        pickle.dump(timings, f)


if __name__ == '__main__':
    main()
