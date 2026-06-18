import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from tlfair.metrics import prob_parity


def robust_sim_all(n, truth, rng):
    n = n // 2
    d = 5
    x = rng.normal(size=(2*n, d))
    xt = x**2

    beta = np.array([4, 2, 1, -3, -4])
    y_probs = 1/(1+np.exp(-xt@beta))
    y = (y_probs >= 0.5).astype(np.int8)

    bg = np.array([1, 1, -1, -2, 1])
    g_probs = 1/(1+np.exp(-xt@bg))
    g = (g_probs >= 0.5).astype(np.int8)

    common = {
        'X_train': x[:n, :],
        'X_test': x[n:, :],
        'y_train': y[:n],
        'y_test': y[n:],
        'group_train': g[:n],
        'group_test': g[n:],
    }
    return (
        prob_parity(
            **common,
            outcome=LogisticRegression(solver='liblinear'),
            propensity=LogisticRegression(solver='liblinear'),
        ),
        prob_parity(
            **common,
            outcome=GradientBoostingClassifier(),
            propensity=LogisticRegression(solver='liblinear'),
        ),
        prob_parity(
            **common,
            outcome=LogisticRegression(solver='liblinear'),
            propensity=GradientBoostingClassifier(),
        ),
        prob_parity(
            **common,
            outcome=GradientBoostingClassifier(),
            propensity=GradientBoostingClassifier(),
        ),
    )


def robust_truth(n, rng):
    d = 5
    x = rng.normal(size=(n, d))
    xt = x**2
    beta = np.array([4, 2, 1, -3, -4])
    y_probs = 1/(1+np.exp(-xt@beta))
    bg = np.array([1, 1, -1, -2, 1])
    g_probs = 1/(1+np.exp(-xt@bg))
    g = (g_probs >= 0.5).astype(np.int8)
    return np.mean(y_probs[g == 1]) - np.mean(y_probs[g == 0])


def robust_exp(sample_sizes, reps, seed=123, truth_n=10000000):
    rng = np.random.default_rng(seed)
    truth = robust_truth(truth_n, rng)
    rows = []
    cases = [
        'misspecified',
        'outcome_correct',
        'propensity_correct',
        'both_correct',
    ]

    for sample_size in sample_sizes:
        started = time.perf_counter()
        data = {case: {'coverage': 0, 'estimates': []} for case in cases}
        for _ in range(reps):
            hold = robust_sim_all(n=sample_size, truth=truth, rng=rng)
            for case, result in zip(cases, hold):
                data[case]['estimates'].append(result[0])
                if result[1][0] <= truth <= result[1][1]:
                    data[case]['coverage'] += 1/reps

        for case in cases:
            estimates = data[case]['estimates']
            rows.append({
                'cases': case,
                'sample_size': sample_size,
                'coverage': data[case]['coverage'],
                '95-percentile': np.quantile(estimates, 0.95),
                '5-percentile': np.quantile(estimates, 0.05),
                'mean_estimate': np.mean(estimates),
                'error': np.mean(estimates) - truth,
            })
        print(f"sample_size={sample_size}: {time.perf_counter() - started:.2f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-sizes', nargs='+', type=int, default=[250, 500, 1000, 2500])
    parser.add_argument('--reps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--truth-n', type=int, default=10000000)
    parser.add_argument('--output', default='data/generated/robust_res.csv')
    args = parser.parse_args()

    res = robust_exp(
        sample_sizes=args.sample_sizes,
        reps=args.reps,
        seed=args.seed,
        truth_n=args.truth_n,
    )
    res.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
