import argparse
import sys
import time
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from tlfair.metrics import *
from tlfair.superlearner import *


def load_data(input_path, seed, max_rows=None):
    adult = pd.read_csv(input_path)
    if max_rows is not None and max_rows < len(adult):
        adult = adult.sample(n=max_rows, random_state=seed)
    data = adult.copy()
    target = (data["income"] > 50000).astype(int)
    data = data.drop(columns=['income'])
    xtr, xte, ytr, yte = train_test_split(data, target, test_size=0.40, random_state=seed)
    gtr = (xtr['gender'] == 'Male').astype(np.int8)
    gte = (xte['gender'] == 'Male').astype(np.int8)
    continuous = ["hours-per-week", "age", "capital-gain", "capital-loss"]
    xtr = xtr.drop(columns=['education-num', 'gender'])
    xte = xte.drop(columns=['education-num', 'gender'])
    for col in xtr.columns:
        if col in continuous:
            continue
        enc = LabelEncoder().fit(pd.concat([xtr[col], xte[col]]))
        xtr[col] = enc.transform(xtr[col])
        xte[col] = enc.transform(xte[col])
    return xtr, xte, ytr, yte, gtr, gte


def run_experiment(input_path, importance_samples=1000, seed=123, cache_importance=False, metrics_to_run=None, max_rows=None):
    xtr, xte, ytr, yte, gtr, gte = load_data(input_path, seed, max_rows=max_rows)
    metric_map = {
        'parity': parity,
        'prob_parity': prob_parity,
        'opportunity': opportunity,
        'prob_opp': prob_opportunity,
        'cmi': cmi,
    }
    if metrics_to_run is None:
        metrics_to_run = list(metric_map)

    results = {'inference': {}, 'importance': {}, 'timing': {}}
    for title in metrics_to_run:
        started = time.perf_counter()
        metric = metric_map[title]
        if title == 'cmi':
            outcome = HistGradientBoostingClassifier(random_state=seed)
        else:
            outcome = SuperLearnerClassifier(random_state=seed)

        inference = metric(
            xtr=xtr,
            xte=xte,
            ytr=ytr,
            yte=yte,
            gtr=gtr,
            gte=gte,
            outcome=outcome,
            propensity=SuperLearnerClassifier(random_state=seed),
        )
        importance = perm_importance(
            xtr=xtr,
            xte=xte,
            ytr=ytr,
            yte=yte,
            gtr=gtr,
            gte=gte,
            outcome=HistGradientBoostingClassifier(random_state=seed),
            propensity=HistGradientBoostingClassifier(random_state=seed),
            metric=metric,
            n_samples=importance_samples,
            rng=np.random.default_rng(seed),
            cache=cache_importance,
        )
        results['inference'][title] = inference
        results['importance'][title] = importance
        results['timing'][title] = time.perf_counter() - started
        print(f"{title}: {results['timing'][title]:.2f}s", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw/adult.csv')
    parser.add_argument('--output', default='data/generated/adult_results.pkl')
    parser.add_argument('--importance-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cache-importance', action='store_true')
    parser.add_argument('--metrics', nargs='+', default=None)
    parser.add_argument('--max-rows', type=int, default=None)
    args = parser.parse_args()
    print('Beginning Adult Experiment', flush=True)
    results = run_experiment(
        input_path=args.input,
        importance_samples=args.importance_samples,
        seed=args.seed,
        cache_importance=args.cache_importance,
        metrics_to_run=args.metrics,
        max_rows=args.max_rows,
    )
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
