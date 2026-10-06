"""Fixed-model parity and opportunity intervals on the manuscript data splits.

Fit the manuscript's outcome SuperLearner once per dataset. Treat its predictions
as fixed, compare group means on the held-out observations, and use a normal
interval with variance s_1^2/n_1 + s_0^2/n_0. Probabilistic metrics use predicted
probabilities without the data-fairness residual corrections. Thresholded metrics
use the model's binary decisions. Cache predictions with split indices and input
hashes so subsequent calculations need not refit the ensemble.
"""

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed, parallel_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tlfair.superlearner import SuperLearnerClassifier


def fixed_model_interval(predictions, group, eligible=None):
    """Return the group-1 minus group-0 estimate, SE, 95% CI, and counts."""
    predictions, group = np.asarray(predictions), np.asarray(group)
    if predictions.ndim != 1 or predictions.shape != group.shape:
        raise ValueError('Predictions and group must be equal-length vectors')
    if eligible is None:
        eligible = np.ones(len(group), dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != group.shape:
        raise ValueError('Eligibility must have one entry per evaluation row')
    if not np.isfinite(predictions).all() or not np.isin(group, [0, 1]).all():
        raise ValueError('Predictions must be finite and group must be binary')
    samples = [predictions[eligible & (group == g)] for g in (0, 1)]
    if any(len(values) < 2 for values in samples):
        raise ValueError('Each evaluation stratum needs at least two observations')
    estimate = float(samples[1].mean() - samples[0].mean())
    se = float(np.sqrt(sum(values.var(ddof=1) / len(values) for values in samples)))
    return dict(estimate=estimate, se=se, ci_low=estimate - 1.96 * se,
                ci_high=estimate + 1.96 * se, n_group0=len(samples[0]),
                n_group1=len(samples[1]))


def analyze(dataset, seed, output):
    started = time.perf_counter()
    raw = ROOT / f'data/raw/{dataset}.csv'
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    loader_module = importlib.import_module(f'analysis.analyze_{dataset}.run')
    xtr, xte, ytr, yte, gtr, gte = loader_module.load_data(raw, seed)
    cache = output / f'{dataset}_outcome_predictions.npz'
    metadata_path = output / f'{dataset}_provenance.json'
    specification = dict(dataset=dataset, seed=seed, raw_sha256=raw_hash,
                         sklearn_version=sklearn.__version__,
                         numpy_version=np.__version__, pandas_version=pd.__version__,
                         learner_sha256=hashlib.sha256((ROOT / 'tlfair/superlearner.py').read_bytes()).hexdigest(),
                         preprocessing_sha256=hashlib.sha256(Path(loader_module.__file__).read_bytes()).hexdigest())
    if cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != value for key, value in specification.items()):
            raise ValueError(f'{dataset}: cached predictions have different provenance')
        with np.load(cache, allow_pickle=False) as fitted:
            for name, values in [('train_index', xtr.index), ('test_index', xte.index),
                                 ('y_test', yte), ('group_test', gte)]:
                np.testing.assert_array_equal(fitted[name], np.asarray(values))
            probabilities, decisions = fitted['probabilities'], fitted['decisions']
        print(f'{dataset}: reusing cached predictions', flush=True)
    else:
        print(f'{dataset}: fitting outcome SuperLearner on {len(xtr)} training rows', flush=True)
        outcome = SuperLearnerClassifier(random_state=seed)
        for base in outcome.models:
            if hasattr(base, 'n_jobs'):
                base.set_params(n_jobs=2)
        outcome.fit(xtr, ytr)
        np.testing.assert_array_equal(outcome.classes_, [0, 1])
        probabilities = outcome.predict_proba(xte)[:, 1]
        decisions = outcome.predict(xte)
        np.savez_compressed(cache, probabilities=probabilities, decisions=decisions,
                            train_index=xtr.index.to_numpy(), test_index=xte.index.to_numpy(),
                            y_test=np.asarray(yte), group_test=np.asarray(gte))
        metadata = dict(specification, train_rows=len(xtr), evaluation_rows=len(xte),
                        fit_seconds=time.perf_counter() - started,
                        learner='SuperLearnerClassifier, default six learners and ten folds')
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    rows = []
    for metric, prediction, eligible in [
        ('parity', decisions, None), ('prob_parity', probabilities, None),
        ('opportunity', decisions, np.asarray(yte) == 1),
        ('prob_opp', probabilities, np.asarray(yte) == 1),
    ]:
        row = dict(dataset=dataset, metric=metric,
                   **fixed_model_interval(prediction, gte, eligible))
        rows.append(row)
        print(f"{dataset} {metric}: {row['estimate']:.6f} "
              f"({row['ci_low']:.6f}, {row['ci_high']:.6f})", flush=True)
    pd.DataFrame(rows).to_csv(output / f'{dataset}_naive_intervals.csv', index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['adult', 'law'], default=['adult', 'law'])
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--n-jobs', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, default=Path('data/generated/naive_real_data'))
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with parallel_config(backend='loky', inner_max_num_threads=1):
        batches = Parallel(n_jobs=args.n_jobs)(delayed(analyze)(dataset, args.seed, output)
                                              for dataset in args.datasets)
    table = pd.DataFrame([row for rows in batches for row in rows])
    result_path = ROOT / 'results/data/naive_real_data.csv'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(result_path, index=False)
    print(f'Wrote {result_path}', flush=True)


if __name__ == '__main__':
    main()
