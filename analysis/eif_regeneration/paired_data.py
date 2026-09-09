"""Refit each real-data nuisance once and compare old/new intervals on that fit."""
import argparse
import importlib
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tlfair import metrics, tmle
from tlfair.superlearner import SuperLearnerClassifier


def fit_predictions(xtr: pd.DataFrame, xte: pd.DataFrame, labels: np.ndarray,
                    name: str, output: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = output / f'{name}_predictions.npz'
    model = SuperLearnerClassifier(random_state=123)
    for base in model.models:
        if hasattr(base, 'n_jobs'):
            base.set_params(n_jobs=4)
    model.fit(xtr, labels)
    probabilities = model.predict_proba(xte)
    predictions = model.predict(xte)
    np.savez_compressed(path, probabilities=probabilities, predictions=predictions,
                        classes=model.classes_, labels=labels)
    return probabilities, predictions, model.classes_


class SavedPredictions:
    """Metric-interface adapter for predictions on this exact evaluation frame."""
    def __init__(self, result: tuple[np.ndarray, np.ndarray, np.ndarray],
                 labels: np.ndarray, index: pd.Index) -> None:
        self.probabilities, self.predictions, self.classes_ = result
        self.labels, self.index = labels, index

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> 'SavedPredictions':
        np.testing.assert_array_equal(y, self.labels)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert x.index.equals(self.index)
        return self.probabilities

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        assert x.index.equals(self.index)
        return self.predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['adult', 'law'], required=True)
    parser.add_argument('--n-jobs', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/out/eif_regeneration'))
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    loader = importlib.import_module(f'analysis.analyze_{args.dataset}.run').load_data
    xtr, xte, ytr, yte, gtr, gte = loader(ROOT/f'data/raw/{args.dataset}.csv', 123)
    labels = [np.asarray(ytr), np.asarray(gtr), metrics._encode_joint(gtr, ytr)]
    logging.info('Fitting outcome, group, and joint nuisances once each for %s', args.dataset)
    results = Parallel(n_jobs=args.n_jobs)(delayed(fit_predictions)(xtr, xte, label,
        f'{args.dataset}_{name}', output) for name, label in zip(['outcome', 'group', 'joint'], labels))
    outcome, group, joint = [SavedPredictions(result, label, xte.index) for result, label in zip(results, labels)]
    g, y = np.asarray(gte), np.asarray(yte)
    d = outcome.predict_proba(xte)[:, 1]
    common = dict(X_train=xtr, X_test=xte, y_train=ytr, y_test=yte, group_train=gtr, group_test=gte, outcome=outcome)
    corrected, old, rows = {}, {}, []
    for key, fn, prop, positive, targeted in [
        ('parity', metrics.parity, group, False, False),
        ('opportunity', metrics.opportunity, joint, True, False),
        ('prob_parity', metrics.prob_parity, group, False, False),
        ('prob_opp', metrics.prob_opportunity, joint, True, False),
        ('prob_parity_tmle', tmle.prob_parity_tmle, group, False, True),
        ('prob_opp_tmle', tmle.prob_opportunity_tmle, joint, True, True),
    ]:
        eligible = y == 1 if positive else np.ones(len(y), dtype=bool)
        strata = np.column_stack([(g == k) & eligible for k in (0, 1)])
        proportions = strata.mean(axis=0)
        if targeted:
            est, ci, diag = fn(**common, propensity=prop, return_diagnostics=True)
            score = diag['d_star']
        else:
            est, ci = fn(**common, propensity=prop)
            score = d
        if key in ['parity', 'opportunity']:
            prediction = outcome.predict(xte)
            phi = np.concatenate([-prediction[strata[:, 0]] / proportions[0],
                                   prediction[strata[:, 1]] / proportions[1]])
        else:
            weights = prop.predict_proba(xte)
            if positive:
                weights = weights[:, [2, 3]]
            contribution = (weights * (y - score)[:, None] + strata * score[:, None]) / proportions
            phi = contribution[:, 1] - contribution[:, 0]
        corrected[key] = (est, ci)
        old[key] = (est, metrics._wald_ci(est, phi - est))
        rows.append(dict(metric=key, estimate=est, old_lo=old[key][1][0], old_hi=old[key][1][1], new_lo=ci[0], new_hi=ci[1]))
        logging.info('%s: %s -> %s', key, old[key], corrected[key])
    pd.DataFrame(rows).to_csv(output/f'{args.dataset}_paired_intervals.csv', index=False)
    with (output/f'{args.dataset}_results.pkl').open('wb') as f:
        pickle.dump({'inference': corrected, 'old_inference_same_fit': old, 'timing': {}}, f)


if __name__ == '__main__':
    main()
