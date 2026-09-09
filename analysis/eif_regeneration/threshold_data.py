"""Regenerate threshold intervals, reusing one fitted outcome model per dataset."""
import argparse
import importlib
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tlfair.metrics import _prob_group_contrast, _wald_ci
from tlfair.superlearner import SuperLearnerClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['adult', 'law'], required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/out/eif_regeneration'))
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    loader = importlib.import_module(f'analysis.analyze_{args.dataset}.run').load_data
    xtr, xte, ytr, yte, gtr, gte = loader(ROOT / f'data/raw/{args.dataset}.csv', 123)
    model = SuperLearnerClassifier(random_state=123).fit(xtr, ytr)
    predictions = model.predict(xte)
    y, g = np.asarray(yte), np.asarray(gte)
    np.savez_compressed(output / f'{args.dataset}_threshold_predictions.npz', predictions=predictions, y=y, g=g)
    results, old = {}, {}
    for metric in ['parity', 'opportunity']:
        eligible = (y == 1) if metric == 'opportunity' else np.ones(len(y), dtype=bool)
        strata = np.column_stack([(g == k) & eligible for k in (0, 1)])
        est, ci = _prob_group_contrast(predictions, np.zeros(len(y)), np.zeros_like(strata, dtype=float), strata)
        results[metric] = (est, ci)
        # Exact historical formula on the same predictions, including its
        # shortened row array for equal opportunity.
        phi = np.concatenate([-predictions[strata[:, 0]] / strata[:, 0].mean(),
                              predictions[strata[:, 1]] / strata[:, 1].mean()])
        old[metric] = (est, _wald_ci(est, phi - est))
        logging.info('%s %s: old=%s corrected=%s', args.dataset, metric, old[metric], results[metric])
    with (output / f'{args.dataset}_threshold.pkl').open('wb') as f:
        pickle.dump({'inference': results, 'old_inference_same_fit': old}, f)


if __name__ == '__main__':
    main()
