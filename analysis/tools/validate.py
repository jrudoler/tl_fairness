import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CSVS = {
    'robust_res.csv': ['cases', 'sample_size', 'coverage', '95-percentile', '5-percentile', 'mean_estimate', 'error'],
    'tmle_coverage.csv': ['estimator', 'sample_size', 'coverage', 'bias', 'var', 'mean_estimate', 'mean_ci_width'],
    'cmi_coverage.csv': ['sample_size', 'c', 'error', 'coverage'],
    'cmi_compare.csv': ['sample size', 'type', 'c', 'mean', 'bottom_five', 'top_five'],
}

EXPECTED_PICKLES = [
    'truth_dict.pkl',
    'adult_results.pkl',
    'law_results.pkl',
]

EXPECTED_METRICS = ['parity', 'prob_parity', 'opportunity', 'prob_opp', 'cmi',
                    'prob_parity_tmle', 'prob_opp_tmle', 'cmi_tmle']


def validate_csv(path, columns):
    df = pd.read_csv(path)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    if df.empty:
        raise ValueError(f'{path}: file is empty')
    numeric = df.select_dtypes(include=[np.number])
    if not np.all(np.isfinite(numeric.to_numpy())):
        raise ValueError(f'{path}: contains non-finite numeric values')
    return df.shape


def validate_results_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    for key in ['inference', 'importance']:
        if key not in data:
            raise ValueError(f'{path}: missing key {key}')
    # Feature importance is optional (off by default); only validate it when the
    # run actually produced importance results.
    has_importance = bool(data['importance'])
    for metric in EXPECTED_METRICS:
        if metric not in data['inference']:
            raise ValueError(f'{path}: missing inference metric {metric}')
        if has_importance and metric not in data['importance']:
            raise ValueError(f'{path}: missing importance metric {metric}')
        est, ci = data['inference'][metric]
        if not (np.isfinite(est) and np.all(np.isfinite(ci))):
            raise ValueError(f'{path}: non-finite inference for {metric}')
    return list(data.keys())


def validate_truth_pickle(path):
    with open(path, 'rb') as f:
        truth = pickle.load(f)
    if not truth:
        raise ValueError(f'{path}: empty truth dictionary')
    if not np.all(np.isfinite(list(truth.values()))):
        raise ValueError(f'{path}: non-finite truth values')
    return sorted(truth.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()

    for name, columns in EXPECTED_CSVS.items():
        path = args.directory / name
        if not path.exists():
            raise FileNotFoundError(path)
        print(f'{name}: shape={validate_csv(path, columns)}')

    for name in EXPECTED_PICKLES:
        path = args.directory / name
        if not path.exists():
            raise FileNotFoundError(path)
        if name == 'truth_dict.pkl':
            print(f'{name}: keys={validate_truth_pickle(path)}')
        else:
            print(f'{name}: keys={validate_results_pickle(path)}')


if __name__ == '__main__':
    main()
