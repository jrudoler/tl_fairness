"""Compare completed new computations with frozen manuscript inputs; never select them."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from tlfair.provenance import runtime, sha256, write_json

LOG = logging.getLogger(__name__)
KEYS = {
    'parity_retraining': ['effect', 'train_size', 'test_size', 'method'],
    'cmi_estimation': ['sample_size', 'kappa'],
    'cmi_conditioning_set': ['sample_size', 'n_confounders_included'],
    'nuisance_reference': ['scenario', 'train_size', 'test_size', 'case', 'outcome_model', 'group_model', 'method'],
    'nuisance_early_stopping': ['scenario', 'train_size', 'test_size', 'case', 'outcome_model', 'group_model', 'method'],
}


def compare_csv(old: Path, new: Path, analysis: str) -> pd.DataFrame:
    previous, current = pd.read_csv(old), pd.read_csv(new)
    keys = KEYS[analysis]
    if old.name.endswith('comparison.csv'):
        keys = ['sample size', 'type', 'kappa']
        # The manuscript renderer explicitly excludes this retired estimator.
        # Preserve strict grid checks for TL and KNN, rather than treating its
        # absence from new computations as a failed scientific reproduction.
        previous = previous[previous['type'] != 'TL-sep']
        current = current[current['type'] != 'TL-sep']
        if 'estimator' in current:
            previous['type'] = previous['type'].replace({'TL':'One-step'})
            current['type'] = current['type'].replace({'TL':'One-step'})
            # A historical baseline can only assess the estimators it saved.
            current = current[current['type'].isin(previous['type'].unique())]
    elif old.name == 'population.csv':
        keys = ['scenario']
    elif old.name == 'paired_comparisons.csv':
        keys = [p for p in keys if p != 'method']
    if 'estimator' in current and not old.name.endswith('comparison.csv'):
        if 'estimator' in previous:
            keys = keys+['estimator']
        else:
            current = current[current.estimator == 'one_step']
    merged = previous.merge(current, on=keys, suffixes=('_old', '_new'), how='outer',
                            validate='one_to_one', indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError(f'Comparison grids differ: {old.name}')
    columns = [c for c in previous.select_dtypes(include='number').columns
               if c not in keys and c in current.columns]
    frames = []
    for column in columns:
        frame = merged[keys].copy()
        frame['quantity'] = column
        frame['old'] = merged[column+'_old']
        frame['new'] = merged[column+'_new']
        frame['difference'] = frame['new'] - frame['old']
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def null_diagnostics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep normal-Wald tail failures separate from positive-direction testing."""
    rows = []
    for group, part in frame.groupby(keys):
        group = group if isinstance(group, tuple) else (group,)
        se = (part.ci_high - part.ci_low) / (2*1.96)
        covered = (part.ci_low <= 0) & (part.ci_high >= 0)
        positive = part.ci_low > 0
        negative = part.ci_high < 0
        if not np.isfinite(part[['estimate', 'ci_low', 'ci_high']]).all().all() or not (se >= 0).all():
            raise ValueError('Invalid CMI replicate intervals')
        np.testing.assert_allclose(covered.mean(), 1-positive.mean()-negative.mean(), atol=1e-12)
        rows.append({**dict(zip(keys, group)), 'replicates': len(part),
                     'bias_at_null': part.estimate.mean(), 'empirical_sd': part.estimate.std(ddof=1),
                     'rms_reported_se': np.sqrt(np.mean(se**2)),
                     'two_sided_coverage': covered.mean(),
                     'two_sided_positive_exclusion': positive.mean(),
                     'two_sided_negative_exclusion': negative.mean(),
                     'one_sided_positive_rejection': (part.estimate-1.645*se > 0).mean()})
    return pd.DataFrame(rows)


def compare(run_id: str) -> dict[str, Any]:
    registry = json.loads((ROOT/'config/publication-runs.json').read_text())['analyses']
    reports = {}
    destination = ROOT/'results/data'/f'reproduction-{run_id}'
    destination.mkdir(parents=True, exist_ok=True)
    for analysis, selection in registry.items():
        directory = ROOT/'data/runs'/run_id/analysis
        try:
            manifest = directory/'run_manifest.json'
            record = json.loads(manifest.read_text())
            complete = json.loads((directory/'complete.json').read_text())
            if record['status'] != 'complete' or complete['manifest_sha256'] != sha256(manifest):
                raise ValueError('Run completion is not verified')
            # Refuse to compare outputs changed since their completion record.
            for path, expected in record['outputs'].items():
                if sha256(directory/path) != expected:
                    raise ValueError(f'Run output changed: {path}')
            files = {}
            for published, original in selection['inputs'].items():
                old, new = ROOT/published, directory/original
                files[original] = {'old_sha256': sha256(old), 'new_sha256': sha256(new)}
                if old.suffix == '.csv' and old.name != 'stopping_summary.csv':
                    if old.name.endswith('comparison.csv'):
                        methods = pd.read_csv(old, usecols=['type'])['type']
                        files[original]['excluded_historical_rows'] = int(methods.eq('TL-sep').sum())
                        files[original]['exclusion_reason'] = 'Retired TL-sep estimator is excluded by the manuscript renderer.'
                    result = compare_csv(old, new, analysis)
                    path = destination/f'{analysis}-{old.stem}.csv'
                    result.to_csv(path, index=False)
                    files[original]['comparison'] = str(path.relative_to(ROOT))
                elif original in ['inference.pkl', 'comparison.pkl']:
                    with old.open('rb') as stream:
                        previous = pickle.load(stream)
                    with new.open('rb') as stream:
                        current = pickle.load(stream)
                    rows = []
                    targets = ['inference'] if original == 'inference.pkl' else ['data_fairness','model_fairness']
                    for target in targets:
                        for metric in current[target]:
                            old_metric = 'cmi' if metric == 'cmi_one_step' and 'cmi' in previous[target] else metric
                            if old_metric not in previous[target]:
                                continue
                            for quantity, a, b in zip(['estimate','ci_low','ci_high'],
                                [previous[target][old_metric][0], *previous[target][old_metric][1]],
                                [current[target][metric][0], *current[target][metric][1]]):
                                rows.append({'target':target,'metric':metric,'quantity':quantity,'old':a,'new':b,'difference':b-a})
                    path = destination/f'{analysis}-inference.csv'
                    pd.DataFrame(rows).to_csv(path,index=False)
                    files[original]['comparison'] = str(path.relative_to(ROOT))
                elif original == 'population_truth.pkl':
                    with old.open('rb') as stream:
                        previous = pickle.load(stream)
                    with new.open('rb') as stream:
                        current = pickle.load(stream)
                    path = destination/'cmi_estimation-population-truth.csv'
                    pd.DataFrame([{'kappa':key,'old':previous[key],'new':value,'difference':value-previous[key]}
                                  for key,value in current.items()]).to_csv(path,index=False)
                    files[original]['comparison'] = str(path.relative_to(ROOT))
            if analysis == 'cmi_conditioning_set':
                records = pd.read_csv(directory/'replicates.csv.gz')
                n_conf = record['settings']['parameters']['n-confounders']
                diagnostic_keys = ['sample_size']+(['estimator'] if 'estimator' in records else [])
                null_diagnostics(records[records.n_confounders_included == n_conf], diagnostic_keys).to_csv(
                    destination/'cmi_conditioning_set-null-diagnostics.csv',index=False)
            if analysis == 'cmi_estimation':
                records = pd.read_csv(directory/'coverage_replicates.csv.gz')
                diagnostic_keys = ['sample_size']+(['estimator'] if 'estimator' in records else [])
                null_diagnostics(records[records.kappa == 0], diagnostic_keys).to_csv(
                    destination/'cmi_estimation-null-diagnostics.csv',index=False)
            reports[analysis] = {'status':'compared','run':str(directory.relative_to(ROOT)),
                                 'manifest_sha256':sha256(manifest),'files':files}
        except (OSError, ValueError, KeyError) as error:
            reports[analysis] = {'status':'unavailable','reason':str(error)}
        LOG.info('%s: %s', analysis, reports[analysis]['status'])
    return {'run_id':run_id,'analyses':reports,'publication_changed':False,
            'comparison_code_sha256':sha256(Path(__file__)), 'runtime':runtime()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = compare(args.run_id)
    write_json(ROOT/'results/provenance'/f'reproduction-{args.run_id}.json',result)
    if any(r['status'] != 'compared' for r in result['analyses'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
