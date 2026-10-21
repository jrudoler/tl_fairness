"""Apply the exact Figure 2 boosting protocol to matched CMI review datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier

from analysis.cmi_learner_review.run import PLAN, SOURCE_FILES, accuracy, true_joint
from experiments.nuisance_early_stopping import BOOST, HIGH
from tlfair.cmi_estimation import paired_cmi, replicate_records, require_convergence
from tlfair.metrics import _encode_joint
from tlfair.provenance import git_state, runtime, sha256, write_json

LOG = logging.getLogger(__name__)
MODELS = {'figure2_lower': BOOST, 'figure2_higher': HIGH}
SOURCES = sorted(set(SOURCE_FILES + [
    'analysis/cmi_learner_review/figure2.py', 'experiments/nuisance_early_stopping.py',
    'experiments/parity_retraining.py', 'workflow/cmi_figure2_review.smk',
    'slurm/cmi_figure2_review.sbatch']))


def one_replicate(output: Path, kappa: float, replicate: int,
                  reference: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    n = PLAN['sample_size']
    bits = np.asarray([kappa], dtype='<f8').view('<u4').ravel().tolist()
    rng = np.random.default_rng(np.random.SeedSequence([123, 1, n, *bits])).spawn(100)[replicate]
    x = rng.normal(size=(n, 3))
    shared = kappa*rng.uniform(size=n)
    p = 1/(1+np.exp(-x@np.ones(3)))
    y = ((shared+rng.uniform(size=n)+p)/(kappa+2) > .5).astype(np.int8)
    g = ((shared+rng.uniform(size=n)+p)/(kappa+2) > .5).astype(np.int8)
    seed = int(rng.integers(0, 2**31-1))
    data_hash = hashlib.sha256(x.astype('<f8').tobytes()+g.tobytes()+y.tobytes()).hexdigest()
    if data_hash != reference['data_hash']:
        raise ValueError('Dataset differs from original matched comparison')
    half = n//2
    classes = _encode_joint(g, y)
    truth = true_joint(x[half:], kappa)
    rows, diagnostics = [], []
    for learner, params in MODELS.items():
        # Exact Figure 2 settings: built-in stopping, no calibration or refit.
        model = GradientBoostingClassifier(**params, random_state=seed).fit(x[:half], classes[:half])
        np.testing.assert_array_equal(model.classes_, np.arange(4))
        result = paired_cmi(model.predict_proba(x[half:]), classes[half:])
        records = replicate_records(result, kappa=kappa, replicate=replicate, learner=learner,
                                    sample_size=n, data_hash=data_hash, boosting_rounds=model.n_estimators_,
                                    trees=model.estimators_.size)
        for row in records:
            row.update(truth=reference['truth'], error=row['estimate']-reference['truth'],
                       covered=row['ci_low'] <= reference['truth'] <= row['ci_high'])
        rows.extend(records)
        for stage, law in [('initial', result['diagnostics']['initial_q']),
                           ('targeted', result['diagnostics']['q_star'])]:
            diagnostics.append(dict(kappa=kappa, replicate=replicate, learner=learner, stage=stage,
                                    **accuracy(law.copy(), truth)))
    raw = pd.DataFrame(rows)
    require_convergence(raw)
    if not np.isfinite(raw[['estimate', 'ci_low', 'ci_high', 'se']]).all().all():
        raise ValueError('Nonfinite estimate or interval')
    cell = output/f'kappa_{kappa:g}'
    cell.mkdir(parents=True, exist_ok=True)
    raw.to_csv(cell/f'replicate_{replicate:03d}.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(cell/f'accuracy_{replicate:03d}.csv', index=False)
    paths = [cell/f'replicate_{replicate:03d}.csv', cell/f'accuracy_{replicate:03d}.csv']
    write_json(cell/f'complete_{replicate:03d}.json', {p.name: sha256(p) for p in paths})
    return dict(kappa=kappa, replicate=replicate, elapsed_seconds=time.monotonic()-started)


def collect(output: Path, report: Path, reference: pd.DataFrame) -> None:
    records, diagnostics = [], []
    for kappa in PLAN['kappas']:
        for replicate in range(100):
            cell = output/f'kappa_{kappa:g}'
            receipt = json.loads((cell/f'complete_{replicate:03d}.json').read_text())
            for name, expected in receipt.items():
                if sha256(cell/name) != expected:
                    raise ValueError(f'Replicate checksum mismatch: {cell/name}')
            records.append(pd.read_csv(cell/f'replicate_{replicate:03d}.csv'))
            diagnostics.append(pd.read_csv(cell/f'accuracy_{replicate:03d}.csv'))
    raw = pd.concat(records, ignore_index=True)
    assert len(raw) == 1800 and not raw.duplicated(['kappa', 'replicate', 'learner', 'estimator']).any()
    require_convergence(raw)
    combined = pd.concat([reference, raw], ignore_index=True)
    assert combined.groupby(['kappa', 'replicate']).data_hash.nunique().eq(1).all()
    rows, pairs = [], []
    for (kappa, learner), part in combined[combined.estimator == 'tmle'].groupby(['kappa', 'learner']):
        coverage = part.covered.mean()
        rows.append(dict(kappa=kappa, learner=learner, replicates=len(part), coverage=coverage,
                         coverage_mcse=np.sqrt(coverage*(1-coverage)/len(part)), bias=part.error.mean(),
                         bias_mcse=part.error.sem(), mean_se=part.se.mean(), empirical_sd=part.estimate.std(),
                         mean_rounds=part.boosting_rounds.mean() if learner in MODELS else np.nan,
                         median_rounds=part.boosting_rounds.median() if learner in MODELS else np.nan))
        if learner in MODELS:
            baseline = reference[(reference.kappa == kappa) & (reference.estimator == 'tmle') &
                                 (reference.learner == 'default_sigmoid_ensemble')].set_index('replicate')
            candidate = part.set_index('replicate').loc[baseline.index]
            error = candidate.estimate-baseline.estimate
            coverage_diff = candidate.covered.astype(float)-baseline.covered.astype(float)
            pairs.append(dict(kappa=kappa, learner=learner, bias_difference=error.mean(),
                              bias_difference_mcse=error.sem(), coverage_difference=coverage_diff.mean(),
                              coverage_difference_mcse=coverage_diff.sem()))
    report.mkdir(parents=True, exist_ok=True)
    raw.to_csv(report/'replicates.csv.gz', index=False)
    pd.DataFrame(rows).to_csv(report/'summary.csv', index=False)
    pd.DataFrame(pairs).to_csv(report/'paired_differences.csv', index=False)
    diagnostics_raw = pd.concat(diagnostics, ignore_index=True)
    diagnostics_raw.to_csv(report/'accuracy_replicates.csv.gz', index=False)
    metrics = [c for c in diagnostics_raw if c not in ['kappa', 'replicate', 'learner', 'stage']]
    accuracy_summary = diagnostics_raw.groupby(['kappa', 'learner', 'stage'])[metrics].agg(['mean', 'sem'])
    accuracy_summary.columns = ['_'.join(c) for c in accuracy_summary.columns]
    accuracy_summary.to_csv(report/'accuracy_summary.csv')
    write_json(report/'manifest.json', dict(run=str(output.relative_to(ROOT)),
               run_manifest_sha256=sha256(output/'manifest.json'),
               matched_datasets=300, converged_tmle_fits=600,
               targeted_accuracy_caveat='Oracle accuracy at evaluation X after using evaluation labels; not independent validation.',
               files={p.name: sha256(p) for p in report.iterdir() if p.name != 'manifest.json'}))
    LOG.info('Figure 2 protocol comparison complete:\n%s',
             pd.DataFrame(rows).query('learner in @MODELS').to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/runs/cmi-figure2-review-20261021'))
    parser.add_argument('--report-dir', type=Path, default=Path('results/review/cmi-figure2-review-20261021'))
    parser.add_argument('--reference-dir', type=Path, default=Path('results/review/cmi-learner-review-20261021'))
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    output, report, reference_dir = [p.resolve() for p in [args.output_dir, args.report_dir, args.reference_dir]]
    source_hashes = {p: sha256(ROOT/p) for p in SOURCES}
    meta = dict(models=MODELS, sample_size=10000, kappas=PLAN['kappas'], replicates=100,
                seed=123, calibration=False, full_training_refit=False, sources=source_hashes,
                reference=str(reference_dir.relative_to(ROOT)),
                reference_sha256=sha256(reference_dir/'replicates.csv.gz'),
                reference_manifest_sha256=sha256(reference_dir/'manifest.json'))
    path = output/'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous[k] != value for k, value in meta.items()):
            raise ValueError('Refusing incompatible resume; use a fresh run directory')
    else:
        for name in SOURCES:
            dest = output/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, dest)
        write_json(path, dict(**meta, git=git_state(), runtime=runtime(), slurm_job=os.getenv('SLURM_JOB_ID')))
    reference = pd.read_csv(reference_dir/'replicates.csv.gz')
    base = reference[(reference.estimator == 'tmle') & (reference.learner == 'default_sigmoid_ensemble')]
    assert len(base) == 300
    tasks = [r for r in base.to_dict('records') if not
             (output/f'kappa_{r["kappa"]:g}'/f'complete_{r["replicate"]:03d}.json').exists()]
    from analysis.cmi_learner_review.figure2 import one_replicate as worker
    for audit in Parallel(n_jobs=args.n_jobs, return_as='generator_unordered', batch_size=1)(
            delayed(worker)(output, r['kappa'], r['replicate'], r) for r in tasks):
        LOG.info('kappa=%g replicate=%d both configurations complete in %.1fs',
                 audit['kappa'], audit['replicate'], audit['elapsed_seconds'])
    collect(output, report, reference)


if __name__ == '__main__':
    main()
