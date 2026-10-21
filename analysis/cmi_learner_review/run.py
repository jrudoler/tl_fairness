"""Matched diagnostic comparison of CMI nuisance learners; no publication writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import pickle
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import quad
from scipy.special import expit, xlogy
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from tlfair.cmi_estimation import paired_cmi, replicate_records
from tlfair.metrics import _encode_joint
from tlfair.provenance import git_state, runtime, sha256, write_json
from tlfair.tmle import _cmi_log_ratio

LOG = logging.getLogger(__name__)
METHODS = ['default_sigmoid_ensemble', 'default_raw_ensemble', 'default_raw_full',
           'default_sigmoid_full', 'tuned_raw_full', 'tuned_sigmoid_full',
           'smooth_multinomial', 'oracle']
BOOST_PATHS = [dict(max_depth=d, min_samples_leaf=20, learning_rate=.05)
               for d in (1, 3, 5)] + [dict(max_depth=3, min_samples_leaf=1, learning_rate=.1)]
PLAN = dict(kappas=[0.0, 2.5, 4.0], sample_size=10000, replicates=100,
            validation_fraction=.2, boosting_paths=BOOST_PATHS,
            boosting_min_rounds=25, boosting_max_rounds=1000, boosting_step=25,
            stopping_patience_checks=4, stopping_min_improvement=1e-5,
            polynomial_degrees=[1, 2, 3], logistic_C=[.1, 1., 10.],
            selection='minimum training-holdout multiclass log loss; refit on all training rows',
            calibration='sigmoid, cv=3; no evaluation observations used in fitting or selection',
            methods=METHODS)
SOURCE_FILES = ['analysis/cmi_learner_review/run.py', 'slurm/cmi_learner_review.sbatch',
                'workflow/cmi_learner_review.smk', 'tlfair/cmi_estimation.py',
                'tlfair/tmle.py', 'tlfair/metrics.py', 'tlfair/provenance.py',
                'tlfair/plotting.py', 'clean-figs.mplstyle', 'pyproject.toml', 'uv.lock']


def true_joint(x: np.ndarray, kappa: float) -> np.ndarray:
    """Analytic joint law for the prespecified null and kappa >= 2 settings."""
    p = expit(np.sum(x, axis=1))
    if kappa == 0:
        return np.column_stack(((1-p)**2, p*(1-p), p*(1-p), p**2))
    if kappa < 2:
        raise ValueError('This review supports kappa=0 or kappa>=2')
    p11 = .5+(p-2/3)/kappa
    off = np.full_like(p, 1/(6*kappa))
    return np.column_stack((1-p11-2*off, off, off, p11))


def exact_truth(kappa: float) -> tuple[float, float]:
    if kappa == 0:
        return 0., 0.
    def integrand(z: float) -> float:
        q = true_joint(np.array([[z, 0., 0.]]), kappa)
        return float(np.sum(q*_cmi_log_ratio(q))*norm.pdf(z, scale=np.sqrt(3)))
    value, error = quad(integrand, -16*np.sqrt(3), 16*np.sqrt(3), epsabs=1e-12, epsrel=1e-12)
    return float(value), float(error)


def accuracy(q: np.ndarray, p: np.ndarray) -> dict[str, float]:
    q = np.maximum(q, np.finfo(float).tiny)
    q /= q.sum(axis=1, keepdims=True)
    pg, py = p[:, 1]+p[:, 3], p[:, 2]+p[:, 3]
    qg, qy = q[:, 1]+q[:, 3], q[:, 2]+q[:, 3]
    def bernoulli_kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(xlogy(a, a/b)+xlogy(1-a, (1-a)/(1-b))))
    joint_kl = float(np.mean(np.sum(xlogy(p, p/q), axis=1)))
    group_kl, outcome_kl = bernoulli_kl(pg, qg), bernoulli_kl(py, qy)
    remainder = group_kl+outcome_kl-joint_kl
    direct = float(np.mean(np.sum(p*(_cmi_log_ratio(q)-_cmi_log_ratio(p)), axis=1)))
    np.testing.assert_allclose(remainder, direct, atol=1e-12)
    return dict(joint_rmse=float(np.sqrt(np.mean((p-q)**2))),
                group_rmse=float(np.sqrt(np.mean((pg-qg)**2))),
                outcome_rmse=float(np.sqrt(np.mean((py-qy)**2))),
                joint_kl=joint_kl, logratio_remainder=remainder,
                group_variation_ratio=float(np.std(qg)/np.std(pg)),
                outcome_variation_ratio=float(np.std(qy)/np.std(py)))


def tune_boosting(x: np.ndarray, classes: np.ndarray, train: np.ndarray,
                  valid: np.ndarray, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Early-stop each fixed capacity path using only a training holdout."""
    audit: list[dict[str, Any]] = []
    for params in BOOST_PATHS:
        model = GradientBoostingClassifier(**params, random_state=seed, warm_start=True)
        best_loss, no_improvement = np.inf, 0
        for rounds in range(25, 1001, 25):
            model.set_params(n_estimators=rounds).fit(x[train], classes[train])
            loss = float(log_loss(classes[valid], model.predict_proba(x[valid]), labels=np.arange(4)))
            audit.append(dict(**params, n_estimators=rounds, validation_log_loss=loss))
            if loss < best_loss-1e-5:
                best_loss, no_improvement = loss, 0
            else:
                no_improvement += 1
            if no_improvement >= 4:
                break
    selected = min(audit, key=lambda r: (r['validation_log_loss'],
                   r['n_estimators']*2**r['max_depth'], r['n_estimators']))
    return {k: v for k, v in selected.items() if k != 'validation_log_loss'}, audit


def smooth_model(degree: int, regularization: float) -> Any:
    return make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),
                         StandardScaler(), LogisticRegression(C=regularization, max_iter=2000, tol=1e-6))


def fit_candidates(x: np.ndarray, classes: np.ndarray, evaluate: np.ndarray,
                   seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Evaluation features are used for prediction only; labels are unavailable."""
    predictions: dict[str, np.ndarray] = {}
    default = CalibratedClassifierCV(GradientBoostingClassifier(random_state=seed), cv=3).fit(x, classes)
    predictions['default_sigmoid_ensemble'] = default.predict_proba(evaluate)
    predictions['default_raw_ensemble'] = np.mean([
        fit.estimator.predict_proba(evaluate) for fit in default.calibrated_classifiers_], axis=0)
    full = CalibratedClassifierCV(GradientBoostingClassifier(random_state=seed),
                                  cv=3, ensemble=False).fit(x, classes)
    predictions['default_sigmoid_full'] = full.predict_proba(evaluate)
    predictions['default_raw_full'] = full.calibrated_classifiers_[0].estimator.predict_proba(evaluate)
    train, valid = train_test_split(np.arange(len(x)), test_size=.2, stratify=classes,
                                    random_state=seed)
    selected, boost_audit = tune_boosting(x, classes, train, valid, seed)
    tuned = CalibratedClassifierCV(GradientBoostingClassifier(**selected, random_state=seed),
                                   cv=3, ensemble=False).fit(x, classes)
    predictions['tuned_sigmoid_full'] = tuned.predict_proba(evaluate)
    predictions['tuned_raw_full'] = tuned.calibrated_classifiers_[0].estimator.predict_proba(evaluate)
    smooth_audit = []
    for degree in PLAN['polynomial_degrees']:
        for regularization in PLAN['logistic_C']:
            fit = smooth_model(degree, regularization).fit(x[train], classes[train])
            loss = log_loss(classes[valid], fit.predict_proba(x[valid]), labels=np.arange(4))
            smooth_audit.append(dict(degree=degree, C=regularization, validation_log_loss=float(loss)))
    smooth_selected = min(smooth_audit, key=lambda r: (r['validation_log_loss'], r['degree'], r['C']))
    fit = smooth_model(smooth_selected['degree'], smooth_selected['C']).fit(x, classes)
    predictions['smooth_multinomial'] = fit.predict_proba(evaluate)
    audit = dict(boosting_selected=selected, boosting_candidates=boost_audit,
                 smooth_selected=smooth_selected, smooth_candidates=smooth_audit,
                 training_rows=len(x), validation_rows=len(valid), model_seed=seed,
                 calibration_note='Hyperparameters selected within training data before calibration CV; no evaluation-label access.')
    return predictions, audit


def source_hashes() -> dict[str, str]:
    return {path: sha256(ROOT/path) for path in SOURCE_FILES}


def prepare(output: Path, native: Path) -> None:
    """Freeze the plan and source snapshot before any comparison is run."""
    hashes = source_hashes()
    metadata = dict(plan=PLAN, sources=hashes, native=str(native.relative_to(ROOT)),
                    native_manifest_sha256=sha256(native/'run_manifest.json'),
                    native_replicates_sha256=sha256(native/'coverage_replicates.csv.gz'),
                    native_truth_sha256=sha256(native/'population_truth.pkl'), runtime=runtime())
    path = output/'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[key] != value for key, value in metadata.items()):
            raise ValueError('Refusing to mix different plans, source code, runtime, or native inputs')
        return
    metadata['git'] = git_state()
    for name in hashes:
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    write_json(path, metadata)


def one_replicate(output: Path, native: Path, kappa: float, replicate: int,
                  saved: pd.DataFrame, truth: float, saved_truth: float) -> dict[str, Any]:
    start = time.monotonic()
    metadata = json.loads((native/'run_manifest.json').read_text())['settings']['parameters']
    n = PLAN['sample_size']
    bits = np.asarray([kappa], dtype='<f8').view('<u4').ravel().tolist()
    rng = np.random.default_rng(np.random.SeedSequence([metadata['seed'], 1, n, *bits])).spawn(metadata['sims'])[replicate]
    x = rng.normal(size=(n, 3))
    shared = kappa*rng.uniform(size=n)
    p = 1/(1+np.exp(-x@np.ones(3)))
    y = ((shared+rng.uniform(size=n)+p)/(kappa+2) > .5).astype(np.int8)
    g = ((shared+rng.uniform(size=n)+p)/(kappa+2) > .5).astype(np.int8)
    seed = int(rng.integers(0, 2**31-1))
    h = n//2
    classes = _encode_joint(g, y)
    if np.bincount(classes[:h], minlength=4).min() < 3:
        raise ValueError('Native retry would be required; do not silently change the paired datasets')
    predictions, audit = fit_candidates(x[:h], classes[:h], x[h:], seed)
    true_q = true_joint(x[h:], kappa)
    predictions['oracle'] = true_q
    rows, diagnostics, examples = [], [], {}
    data_hash = hashlib.sha256(x.astype('<f8').tobytes()+g.tobytes()+y.tobytes()).hexdigest()
    for name in METHODS:
        q = predictions[name]
        np.testing.assert_allclose(q.sum(axis=1), 1., atol=1e-12)
        result = paired_cmi(q, classes[h:])
        if name == 'default_sigmoid_ensemble':
            for method, (estimate, interval) in result['inference'].items():
                archived = saved[(saved.replicate == replicate) & (saved.estimator == method)].iloc[0]
                np.testing.assert_allclose([estimate, *interval], [archived.estimate, archived.ci_low, archived.ci_high],
                                           atol=1e-11, rtol=1e-10)
        records = replicate_records(result, learner=name, kappa=kappa, replicate=replicate,
                                    sample_size=n, data_hash=data_hash)
        for row in records:
            row.update(truth=truth, saved_truth=saved_truth, error=row['estimate']-truth,
                       covered=row['ci_low'] <= truth <= row['ci_high'],
                       covered_saved_truth=row['ci_low'] <= saved_truth <= row['ci_high'])
        rows.extend(records)
        for stage, law in [('initial', result['diagnostics']['initial_q']),
                           ('targeted', result['diagnostics']['q_star'])]:
            diagnostics.append(dict(learner=name, kappa=kappa, replicate=replicate, stage=stage,
                                    evaluation_log_loss=float(log_loss(classes[h:], law, labels=np.arange(4))),
                                    **accuracy(law.copy(), true_q)))
        examples[name] = result['diagnostics']['q_star']
    cell = output/f'kappa_{kappa:g}'
    cell.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cell/f'replicate_{replicate:03d}.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(cell/f'accuracy_{replicate:03d}.csv', index=False)
    audit.update(replicate=replicate, kappa=kappa, data_hash=data_hash,
                 elapsed_seconds=time.monotonic()-start, native_estimates_reproduced=True)
    write_json(cell/f'audit_{replicate:03d}.json', audit)
    if replicate == 0:
        np.savez_compressed(cell/'example.npz', truth=true_q, x=x[h:], **examples)
    write_json(cell/f'complete_{replicate:03d}.json',
               {p.name: sha256(p) for p in [cell/f'replicate_{replicate:03d}.csv',
                                           cell/f'accuracy_{replicate:03d}.csv', cell/f'audit_{replicate:03d}.json']})
    return audit


def run(output: Path, native: Path, kappa: float, start: int, reps: int, jobs: int) -> None:
    metadata = json.loads((output/'manifest.json').read_text())
    if metadata['sources'] != source_hashes() or metadata['plan'] != PLAN:
        raise ValueError('Source or plan differs from the prepared run')
    native_manifest = json.loads((native/'run_manifest.json').read_text())
    for path, expected in native_manifest['code'].items():
        if sha256(ROOT/path) != expected:
            raise ValueError(f'Native computation has changed: {path}')
    if kappa not in PLAN['kappas'] or not 0 <= start < start+reps <= PLAN['replicates']:
        raise ValueError('Requested shard is outside the fixed plan')
    saved = pd.read_csv(native/'coverage_replicates.csv.gz')
    saved = saved[(saved.sample_size == PLAN['sample_size']) & (saved.kappa == kappa)]
    with (native/'population_truth.pkl').open('rb') as stream:
        saved_truth = pickle.load(stream)[kappa]
    truth, _ = exact_truth(kappa)
    pending = [r for r in range(start, start+reps)
               if not (output/f'kappa_{kappa:g}'/f'complete_{r:03d}.json').exists()]
    # Importable workers avoid serializing SciPy ufuncs through __main__.
    from analysis.cmi_learner_review.run import one_replicate as worker
    for audit in Parallel(n_jobs=jobs, return_as='generator_unordered', batch_size=1)(
            delayed(worker)(output, native, kappa, r, saved, truth, saved_truth) for r in pending):
        LOG.info('kappa=%g replicate=%d: all learners complete in %.1fs; baseline reproduced',
                 kappa, audit['replicate'], audit['elapsed_seconds'])


def collect(output: Path, report: Path) -> None:
    metadata = json.loads((output/'manifest.json').read_text())
    if metadata['sources'] != source_hashes():
        raise ValueError('Source changed since run preparation')
    files = []
    for kappa in PLAN['kappas']:
        for replicate in range(PLAN['replicates']):
            cell = output/f'kappa_{kappa:g}'
            receipt = cell/f'complete_{replicate:03d}.json'
            for filename, expected in json.loads(receipt.read_text()).items():
                if sha256(cell/filename) != expected:
                    raise ValueError(f'Corrupt replicate: {cell/filename}')
            files.append(cell/f'replicate_{replicate:03d}.csv')
    raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert len(raw) == len(PLAN['kappas'])*PLAN['replicates']*len(METHODS)*3
    assert not raw.duplicated(['kappa', 'replicate', 'learner', 'estimator']).any()
    assert raw.groupby(['kappa', 'replicate']).data_hash.nunique().eq(1).all()
    accuracy_raw = pd.concat([pd.read_csv(p.with_name(p.name.replace('replicate_', 'accuracy_'))) for p in files])
    rows = []
    for (kappa, learner, estimator), part in raw.groupby(['kappa', 'learner', 'estimator']):
        coverage = float(part.covered.mean())
        rows.append(dict(kappa=kappa, learner=learner, estimator=estimator, replicates=len(part),
                         coverage=coverage, coverage_mcse=np.sqrt(coverage*(1-coverage)/len(part)),
                         coverage_saved_truth=part.covered_saved_truth.mean(), bias=part.error.mean(),
                         bias_mcse=part.error.sem(), rmse=np.sqrt(np.mean(part.error**2)),
                         mean_se=part.se.mean(), empirical_sd=part.estimate.std(),
                         convergence=part.converged.mean(), mean_iterations=part.iterations.mean()))
    summary = pd.DataFrame(rows)
    metrics = [c for c in accuracy_raw.columns if c not in ['learner', 'kappa', 'replicate', 'stage']]
    accuracy_summary = accuracy_raw.groupby(['kappa', 'learner', 'stage'])[metrics].agg(['mean', 'sem'])
    accuracy_summary.columns = ['_'.join(c) for c in accuracy_summary.columns]
    differences = []
    tmle = raw[raw.estimator == 'tmle']
    for kappa, part in tmle.groupby('kappa'):
        reference = part[part.learner == METHODS[0]].set_index('replicate')
        for learner in METHODS[1:]:
            candidate = part[part.learner == learner].set_index('replicate').loc[reference.index]
            error = candidate.estimate-reference.estimate
            coverage = candidate.covered.astype(float)-reference.covered.astype(float)
            differences.append(dict(kappa=kappa, learner=learner, bias_difference=error.mean(),
                                    bias_difference_mcse=error.sem(), coverage_difference=coverage.mean(),
                                    coverage_difference_mcse=coverage.sem()))
    audits = [json.loads(p.with_name(p.name.replace('replicate_', 'audit_')).with_suffix('.json').read_text()) for p in files]
    selections = [dict(kappa=a['kappa'], replicate=a['replicate'], **a['boosting_selected'],
                       smooth_degree=a['smooth_selected']['degree'], smooth_C=a['smooth_selected']['C'],
                       elapsed_seconds=a['elapsed_seconds']) for a in audits]
    report.mkdir(parents=True, exist_ok=True)
    raw.to_csv(report/'replicates.csv.gz', index=False)
    summary.to_csv(report/'summary.csv', index=False)
    accuracy_raw.to_csv(report/'accuracy_replicates.csv.gz', index=False)
    accuracy_summary.to_csv(report/'accuracy_summary.csv')
    pd.DataFrame(differences).to_csv(report/'paired_differences.csv', index=False)
    pd.DataFrame(selections).to_csv(report/'tuning_selections.csv', index=False)
    pd.DataFrame([dict(kappa=k, truth=exact_truth(k)[0], integration_error=exact_truth(k)[1])
                  for k in PLAN['kappas']]).to_csv(report/'truth.csv', index=False)
    plot(summary, accuracy_summary.reset_index(), report)
    write_json(report/'manifest.json', dict(run=str(output.relative_to(ROOT)), run_manifest_sha256=sha256(output/'manifest.json'),
               source_hashes=metadata['sources'], runtime=runtime(),
               native_replicates_reproduced=len(audits),
               targeted_accuracy_caveat='Uses evaluation labels for targeting; oracle accuracy at these X, not independent validation.',
               null_caveat='The oracle EIF is zero at kappa=0; nominal Wald coverage is not a regular inference diagnostic there.',
               files={p.name: sha256(p) for p in report.iterdir() if p.name != 'manifest.json'}))
    LOG.info('Completed matched comparison of %d datasets; TMLE summary:\n%s', len(audits),
             summary[summary.estimator == 'tmle'].to_string(index=False))


def plot(summary: pd.DataFrame, accuracy_summary: pd.DataFrame, report: Path) -> None:
    from matplotlib import pyplot as plt
    from tlfair.plotting import configure_matplotlib
    configure_matplotlib()
    labels = ['Current calibrated', 'Raw fold ensemble', 'Raw full fit', 'Calibrated full fit',
              'Tuned raw', 'Tuned calibrated', 'Smooth multinomial', 'Oracle']
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), layout='constrained')
    positions = np.arange(len(METHODS))
    for j, kappa in enumerate([2.5, 4.]):
        values = summary[(summary.kappa == kappa) & (summary.estimator == 'tmle')].set_index('learner').loc[METHODS]
        accuracy_values = accuracy_summary[(accuracy_summary.kappa == kappa) & (accuracy_summary.stage == 'targeted')].set_index('learner').loc[METHODS]
        ypos = positions+(j-.5)*.22
        for ax, value, error in [(axes[0], values.coverage, values.coverage_mcse),
                                  (axes[1], values.bias, values.bias_mcse),
                                  (axes[2], accuracy_values.group_rmse_mean, accuracy_values.group_rmse_sem)]:
            ax.errorbar(value, ypos, xerr=1.96*error, fmt='o', ms=4, capsize=2, label=rf'$\kappa={kappa:g}$')
    for ax in axes:
        ax.set_yticks(positions, labels if ax is axes[0] else [])
        ax.invert_yaxis()
    axes[0].axvline(.95, color='gray', linestyle='--', linewidth=1)
    axes[0].set_xlabel('TMLE coverage')
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].set_xlabel('TMLE bias (nats)')
    axes[2].set_xlabel('Group probability RMSE')
    axes[2].legend(loc='lower right')
    fig.suptitle('Matched CMI learner comparison: n=10,000; R=100 per setting\nBars show approximate 95% Monte Carlo uncertainty', fontsize=11)
    for suffix in ['png', 'pdf']:
        fig.savefig(report/f'comparison.{suffix}', dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'collect'])
    parser.add_argument('--output-dir', type=Path, default=Path('data/runs/cmi-learner-review-20261021'))
    parser.add_argument('--report-dir', type=Path, default=Path('results/review/cmi-learner-review-20261021'))
    parser.add_argument('--native-dir', type=Path, default=Path('data/runs/cmi-tmle-20261019/cmi_estimation'))
    parser.add_argument('--kappa', type=float, default=2.5)
    parser.add_argument('--rep-start', type=int, default=0)
    parser.add_argument('--reps', type=int, default=10)
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    output, native, report = [p.resolve() for p in [args.output_dir, args.native_dir, args.report_dir]]
    if args.command == 'prepare':
        prepare(output, native)
    elif args.command == 'run':
        run(output, native, args.kappa, args.rep_start, args.reps, args.n_jobs)
    else:
        collect(output, report)


if __name__ == '__main__':
    main()
