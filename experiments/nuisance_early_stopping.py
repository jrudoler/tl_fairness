"""Matched boosting experiments with training-only early stopping.

Primary family extends the shared/outcome-only predictor structure of Fig. 1.
Derivative of the frozen original experiment; all nuisances are refit on the original draws.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, ndtri
from scipy.stats import qmc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import sklearn
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tlfair.metrics import _prob_group_contrast
from experiments.parity_retraining import model_interval

SCENARIOS = {'primary': 0.5, 'linear': 0.0, 'nonlinear': 0.75, 'stress': None}
SIZES = [100, 250, 500, 1000, 2500, 5000, 10000]
CONTROL_SIZES = [500, 2500, 10000]
EARLY_STOPPING = dict(validation_fraction=0.2, n_iter_no_change=20, tol=1e-4)
BOOST = dict(n_estimators=100, max_depth=3, learning_rate=0.1, **EARLY_STOPPING)
HIGH = dict(n_estimators=1000, max_depth=5, learning_rate=0.1, **EARLY_STOPPING)
METHODS = ['TL data fairness', 'Model fairness']
CASES = {
    'default': ('default', 'default'),
    'higher_capacity': ('higher_capacity', 'higher_capacity'),
    'boost_linear': ('default', 'linear'),
    'linear_boost': ('linear', 'default'),
    'linear_linear': ('linear', 'linear'),
    'oracle_oracle': ('oracle', 'oracle'),
    'oracle_linear': ('oracle', 'linear'),
    'linear_oracle': ('linear', 'oracle'),
    'quadratic_quadratic': ('quadratic', 'quadratic'),
    'quadratic_linear': ('quadratic', 'linear'),
    'linear_quadratic': ('linear', 'quadratic'),
}
KEYS = ['scenario', 'train_size', 'test_size', 'case', 'outcome_model', 'group_model', 'method']
LOG = logging.getLogger(__name__)


def transform(z, lam):
    if not 0 <= lam <= 1:
        raise ValueError('lambda must lie in [0, 1]')
    return np.sqrt(1-lam**2)*z + lam*(z**2-1)/np.sqrt(2)


def probabilities(x, scenario='primary', a=1., b=1., delta=.6):
    if scenario == 'stress':
        return expit(x**2 @ np.array([4, 2, 1, -3, -4])), expit(x**2 @ np.array([1, 1, -1, -2, 1]))
    lam = SCENARIOS[scenario]
    u, v = transform(x[:, 0], lam), transform(x[:, 1], lam)
    return expit(b*v + delta*u), expit(a*u)


def draw(n, rng, scenario='primary', **params):
    x = rng.normal(size=(n, 5))
    d, pi = probabilities(x, scenario, **params)
    # Independent conditional Bernoulli draws, not deterministic thresholds.
    g = rng.binomial(1, pi)
    y = rng.binomial(1, d)
    return x, g, y, d, pi


def integrate(scenario, power=18, **params):
    values, prevalence = [], []
    pi_all = []
    for scramble in range(8):
        dimension = 5 if scenario == 'stress' else 2
        uniforms = qmc.Sobol(dimension, scramble=True, seed=3100+scramble).random_base2(power)
        x = ndtri(uniforms)
        d, pi = probabilities(x, scenario, **params)
        p = pi.mean()
        values.append(np.mean(pi*d)/p - np.mean((1-pi)*d)/(1-p).mean())
        prevalence.append([p, d.mean()])
        if scramble == 0:
            pi_all = pi
    return dict(scenario=scenario, truth=float(np.mean(values)),
                integration_se=float(np.std(values, ddof=1)/np.sqrt(8)), power=power,
                group_prevalence=float(np.mean(prevalence, axis=0)[0]),
                outcome_prevalence=float(np.mean(prevalence, axis=0)[1]),
                propensity_q01=float(np.quantile(pi_all, .01)),
                propensity_q05=float(np.quantile(pi_all, .05)),
                propensity_q50=float(np.quantile(pi_all, .5)),
                propensity_q95=float(np.quantile(pi_all, .95)),
                propensity_q99=float(np.quantile(pi_all, .99)))


def cases_for(scenario, n_train):
    return list(CASES) if scenario == 'primary' and n_train in CONTROL_SIZES else ['default', 'higher_capacity']


def one_rep(scenario, n_train, n_eval, replicate, seed=20261005, cases=None):
    scenario_id = list(SCENARIOS).index(scenario)
    rng = np.random.default_rng(np.random.SeedSequence([seed, scenario_id, n_train, n_eval, replicate]))
    x, g, y, d, pi = draw(n_train+n_eval, rng, scenario)
    xt, xe, ge, ye = x[:n_train], x[n_train:], g[n_train:], y[n_train:]
    case_names = cases if cases is not None else cases_for(scenario, n_train)
    needed = {('outcome', CASES[c][0]) for c in case_names} | {('group', CASES[c][1]) for c in case_names}
    fitted, hashes, diagnostics = {}, {}, {}
    for role, response, truth in [('outcome', y, d), ('group', g, pi)]:
        for name in ['default', 'higher_capacity', 'linear', 'quadratic', 'oracle']:
            if (role, name) not in needed:
                continue
            if name == 'oracle':
                pred = truth[n_train:]
            else:
                model_seed = int(np.random.SeedSequence([seed, scenario_id, n_train, n_eval, replicate,
                                                        0 if role == 'outcome' else 1]).generate_state(1)[0])
                if name in ['default', 'higher_capacity']:
                    model = GradientBoostingClassifier(**(BOOST if name == 'default' else HIGH), random_state=model_seed)
                    train, evaluate = xt, xe
                else:
                    model = LogisticRegression(C=1e8, max_iter=1000, tol=1e-9, solver='lbfgs')
                    train, evaluate = ((np.column_stack([xt, xt**2]), np.column_stack([xe, xe**2]))
                                       if name == 'quadratic' else (xt, xe))
                model.fit(train, response[:n_train])
                if name in ['linear', 'quadratic'] and model.n_iter_[0] >= 1000:
                    raise RuntimeError('Logistic fit did not converge')
                pred = model.predict_proba(evaluate)[:, 1]
            diagnostics[role, name] = dict(
                trees=int(model.n_estimators_) if name in ['default', 'higher_capacity'] else 0,
                probability_mse=float(np.mean((pred-truth[n_train:])**2)),
                evaluation_brier=float(np.mean((pred-response[n_train:])**2)))
            fitted[role, name] = pred
            hashes[role, name] = hashlib.sha256(np.asarray(pred, dtype='<f8').tobytes()).hexdigest()
    data_hash = hashlib.sha256(x.astype('<f8').tobytes()+g.tobytes()+y.tobytes()).hexdigest()
    rows = []
    strata = np.column_stack([ge == 0, ge == 1])
    for case in case_names:
        outcome, group = CASES[case]
        dh, ph = fitted['outcome', outcome], fitted['group', group]
        estimates = [
            _prob_group_contrast(dh, ye-dh, np.column_stack([1-ph, ph]), strata),
            model_interval(dh, ge),
        ]
        for method, (estimate, (lo, hi)) in zip(METHODS, estimates):
            rows.append(dict(scenario=scenario, train_size=n_train, test_size=n_eval,
                             replicate=replicate, case=case, outcome_model=outcome, group_model=group,
                             method=method, estimate=estimate, ci_low=lo, ci_high=hi,
                             se=(hi-lo)/3.92, width=hi-lo, data_hash=data_hash,
                             outcome_hash=hashes['outcome', outcome], group_hash=hashes['group', group],
                             **{role+'_'+key:value for role,name in [('outcome',outcome),('group',group)]
                                for key,value in diagnostics[role,name].items()}))
    return rows


def summarize(raw, truths):
    raw = raw.copy()
    raw['truth'] = raw.scenario.map(truths.set_index('scenario').truth)
    if raw.truth.isna().any():
        raise ValueError('Missing population truth')
    raw['error'] = raw.estimate-raw.truth
    raw['covered'] = (raw.ci_low <= raw.truth) & (raw.truth <= raw.ci_high)
    summary = raw.groupby(KEYS, sort=True).agg(
        reps=('estimate', 'size'), coverage=('covered', 'mean'), bias=('error', 'mean'),
        empirical_sd=('estimate', 'std'), mean_se=('se', 'mean'),
        mean_ci_width=('width', 'mean'), mean_estimate=('estimate', 'mean'), truth=('truth', 'first')).reset_index()
    summary['coverage_mcse'] = np.sqrt(summary.coverage*(1-summary.coverage)/summary.reps)
    summary['bias_mcse'] = summary.empirical_sd/np.sqrt(summary.reps)
    summary['coverage_low'] = (summary.coverage-1.96*summary.coverage_mcse).clip(0, 1)
    summary['coverage_high'] = (summary.coverage+1.96*summary.coverage_mcse).clip(0, 1)
    summary['bias_low'] = summary.bias-1.96*summary.bias_mcse
    summary['bias_high'] = summary.bias+1.96*summary.bias_mcse
    pairs = []
    for key, sub in raw.groupby(KEYS[:-1], sort=True):
        pivot = sub.pivot(index='replicate', columns='method', values=['covered', 'error'])
        row = dict(zip(KEYS[:-1], key))
        row['reps'] = len(pivot)
        for name, column in [('coverage_difference', 'covered'), ('bias_difference', 'error')]:
            diff = pivot[column][METHODS[0]].astype(float)-pivot[column][METHODS[1]].astype(float)
            row[name] = diff.mean()
            row[name+'_mcse'] = diff.std(ddof=1)/np.sqrt(len(diff))
        pairs.append(row)
    return raw, summary, pd.DataFrame(pairs)


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT/'tlfair/metrics.py', ROOT/'experiments/parity_retraining.py', ROOT/'experiments/nuisance_training.py']}


def run_shard(output, scenario, n_train, n_eval=2000, reps=1000, rep_start=0, seed=20261005, n_jobs=1):
    if reps < 1 or rep_start < 0 or n_jobs == 0 or n_train < 20 or n_eval < 20:
        raise ValueError('Invalid replication or sample-size settings')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(scenario=scenario, train_size=n_train, test_size=n_eval, reps=reps,
                rep_start=rep_start, seed=seed, source_hashes=source_hashes(),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, default_learner=BOOST, higher_capacity_learner=HIGH,
                primary_parameters=dict(a=1, b=1, delta=.6, lam=.5))
    meta_path = output/'metadata.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != spec:
        raise ValueError(f'Refusing to resume incompatible run: {output}')
    meta_path.write_text(json.dumps(spec, indent=2)+'\n')
    pending = [r for r in range(rep_start, rep_start+reps) if not (output/f'rep_{r:05d}.csv').exists()]
    generator = Parallel(n_jobs=n_jobs, return_as='generator_unordered', batch_size=1)(
        delayed(one_rep)(scenario, n_train, n_eval, r, seed) for r in pending)
    for rows in generator:
        r = rows[0]['replicate']
        temp = output/f'rep_{r:05d}.csv.tmp'
        pd.DataFrame(rows).to_csv(temp, index=False)
        temp.replace(output/f'rep_{r:05d}.csv')
        LOG.info('%s n=%s rep=%s complete', scenario, n_train, r)


def collect(input_dir, output_dir, expected_reps=1000, reference_dir='data/generated/nuisance_training'):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    files = sorted(input_dir.rglob('rep_*.csv'))
    if not files:
        raise ValueError('No replicate files found')
    raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    expected_grid = {(s, n) for s in SCENARIOS for n in (SIZES if s == 'primary' else CONTROL_SIZES)}
    if set(raw[['scenario', 'train_size']].itertuples(index=False, name=None)) != expected_grid:
        raise ValueError('Full prespecified scenario/sample-size grid is incomplete')
    identity = KEYS+['replicate']
    if raw.duplicated(identity).any():
        raise ValueError('Duplicate replicate records')
    for key, sub in raw.groupby(['scenario', 'train_size']):
        expected_cases = cases_for(*key)
        expected = {(c, m, r) for c in expected_cases for m in METHODS for r in range(expected_reps)}
        if set(sub[['case', 'method', 'replicate']].itertuples(index=False, name=None)) != expected:
            raise ValueError(f'Incomplete simulation configuration {key}')
    baseline = pd.read_csv(Path(reference_dir)/'replicates.csv.gz')
    from analysis.nuisance_tables.control_audit import audit_unchanged
    control_audit = audit_unchanged(raw, baseline)
    truths=pd.read_csv(Path(reference_dir)/'population.csv')
    for scenario,sub in raw.groupby('scenario'):
        min_se=sub.groupby(KEYS).estimate.std().min()/np.sqrt(expected_reps)
        idx=truths.index[truths.scenario==scenario][0]
        while truths.loc[idx,'integration_se'] >= .01*min_se:
            power=int(truths.loc[idx,'power'])+1
            if power>23: raise RuntimeError('Population integration precision not reached')
            info=integrate(scenario,power)
            for key,value in info.items(): truths.loc[idx,key]=value
    raw, summary, pairs = summarize(raw, truths)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'control_audit.json').write_text(json.dumps(control_audit,indent=2)+'\n')
    raw.to_csv(output_dir/'replicates.csv.gz', index=False)
    summary.to_csv(output_dir/'summary.csv', index=False)
    pairs.to_csv(output_dir/'paired_comparisons.csv', index=False)
    truths.to_csv(output_dir/'population.csv', index=False)
    metas = {str(p.relative_to(input_dir)): json.loads(p.read_text()) for p in sorted(input_dir.rglob('metadata.json'))}
    (output_dir/'manifest.json').write_text(json.dumps(dict(expected_reps=expected_reps, shards=metas,
                                                          collection_source_hashes=source_hashes(), early_stopping=EARLY_STOPPING), indent=2)+'\n')
    diagnostic_rows = raw.drop_duplicates(['scenario','train_size','replicate','outcome_model','group_model'])
    diagnostic_rows.to_csv(output_dir/'fit_diagnostics.csv.gz',index=False)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--scenario', choices=SCENARIOS, default='primary')
    run.add_argument('--train-size', type=int, required=True)
    run.add_argument('--test-size', type=int, default=2000)
    run.add_argument('--reps', type=int, default=1000)
    run.add_argument('--rep-start', type=int, default=0)
    run.add_argument('--seed', type=int, default=20261005)
    run.add_argument('--n-jobs', type=int, default=1)
    run.add_argument('--output-dir', type=Path, required=True)
    merge = sub.add_parser('collect')
    merge.add_argument('--input-dir', type=Path, required=True)
    merge.add_argument('--output-dir', type=Path, default=Path('data/generated/nuisance_early_stopping'))
    merge.add_argument('--expected-reps', type=int, default=1000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.command == 'run':
        run_shard(args.output_dir, args.scenario, args.train_size, args.test_size,
                  args.reps, args.rep_start, args.seed, args.n_jobs)
    else:
        collect(args.input_dir, args.output_dir, args.expected_reps)


if __name__ == '__main__':
    main()
