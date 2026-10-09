"""Checks for the matched nuisance-training experiment and MC summaries."""
import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.hermite import hermgauss
from joblib import Parallel, delayed
from experiments.nuisance_training import (
    transform, probabilities, draw, integrate, one_rep, summarize, METHODS,
)


def test_transform_moments_and_known_null():
    nodes, weights = hermgauss(60)
    z, w = np.sqrt(2)*nodes, weights/np.sqrt(np.pi)
    for lam in [0, .5, .75, 1]:
        h = transform(z, lam)
        assert w@h == pytest.approx(0, abs=1e-14)
        assert w@(h*h) == pytest.approx(1, abs=1e-14)
    for scenario in ['primary', 'linear', 'nonlinear']:
        result = integrate(scenario, power=14, delta=0)
        assert abs(result['truth']) < max(1e-5, 5*result['integration_se'])


def test_conditional_bernoulli_and_noise_features():
    x, g, y, d, pi = draw(100000, np.random.default_rng(19))
    assert abs(np.mean(g-pi)) < .005
    assert abs(np.mean(y-d)) < .005
    assert abs(np.mean((g-pi)*(y-d))) < .003
    changed = x.copy()
    changed[:, 2:] += 100
    d2, pi2 = probabilities(changed)
    np.testing.assert_array_equal(d, d2)
    np.testing.assert_array_equal(pi, pi2)


def test_population_quadrature_agrees_with_sobol():
    z, w = hermgauss(100)
    u, v = np.meshgrid(np.sqrt(2)*z, np.sqrt(2)*z)
    weights = np.outer(w, w).ravel()/np.pi
    d, pi = probabilities(np.column_stack([u.ravel(), v.ravel()]))
    target = weights@(d*pi)/(weights@pi) - weights@(d*(1-pi))/(weights@(1-pi))
    integration = integrate('primary', power=16)
    assert integration['truth'] == pytest.approx(target, abs=max(2e-6, 5*integration['integration_se']))


def test_matched_predictions_and_worker_reproducibility():
    args = [('primary', 100, 100, r) for r in range(3)]
    sequential = [one_rep(*a, cases=['default', 'linear_linear', 'oracle_oracle']) for a in args]
    parallel = Parallel(n_jobs=2)(delayed(one_rep)(*a, cases=['default', 'linear_linear', 'oracle_oracle']) for a in args)
    pd.testing.assert_frame_equal(pd.DataFrame(sum(sequential, [])), pd.DataFrame(sum(parallel, [])))
    raw = pd.DataFrame(sum(sequential, []))
    assert raw.groupby(['replicate', 'case']).outcome_hash.nunique().eq(1).all()
    assert raw.groupby('replicate').data_hash.nunique().eq(1).all()
    assert raw.groupby(['replicate', 'case']).method.nunique().eq(2).all()


def test_mcse_signed_bias_boundaries_and_paired_covariance():
    rows = []
    for case, covers in [('default', [True, False, True, False]),
                         ('all', [True]*4), ('none', [False]*4)]:
        for r, covered in enumerate(covers):
            for method, shift in zip(METHODS, [0, .5]):
                # Coverage events are identical within each matched pair.
                rows.append(dict(scenario='primary', train_size=100, test_size=2000,
                                 case=case, outcome_model='default', group_model='default',
                                 method=method, replicate=r, estimate=r-4+shift,
                                 ci_low=-10 if covered else 1, ci_high=10 if covered else 2,
                                 width=20 if covered else 1, se=1))
    _, summary, paired = summarize(pd.DataFrame(rows), pd.DataFrame([dict(scenario='primary', truth=0)]))
    tl = summary[(summary.case == 'default') & (summary.method == METHODS[0])].iloc[0]
    assert tl.bias == -2.5
    assert tl.bias_mcse == pytest.approx(np.std([-4,-3,-2,-1], ddof=1)/2)
    assert tl.coverage_mcse == .25
    assert summary.loc[summary.case.isin(['all','none']), 'coverage_mcse'].eq(0).all()
    assert paired.coverage_difference_mcse.eq(0).all()
    assert paired.bias_difference_mcse.eq(0).all()
    assert paired.bias_difference.eq(-.5).all()
    assert summary.coverage_low.ge(0).all() and summary.coverage_high.le(1).all()


@pytest.mark.parametrize('include_cv', [False, True])
def test_figure_uses_signed_percentage_points_and_mc_intervals(tmp_path, monkeypatch, include_cv):
    from analysis.fig_trainsize.run import plot
    import matplotlib.pyplot as plt
    rows = []
    cases = ['default', 'higher_capacity'] + (['cross_validated'] if include_cv else [])
    for case in cases:
        for method in METHODS:
            for n, bias in [(100, -.02), (1000, .005)]:
                rows.append(dict(scenario='primary', case=case, method=method, train_size=n,
                                 bias=bias, bias_mcse=.001, coverage=.9, coverage_mcse=.01, reps=1000, test_size=2000))
    closed = []
    monkeypatch.setattr(plt, 'close', lambda fig=None: closed.append(fig))
    plot(pd.DataFrame(rows), tmp_path/'plot.pdf')
    fig = next(f for f in closed if hasattr(f, 'axes'))
    assert len(fig.axes) == 2
    assert len(fig.axes[1].containers) == 2*len(cases)
    assert fig.axes[1].get_ylabel() == 'Bias'
    assert ('CV-tuned boosting' in [t.get_text() for t in fig.legends[0].get_texts()]) == include_cv
    for container in fig.axes[1].containers:
        np.testing.assert_allclose(np.asarray(container.lines[0].get_ydata(), dtype=float), [-2, .5])
    assert fig.axes[1].get_ylim()[0] < 0
    for container in fig.axes[0].containers:
        bars = container.lines[2][0].get_segments()
        np.testing.assert_allclose(bars[0][:, 1], [.9-1.96*.01, .9+1.96*.01])


def test_oracle_and_quadratic_models_have_correct_basis():
    from scipy.special import expit
    x = np.random.default_rng(11).normal(size=(100, 5))
    lam = .5
    linear = np.sqrt(1-lam*lam)
    quad = lam/np.sqrt(2)
    d, pi = probabilities(x)
    d_design = expit(-1.6*quad + .6*linear*x[:,0] + linear*x[:,1]
                    + .6*quad*x[:,0]**2 + quad*x[:,1]**2)
    pi_design = expit(-quad + linear*x[:,0] + quad*x[:,0]**2)
    np.testing.assert_allclose(d, d_design)
    np.testing.assert_allclose(pi, pi_design)


def test_resume_does_not_refit_completed_replicates(tmp_path, monkeypatch):
    import experiments.nuisance_training as experiment
    def fake_rep(scenario, nt, ne, r, seed):
        return [dict(replicate=r, scenario=scenario, estimate=float(r))]
    monkeypatch.setattr(experiment, 'one_rep', fake_rep)
    experiment.run_shard(tmp_path, 'primary', 100, reps=2)
    before = {p.name: p.read_bytes() for p in tmp_path.glob('rep_*.csv')}
    def fail(*args):
        raise AssertionError('Completed replicate was refitted')
    monkeypatch.setattr(experiment, 'one_rep', fail)
    experiment.run_shard(tmp_path, 'primary', 100, reps=2, n_jobs=2)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob('rep_*.csv')}
    with pytest.raises(ValueError, match='incompatible'):
        experiment.run_shard(tmp_path, 'primary', 100, reps=3)


def test_publication_tables_reject_smoke_run(tmp_path):
    from analysis.nuisance_tables.run import render
    with pytest.raises(ValueError, match='1000 replicates'):
        render(pd.DataFrame([dict(reps=20, test_size=2000)]), pd.DataFrame(), tmp_path, pd.DataFrame())
