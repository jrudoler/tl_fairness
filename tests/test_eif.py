"""Regression checks for ratio-estimator influence functions and intervals."""

import numpy as np
import pandas as pd
import pytest

from tlfair.metrics import _prob_group_contrast
from tlfair.tmle import _target_parity, _target_opportunity


@pytest.mark.parametrize("opportunity", [False, True])
def test_ratio_eif_matches_numerical_empirical_derivative(opportunity: bool) -> None:
    """Perturb each observation's probability mass, including denominators.

    This detects the historical overall-centering bug without reproducing the
    analytic EIF implementation inside the test.
    """
    rng = np.random.default_rng(41)
    n = 160
    g, y = rng.binomial(1, 0.3, n), rng.binomial(1, 0.65, n)
    d = rng.uniform(0.1, 0.9, n)
    weights = rng.dirichlet([2, 3], n)
    strata = np.column_stack([g == 0, g == 1])
    if opportunity:
        strata &= y[:, None] == 1
        weights *= 0.65
    residual = y - d
    estimate, (lo, hi) = _prob_group_contrast(d, residual, weights, strata)

    def functional(mass: np.ndarray) -> float:
        nums = mass @ (weights * residual[:, None] + strata * d[:, None])
        denoms = mass @ strata
        return float(nums[1] / denoms[1] - nums[0] / denoms[0])

    mass = np.full(n, 1 / n)
    assert estimate == pytest.approx(functional(mass))
    eps = 1e-6
    directions = np.eye(n) - mass
    derivative = np.array([
        (functional(mass + eps * v) - functional(mass - eps * v)) / (2 * eps)
        for v in directions
    ])
    assert (hi - lo) / 3.92 == pytest.approx(np.std(derivative) / np.sqrt(n), rel=1e-7)



def test_fixed_score_limit_has_no_artificial_group_mean_variance() -> None:
    g = np.array([0] * 70 + [1] * 30)
    d = np.linspace(0.1, 0.8, len(g))
    weights = np.column_stack([1 - g, g])
    estimate, (lo, hi) = _prob_group_contrast(d, np.zeros(len(g)), weights, weights)
    expected_var = sum(d[g == k].var() / sum(g == k) for k in (0, 1))
    assert estimate == pytest.approx(d[g == 1].mean() - d[g == 0].mean())
    assert ((hi - lo) / 3.92) ** 2 == pytest.approx(expected_var)
    shifted, shifted_ci = _prob_group_contrast(d + 3, np.zeros(len(g)), weights, weights)
    assert shifted == pytest.approx(estimate)
    np.testing.assert_allclose(shifted_ci, (lo, hi))



@pytest.mark.parametrize("opportunity", [False, True])
def test_tmle_interval_uses_centered_group_scores(opportunity: bool) -> None:
    rng = np.random.default_rng(12)
    n = 2000
    g, y = rng.binomial(1, 0.3, n), rng.binomial(1, 0.6, n)
    d, pi = rng.uniform(0.2, 0.8, n), rng.uniform(0.2, 0.8, n)
    strata = np.column_stack([g == 0, g == 1])
    weights = np.column_stack([1 - pi, pi])
    if opportunity:
        strata &= y[:, None] == 1
        weights *= 0.6
        est, ci, diag = _target_opportunity(d, weights[:, 0], weights[:, 1], y, g,
                                          return_diagnostics=True)
    else:
        est, ci, diag = _target_parity(d, pi, y, g, return_diagnostics=True)
    star = diag["d_star"]
    p = strata.mean(axis=0)
    means = (strata * star[:, None]).mean(axis=0) / p
    phi = ((weights[:, 1] / p[1] - weights[:, 0] / p[0]) * (y - star)
           + strata[:, 1] * (star - means[1]) / p[1]
           - strata[:, 0] * (star - means[0]) / p[0])
    assert est == pytest.approx(means[1] - means[0])
    assert (ci[1] - ci[0]) / 3.92 == pytest.approx(phi.std() / np.sqrt(n))



@pytest.mark.parametrize('metric_name', ['parity', 'opportunity'])
def test_threshold_intervals_match_two_group_mean_variance(metric_name: str) -> None:
    """Conditioning must retain the full row index and use group centering."""
    from tlfair import metrics

    class FixedClassifier:
        def fit(self, x: pd.DataFrame, y: np.ndarray) -> 'FixedClassifier':
            return self

        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return x['prediction'].to_numpy()

    g = np.repeat([0, 1], [80, 120])
    y = np.tile([1, 0, 0, 1], 50)
    pred = np.tile([0, 1, 0, 1, 1], 40)
    x = pd.DataFrame({'prediction': pred})
    eligible = y == 1 if metric_name == 'opportunity' else np.ones(len(g), dtype=bool)
    groups = [pred[(g == k) & eligible] for k in (0, 1)]
    expected = groups[1].mean() - groups[0].mean()
    variance = sum(values.var(ddof=0) / len(values) for values in groups)
    est, (lo, hi) = getattr(metrics, metric_name)(x, x, y, y, g, g, FixedClassifier())
    assert est == pytest.approx(expected)
    assert ((hi - lo) / 3.92) ** 2 == pytest.approx(variance)
    # Constant decisions have no conditional group-mean sampling variance.
    x['prediction'] = 1
    est, ci = getattr(metrics, metric_name)(x, x, y, y, g, g, FixedClassifier())
    assert est == pytest.approx(0)
    np.testing.assert_allclose(ci, [0, 0], atol=1e-14)
