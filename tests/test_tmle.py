"""Tests for the TMLE estimators in :mod:`tlfair.tmle`.

These check the properties that make a TMLE valid:
  * the targeted EIF has (numerically) zero empirical mean;
  * the fluctuated outcome model stays strictly in (0, 1);
  * the TMLE point estimate matches the one-step estimator within Monte-Carlo
    noise (they share the same first-order behaviour for these linear-EIF
    estimands);
  * the helper math (logit/expit round-trip, joint encoding, fluctuation score)
    is correct.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from tlfair.metrics import prob_parity, prob_opportunity
from tlfair.tmle import (
    _encode_joint,
    _expit,
    _logit,
    fit_fluctuation,
    prob_opportunity_tmle,
    prob_parity_tmle,
)
from tlfair.simulations import _setting1_draw


def _binary_outcome():
    return LogisticRegression(solver="liblinear")


def _joint_propensity():
    # 4-class joint (G, Y) | X model needs a multiclass-capable solver.
    return LogisticRegression(solver="lbfgs", max_iter=2000)


@pytest.fixture
def setting1():
    rng = np.random.default_rng(0)
    n = 3000
    xg, g, y, _ = _setting1_draw(2 * n, rng, product=True)
    return {
        "xtr": pd.DataFrame(xg[:n]),
        "xte": pd.DataFrame(xg[n:]),
        "ytr": y[:n],
        "yte": y[n:],
        "gtr": g[:n],
        "gte": g[n:],
    }


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------
def test_logit_expit_roundtrip():
    p = np.array([0.01, 0.3, 0.5, 0.7, 0.99])
    np.testing.assert_allclose(_expit(_logit(p)), p, atol=1e-10)


def test_expit_extremes_are_finite():
    z = np.array([-1000.0, 0.0, 1000.0])
    out = _expit(z)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[2] == pytest.approx(1.0, abs=1e-12)


def test_encode_joint():
    g = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(_encode_joint(g, y), [0, 1, 2, 3])


def test_fit_fluctuation_solves_score():
    rng = np.random.default_rng(7)
    n = 5000
    d_hat = rng.uniform(0.1, 0.9, n)
    H = rng.normal(size=n)
    # Generate Y from a known fluctuation so a non-trivial eps is recoverable.
    true_eps = 0.5
    p = _expit(_logit(d_hat) + true_eps * H)
    y = (rng.uniform(size=n) < p).astype(int)
    eps = fit_fluctuation(y, d_hat, H)
    # Score at the fitted eps must be ~0.
    d_star = _expit(_logit(d_hat) + eps * H)
    assert abs(np.sum(H * (y - d_star))) < 1e-6


def test_fit_fluctuation_degenerate_covariate():
    # All-zero clever covariate -> no information -> eps stays 0.
    y = np.array([0, 1, 0, 1])
    d_hat = np.array([0.4, 0.6, 0.5, 0.5])
    H = np.zeros(4)
    assert fit_fluctuation(y, d_hat, H) == 0.0


# ---------------------------------------------------------------------------
# TMLE estimator properties
# ---------------------------------------------------------------------------
def test_prob_parity_tmle_properties(setting1):
    est, ci, diag = prob_parity_tmle(
        outcome=_binary_outcome(), propensity=_binary_outcome(),
        return_diagnostics=True, **setting1,
    )
    assert abs(diag["eif_mean"]) < 1e-8
    assert diag["d_star"].min() > 0.0 and diag["d_star"].max() < 1.0
    assert ci[0] < est < ci[1]


def test_prob_opportunity_tmle_properties(setting1):
    est, ci, diag = prob_opportunity_tmle(
        outcome=_binary_outcome(), propensity=_joint_propensity(),
        return_diagnostics=True, **setting1,
    )
    assert abs(diag["eif_mean"]) < 1e-8
    assert diag["d_star"].min() > 0.0 and diag["d_star"].max() < 1.0
    assert ci[0] < est < ci[1]


def test_tmle_matches_one_step_parity(setting1):
    os_est, _ = prob_parity(
        outcome=_binary_outcome(), propensity=_binary_outcome(), **setting1,
    )
    tm_est, _ = prob_parity_tmle(
        outcome=_binary_outcome(), propensity=_binary_outcome(), **setting1,
    )
    # Same first-order behaviour: agree to a few thousandths at this n.
    assert abs(os_est - tm_est) < 5e-3


def test_tmle_matches_one_step_opportunity(setting1):
    os_est, _ = prob_opportunity(
        outcome=_binary_outcome(), propensity=_joint_propensity(), **setting1,
    )
    tm_est, _ = prob_opportunity_tmle(
        outcome=_binary_outcome(), propensity=_joint_propensity(), **setting1,
    )
    assert abs(os_est - tm_est) < 5e-3


# ---------------------------------------------------------------------------
# CV-TMLE
# ---------------------------------------------------------------------------
def test_cv_tmle_parity_reproducible(setting1):
    kwargs = dict(outcome=_binary_outcome(), propensity=_binary_outcome(),
                  cross_fit=True, n_folds=5)
    est_a, ci_a = prob_parity_tmle(rng=np.random.default_rng(1), **setting1, **kwargs)
    est_b, ci_b = prob_parity_tmle(rng=np.random.default_rng(1), **setting1, **kwargs)
    assert est_a == est_b
    assert ci_a == ci_b


def test_cv_tmle_parity_close_to_single_split(setting1):
    single, _ = prob_parity_tmle(
        outcome=_binary_outcome(), propensity=_binary_outcome(), **setting1,
    )
    cv, _ = prob_parity_tmle(
        outcome=_binary_outcome(), propensity=_binary_outcome(),
        cross_fit=True, n_folds=5, rng=np.random.default_rng(2), **setting1,
    )
    assert abs(single - cv) < 2e-2
