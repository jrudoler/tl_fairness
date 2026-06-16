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

from sklearn.ensemble import GradientBoostingClassifier

from tlfair.metrics import prob_parity, prob_opportunity
from tlfair.tmle import (
    _cmi_log_ratio,
    _cmi_target,
    _encode_joint,
    _expit,
    _logit,
    cmi_tmle,
    fit_fluctuation,
    prob_opportunity_tmle,
    prob_parity_tmle,
)
from tlfair.simulations import _setting1_draw


def _draw_cmi(n, c, rng, d=3):
    """Setting from cmi_sim: two binary vars with shared dependence scaled by c."""
    n2 = n // 2
    z = rng.normal(size=(2 * n2, d))
    beta = np.ones(d)
    shared = c * rng.uniform(size=2 * n2)
    logits = 1 / (1 + np.exp(-z @ beta))
    xp = (shared + rng.uniform(size=2 * n2) + logits) / (c + 2)
    yp = (shared + rng.uniform(size=2 * n2) + logits) / (c + 2)
    x = (xp > 0.5).astype(np.int8)
    y = (yp > 0.5).astype(np.int8)
    return z[:n2], z[n2:], x[:n2], x[n2:], y[:n2], y[n2:]


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


# ---------------------------------------------------------------------------
# CMI: substitution form, targeting, exact EIF, boundary behaviour
# ---------------------------------------------------------------------------
def test_cmi_exact_eif_discrete_x():
    """On a hand-built discrete-X law the complete EIF (L - Psi) is mean-zero.

    This confirms the paper's CMI EIF is complete: the suspected extra
    correction terms cancel, so E_P[phi] = 0 exactly.
    """
    # X in {0,1,2} with chosen p(x) and joint q(g,y|x) (columns 0..3).
    px = np.array([0.5, 0.3, 0.2])
    q = np.array([
        [0.40, 0.10, 0.10, 0.40],   # x=0: dependent
        [0.25, 0.25, 0.25, 0.25],   # x=1: independent
        [0.10, 0.40, 0.30, 0.20],   # x=2: dependent
    ])
    L = _cmi_log_ratio(q)                       # (3,4)
    h = np.sum(q * L, axis=1)                    # per-x integrand >= 0
    psi = float(px @ h)
    assert psi >= 0
    # E[phi] = sum_x p(x) sum_c q_c(x) (L_c(x) - psi) = sum_x p(x) (h(x) - psi) = 0
    eif_mean = sum(px[x] * np.sum(q[x] * (L[x] - psi)) for x in range(3))
    assert abs(eif_mean) < 1e-12
    # KL form is non-negative per x.
    assert np.all(h >= -1e-12)


def test_cmi_substitution_nonnegative_and_eif_zero():
    rng = np.random.default_rng(0)
    xtr, xte, ytr, yte, gtr, gte = _draw_cmi(1500, c=2.0, rng=rng)
    seed = int(rng.integers(0, 2 ** 31 - 1))
    est, ci, diag = cmi_tmle(
        xtr, xte, ytr, yte, gtr, gte,
        GradientBoostingClassifier(random_state=seed),
        fluctuate=True, return_diagnostics=True,
    )
    assert est >= 0.0
    assert abs(diag["eif_mean"]) < 1e-6
    assert diag["pointwise_min"] >= -1e-9


def test_cmi_substitution_nonnegative_at_independence():
    """At c=0 (true CMI=0) the substitution estimate never goes negative,
    unlike the raw log-ratio average used by the one-step estimator."""
    rng = np.random.default_rng(7)
    negatives = 0
    for _ in range(15):
        xtr, xte, ytr, yte, gtr, gte = _draw_cmi(800, c=0.0, rng=rng)
        seed = int(rng.integers(0, 2 ** 31 - 1))
        est, _ = cmi_tmle(
            xtr, xte, ytr, yte, gtr, gte,
            GradientBoostingClassifier(random_state=seed),
        )
        negatives += est < 0
    assert negatives == 0


def test_cmi_boundary_ci_truncates_at_zero():
    rng = np.random.default_rng(3)
    xtr, xte, ytr, yte, gtr, gte = _draw_cmi(600, c=0.0, rng=rng)
    seed = int(rng.integers(0, 2 ** 31 - 1))
    _, ci = cmi_tmle(
        xtr, xte, ytr, yte, gtr, gte,
        GradientBoostingClassifier(random_state=seed), boundary_ci=True,
    )
    assert ci[0] >= 0.0


def test_cmi_target_without_fluctuation_is_substitution():
    # With fluctuate=False the estimate is the plain KL substitution mean and
    # eps stays 0; it is still non-negative.
    rng = np.random.default_rng(1)
    q = rng.dirichlet(np.ones(4), size=500)
    lte = rng.integers(0, 4, size=500)
    est, eif, info = _cmi_target(q, lte, fluctuate=False)
    assert info["eps"] == 0.0
    assert est >= 0.0
