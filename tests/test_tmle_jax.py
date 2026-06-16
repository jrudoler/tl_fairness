"""Parity tests for the optional JAX backend against the numpy reference.

Skipped entirely when JAX is not installed, so the core test suite never
hard-depends on JAX.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from tlfair.simulations import _setting1_draw
from tlfair.tmle import prob_parity_tmle

jax_backend = pytest.importorskip("tlfair.tmle_jax")
if not jax_backend.HAS_JAX:  # pragma: no cover
    pytest.skip("JAX not installed", allow_module_level=True)

import jax.numpy as jnp  # noqa: E402


def _batch(R, n, seed=123):
    rng = np.random.default_rng(seed)
    Xs, gs, ys = [], [], []
    for _ in range(R):
        xg, g, y, _ = _setting1_draw(n, rng, product=True)
        Xs.append(xg)
        gs.append(g)
        ys.append(y)
    return (np.stack(Xs), np.stack(gs).astype(float), np.stack(ys).astype(float))


def test_jax_matches_numpy_parity():
    R, n = 6, 2000
    X, G, Y = _batch(R, n)
    n_tr = n // 2

    est_j, se_j, lo_j, hi_j = jax_backend.build_batched_parity(n_tr, n_steps=30)(
        jnp.asarray(X), jnp.asarray(G), jnp.asarray(Y)
    )
    est_j = np.asarray(est_j)

    # Unregularised logistic nuisances to match the JAX IRLS (tiny l2).
    def mk():
        return LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    est_n = np.array([
        prob_parity_tmle(
            pd.DataFrame(X[r, :n_tr]), pd.DataFrame(X[r, n_tr:]),
            Y[r, :n_tr], Y[r, n_tr:], G[r, :n_tr], G[r, n_tr:], mk(), mk(),
        )[0]
        for r in range(R)
    ])
    assert np.max(np.abs(est_j - est_n)) < 1e-4


def test_coverage_parity_jax_runs():
    R, n = 8, 1500
    X, G, Y = _batch(R, n, seed=7)
    cov, bias, var = jax_backend.coverage_parity_jax(X, G, Y, n // 2, truth=0.1)
    assert 0.0 <= cov <= 1.0
    assert np.isfinite(bias) and np.isfinite(var)
