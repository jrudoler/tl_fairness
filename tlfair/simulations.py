"""Simulation data-generating processes and estimators for the parity settings.

These functions back the parity/opportunity simulations (paper Section 4.1,
Figures 1 and 3). They were consolidated here from the exploratory notebooks so
that the figure entrypoints in ``analysis/`` import a single, tested source
rather than redefining the DGP inline.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tlfair.metrics import *  # noqa: F401,F403  (re-exported metrics for sim1)

# Setting 1 covariance: Cov(X2,X3) = -Cov(X4,X5) = 0.5 (paper Section 4.1).
SETTING1_COV = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0.5, 0, 0],
    [0, 0.5, 1, 0, 0],
    [0, 0, 0, 1, -0.5],
    [0, 0, 0, -0.5, 1],
])


def _setting1_draw(n, rng, proportion=0.5, product=True):
    """Draw (xg, g, y, y_probs) for Setting 1 with n total samples."""
    x = rng.multivariate_normal(mean=np.zeros(5), cov=SETTING1_COV, size=(n,))
    g = (rng.uniform(size=(n,)) > proportion).astype(np.int8)
    xg = x.copy()
    xg[:, 1] = xg[:, 1] + 0.5 * g
    xg[:, 4] = xg[:, 4] - 0.5 * g

    if product:
        x14 = (xg[:, 1] * xg[:, 4]).reshape((len(xg), 1))
        x23 = (xg[:, 2] * xg[:, 3]).reshape((len(xg), 1))
        coef = np.array([-2, 3, -4, 3, -1, 2, 1])
        xg_input = np.hstack((xg, x14, x23))
    else:
        coef = np.array([-2, 3, -4, 3, -1])
        xg_input = xg

    y_probs = 1 / (1 + np.exp(-xg_input @ coef))
    y = (y_probs > 0.5).astype(np.int8)
    return xg, g, y, y_probs


def parity_ground_truth(n=100000, proportion=0.5, product=True, rng=None):
    """Monte Carlo ground truth for threshold and probabilistic parity (Setting 1)."""
    if rng is None:
        rng = np.random.default_rng()
    _, g, y, y_probs = _setting1_draw(n, rng, proportion=proportion, product=product)
    threshold_truth = np.mean(y[g == 1]) - np.mean(y[g == 0])
    prob_truth = np.mean(y_probs[g == 1]) - np.mean(y_probs[g == 0])
    return threshold_truth, prob_truth


def sim1(n, metric, rng=None, proportion=0.5):
    """Single Setting-1 draw evaluated with a TL ``metric`` from ``tlfair.metrics``."""
    n = n // 2
    if rng is None:
        rng = np.random.default_rng()
    xg, g, y, _ = _setting1_draw(2 * n, rng, proportion=proportion, product=True)

    return metric(
        xtr=pd.DataFrame(xg[:n, :]),
        xte=pd.DataFrame(xg[n:, :]),
        ytr=y[:n],
        yte=y[n:],
        gtr=g[:n],
        gte=g[n:],
        outcome=LogisticRegression(solver='liblinear'),
        propensity=LogisticRegression(solver='liblinear'),
    )


def parity_sim(n, proportion=0.5, parity='threshold', product=True, rng=None):
    """One Setting-1 replicate returning (TL est, TL var, naive est, naive var).

    Used for the TL-vs-naive variance comparison (Figure 3). ``n`` is the
    per-split size (train and test each get ``n`` samples).
    """
    if rng is None:
        rng = np.random.default_rng()
    xg, g, y, _ = _setting1_draw(2 * n, rng, proportion=proportion, product=product)

    xgtrain, xgtest = xg[:n, :], xg[n:, :]
    gtrain, gtest = g[:n], g[n:]
    ytrain, ytest = y[:n], y[n:]

    model = LogisticRegression().fit(xgtrain, ytrain)

    if parity == 'threshold':
        preds = model.predict(xgtest)
        phi0 = -1 / np.mean(gtest == 0) * preds[np.where(gtest == 0)[0]]
        phi1 = 1 / np.mean(gtest == 1) * preds[np.where(gtest == 1)[0]]
        phi = np.hstack([phi0, phi1])
        est = np.mean(phi)
        eif = phi - np.mean(phi)
        var = np.var(eif) / n
        naive_est = np.mean(preds[np.where(gtest == 1)[0]]) - np.mean(preds[np.where(gtest == 0)[0]])
        naive_var = (np.var(preds[np.where(gtest == 1)[0]]) + np.var(preds[np.where(gtest == 0)[0]])) / n
    elif parity == 'prob':
        propensity = LogisticRegression().fit(xgtrain, gtrain)
        m_probs = propensity.predict_proba(xgtest)
        f_probs = model.predict_proba(xgtest)[:, 1]
        phi0 = -1 / (np.mean(gtest == 0)) * (m_probs[:, 0] * ((ytest == 1) - f_probs) + (gtest == 0) * f_probs)
        phi1 = 1 / (np.mean(gtest == 1)) * (m_probs[:, 1] * ((ytest == 1) - f_probs) + (gtest == 1) * f_probs)
        phi = phi0 + phi1
        est = np.mean(phi)
        eif = phi - est
        var = np.var(eif) / n
        naive_est = np.mean(f_probs[np.where(gtest == 1)[0]]) - np.mean(f_probs[np.where(gtest == 0)[0]])
        naive_var = (np.var(f_probs[np.where(gtest == 1)[0]]) + np.var(f_probs[np.where(gtest == 0)[0]])) / n
    else:
        raise ValueError(f"unknown parity type: {parity!r}")

    return est, var, naive_est, naive_var


def coverage_sim_parity(ground_truth, parity='threshold', product=True,
                        n_sim=100, n_samples=100, proportion=0.5, rng=None):
    """Repeated Setting-1 replicates returning (truth, estimates, std, upper, lower).

    Backs the estimate-vs-sample-size and CI panels of Figure 1. ``n_samples``
    is the total sample size (split evenly into train/test).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = n_samples // 2  # split into train/test
    estimates = np.zeros(n_sim)
    upper = np.zeros(n_sim)
    lower = np.zeros(n_sim)
    std = np.zeros(n_sim)

    for i in range(n_sim):
        xg, g, y, _ = _setting1_draw(2 * n, rng, proportion=proportion, product=product)
        xgtrain, xgtest = xg[:n, :], xg[n:, :]
        gtrain, gtest = g[:n], g[n:]
        ytrain, ytest = y[:n], y[n:]

        model = LogisticRegression().fit(xgtrain, ytrain)

        if parity == 'threshold':
            phi0 = -1 / np.mean(gtest == 0) * model.predict(xgtest[np.where(gtest == 0)[0], :])
            phi1 = 1 / np.mean(gtest == 1) * model.predict(xgtest[np.where(gtest == 1)[0], :])
            phi = np.hstack([phi0, phi1])
            est = np.mean(phi)
            eif = phi - np.mean(phi)
        elif parity == 'prob':
            propensity = LogisticRegression().fit(xgtrain, gtrain)
            m_probs = propensity.predict_proba(xgtest)
            f_probs = model.predict_proba(xgtest)[:, 1]
            phi0 = -1 / (np.mean(gtest == 0)) * (m_probs[:, 0] * ((ytest == 1) - f_probs) + (gtest == 0) * f_probs)
            phi1 = 1 / (np.mean(gtest == 1)) * (m_probs[:, 1] * ((ytest == 1) - f_probs) + (gtest == 1) * f_probs)
            phi = phi0 + phi1
            est = np.mean(phi)
            eif = phi - (np.mean(phi1) + np.mean(phi0))
        else:
            raise ValueError(f"unknown parity type: {parity!r}")

        estimates[i] = est
        std[i] = np.sqrt(np.var(eif) / n)
        upper[i] = est + 1.96 * np.sqrt(np.var(eif) / n)
        lower[i] = est - 1.96 * np.sqrt(np.var(eif) / n)

    return ground_truth, estimates, std, upper, lower
