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


def _setting1_draw(n, rng, proportion=0.5, product=True, bernoulli=False):
    """Draw (xg, g, y, y_probs) for Setting 1 with n total samples.

    ``bernoulli`` (default False) selects how the binary outcome is realised:
      * False -- deterministic Bayes decision ``y = 1{y_probs > 0.5}`` (the
        original behaviour; no extra RNG draw, so the default path is unchanged).
      * True  -- a genuine Bernoulli draw ``y ~ Bernoulli(y_probs)``, so that
        ``P(Y=1|X) = y_probs`` actually holds (no zero conditional variance, no
        perfect separation). The estimand truths are unchanged either way -- they
        are defined via ``y_probs`` / the Bayes decision, not the realised ``y``.
    """
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
    if bernoulli:
        y = (rng.uniform(size=(len(y_probs),)) < y_probs).astype(np.int8)
    else:
        y = (y_probs > 0.5).astype(np.int8)
    return xg, g, y, y_probs


def parity_ground_truth(n=100000, proportion=0.5, product=True, rng=None):
    """Monte Carlo ground truth for threshold and probabilistic parity (Setting 1)."""
    if rng is None:
        rng = np.random.default_rng()
    _, g, _, y_probs = _setting1_draw(n, rng, proportion=proportion, product=product)
    # Threshold parity targets the Bayes decision D_c(X)=1{D(X)>=0.5}, so compute
    # it directly from D(X)=y_probs rather than from the realised label. This is
    # a.s. identical to the old `mean(y[...])` under the deterministic draw, but
    # stays correct (targets E[D_c|G=g], not the probabilistic E[D|G=g]) even if a
    # Bernoulli-outcome draw were ever forwarded here.
    decision = (y_probs >= 0.5).astype(np.int8)
    threshold_truth = np.mean(decision[g == 1]) - np.mean(decision[g == 0])
    prob_truth = np.mean(y_probs[g == 1]) - np.mean(y_probs[g == 0])
    return threshold_truth, prob_truth


# Setting 3 coefficients: Y and G are logistic in the *squared* features, so a
# linear logistic model is misspecified for both nuisances (paper Section 4.1,
# Figure 2). Lifted here from analysis/sim_robust/run.py so the misspecification
# DGP has a single tested source shared by the pipeline and the baseline
# experiments. The single ``rng.normal`` draw keeps the order identical to the
# original inline code, so the protected sim_robust output is unchanged.
SETTING3_DIM = 5
SETTING3_OUTCOME_COEF = np.array([4, 2, 1, -3, -4])
SETTING3_GROUP_COEF = np.array([1, 1, -1, -2, 1])


def setting3_draw(n, rng, bernoulli=False, bernoulli_group=False):
    """Draw (x, g, y, y_probs) for Setting 3 with n total samples.

    ``x`` is the raw 5-D Gaussian feature matrix; the true ``y_probs`` and group
    propensity are logistic in ``x**2``. Estimators receive the raw ``x``: a
    flexible learner (e.g. gradient boosting) can recover the quadratic signal,
    while a linear logistic model is misspecified.

    The manuscript defines both as Bernoulli: P(Y=1|X) and P(G=1|X) are logistic
    in x**2. The two flags realise that (default off keeps the original
    deterministic-threshold code path bit-identical -- no extra RNG draw):
      * ``bernoulli``       -- ``y ~ Bernoulli(y_probs)`` (else ``y=1{y_probs>=.5}``);
        makes a well-specified logistic outcome model non-separable.
      * ``bernoulli_group`` -- ``g ~ Bernoulli(g_probs)`` (else ``g=1{g_probs>=.5}``);
        gives a genuine propensity ``pi(x)=g_probs in (0,1)`` so the overlap /
        positivity assumption holds (the deterministic threshold makes pi in
        {0,1}). ``setting3_truth`` must use the SAME ``bernoulli_group`` value so
        its Monte-Carlo estimand matches the data's group law.
    """
    x = rng.normal(size=(n, SETTING3_DIM))
    xt = x ** 2
    y_probs = 1 / (1 + np.exp(-xt @ SETTING3_OUTCOME_COEF))
    if bernoulli:
        y = (rng.uniform(size=n) < y_probs).astype(np.int8)
    else:
        y = (y_probs >= 0.5).astype(np.int8)
    g_probs = 1 / (1 + np.exp(-xt @ SETTING3_GROUP_COEF))
    if bernoulli_group:
        g = (rng.uniform(size=n) < g_probs).astype(np.int8)
    else:
        g = (g_probs >= 0.5).astype(np.int8)
    return x, g, y, y_probs


def setting3_truth(n, rng, bernoulli_group=False):
    """Monte Carlo ground truth for probabilistic parity under Setting 3.

    Uses ``y_probs`` (= D(X)) so it is invariant to outcome noise, but the group
    law matters: ``bernoulli_group`` must match the data's so the conditional
    means E[D(X)|G=g] are taken under the same propensity.
    """
    _, g, _, y_probs = setting3_draw(n, rng, bernoulli_group=bernoulli_group)
    return np.mean(y_probs[g == 1]) - np.mean(y_probs[g == 0])


def sim1(n, metric, rng=None, proportion=0.5):
    """Single Setting-1 draw evaluated with a TL ``metric`` from ``tlfair.metrics``."""
    n = n // 2
    if rng is None:
        rng = np.random.default_rng()
    xg, g, y, _ = _setting1_draw(2 * n, rng, proportion=proportion, product=True)

    return metric(
        X_train=pd.DataFrame(xg[:n, :]),
        X_test=pd.DataFrame(xg[n:, :]),
        y_train=y[:n],
        y_test=y[n:],
        group_train=g[:n],
        group_test=g[n:],
        outcome=LogisticRegression(solver='liblinear'),
        propensity=LogisticRegression(solver='liblinear'),
    )


def parity_sim(n, proportion=0.5, parity='threshold', product=True, rng=None,
               bernoulli=False):
    """One Setting-1 replicate returning (TL est, TL var, naive est, naive var).

    Used for the TL-vs-naive variance comparison (Figure 3). ``n`` is the
    per-split size (train and test each get ``n`` samples). ``bernoulli`` passes
    through to ``_setting1_draw`` (draw y ~ Bernoulli(y_probs) vs. the Bayes
    decision); the truth is unchanged either way.
    """
    if rng is None:
        rng = np.random.default_rng()
    xg, g, y, _ = _setting1_draw(2 * n, rng, proportion=proportion, product=product,
                                 bernoulli=bernoulli)

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
                        n_sim=100, n_samples=100, proportion=0.5, rng=None,
                        bernoulli=False):
    """Repeated Setting-1 replicates returning (truth, estimates, std, upper, lower).

    Backs the estimate-vs-sample-size and CI panels of Figure 1. ``n_samples``
    is the total sample size (split evenly into train/test). ``bernoulli`` passes
    through to ``_setting1_draw``; the truth is unchanged either way.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = n_samples // 2  # split into train/test
    estimates = np.zeros(n_sim)
    upper = np.zeros(n_sim)
    lower = np.zeros(n_sim)
    std = np.zeros(n_sim)

    for i in range(n_sim):
        xg, g, y, _ = _setting1_draw(2 * n, rng, proportion=proportion, product=product,
                                     bernoulli=bernoulli)
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
