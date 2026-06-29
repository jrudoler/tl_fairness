"""Naive baseline estimators / tests for data-fairness inference (exploratory).

These are the "straw man" alternatives the coverage experiments compare the
targeted-learning (TL) estimators against:

1. ``naive_fixed_model_parity`` -- fix one outcome model, treat its per-group
   predicted means as iid samples, and form a two-sample-mean (CLT) Wald CI.
   This is the "model fairness" view: it ignores the uncertainty in estimating
   P(Y|X), so its CI is too narrow for the *data* estimand.
2. ``glm_ame_parity`` -- assume a logistic GLM and report the average marginal
   effect (AME) of the group on the probability scale, with a *model-based*
   (non-robust) delta-method Wald CI. Correct only if the GLM is well specified;
   under misspecification both the point estimate and the SE are wrong.
3. ``permutation_mi_test`` / ``permutation_cmi_test`` -- permutation tests for
   (marginal vs. conditional) independence. A p-value, not a CI; evaluated as
   Type-I error and power.

The parity estimators return ``(estimate, (ci_low, ci_high))`` to match the
``tlfair.metrics`` signature style so they drop into a shared coverage loop.

Kept outside ``tlfair/`` while exploratory; promote to ``tlfair/baselines.py``
(with tests) once the experiments settle.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit as _sigmoid
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

_Z = 1.96  # match tlfair.metrics._Z (95% standard-normal quantile)


# --------------------------------------------------------------------------- #
# 1. Fixed model + CLT ("model fairness")
# --------------------------------------------------------------------------- #
def naive_fixed_model_parity(model, X_train, y_train, X_test, group_test, z=_Z):
    """Difference in group-mean predicted probabilities with a two-sample CLT CI.

    Fits ``model`` once on the training split, predicts P(Y=1|X) on the test
    split, and treats those predictions as iid data: the CI is the textbook
    Welch-style interval for a difference of means. This deliberately omits the
    EIF correction for nuisance-estimation error, so it under-covers the data
    estimand even when ``model`` is correctly specified.
    """
    m = clone(model)
    m.fit(X_train, y_train)
    p = m.predict_proba(X_test)[:, 1]
    g = np.asarray(group_test)
    p1, p0 = p[g == 1], p[g == 0]
    est = p1.mean() - p0.mean()
    var = p1.var(ddof=1) / len(p1) + p0.var(ddof=1) / len(p0)
    hw = z * np.sqrt(var)
    return est, (est - hw, est + hw)


# --------------------------------------------------------------------------- #
# 2. Misspecified GLM coefficient (average marginal effect)
# --------------------------------------------------------------------------- #
def glm_coef_cov(X, g, y, ridge=1e-8):
    """Fit the logistic GLM and return (beta, design D, p, model_cov, robust_cov).

    Shared core for ``glm_ame_parity`` / ``glm_logodds_coef``. The model is an
    unpenalised logistic regression of ``y`` on ``D = [1, X, g]`` (group last).
    Two coefficient covariances are returned:

      * model_cov  = (D' W D)^{-1},  W = diag(p(1-p))  -- the inverse observed
        information. Valid only under correct specification (information identity).
      * robust_cov = (D' W D)^{-1} (D' diag((y-p)^2) D) (D' W D)^{-1}  -- the
        Huber-White HC0 sandwich, which stays valid under misspecification of
        the conditional mean *or* variance (it does not, of course, fix bias in
        beta itself).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    g = np.asarray(g, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    n = len(y)

    feats = np.hstack([X, g])  # group is the last column
    clf = LogisticRegression(C=np.inf, max_iter=1000).fit(feats, y)  # C=inf -> unpenalised MLE
    beta = np.concatenate([clf.intercept_, clf.coef_.ravel()])  # [b0, X..., bg]

    D = np.hstack([np.ones((n, 1)), feats])  # [1, X, g]
    p = _sigmoid(D @ beta)
    w = p * (1.0 - p)
    bread_inv = (D * w[:, None]).T @ D + ridge * np.eye(D.shape[1])  # observed information
    bread = np.linalg.inv(bread_inv)
    resid2 = (y - p) ** 2
    meat = (D * resid2[:, None]).T @ D
    robust_cov = bread @ meat @ bread
    return beta, D, p, bread, robust_cov


def glm_ame_parity(X, g, y, z=_Z, ridge=1e-8, cov_type="robust"):
    """Logistic-GLM average marginal effect of the group, with a delta-method CI.

    Fits an (unpenalised) logistic regression of ``y`` on design ``[1, X, g]``
    and reports the average marginal effect of the group on the probability
    scale -- the apples-to-apples analogue of probabilistic parity *under the
    model*:

        AME = mean_i [ sigma(D1_i b) - sigma(D0_i b) ],

    where D1/D0 set the group column to 1/0. ``cov_type`` selects the coefficient
    covariance feeding the delta method:

      * "robust" / "HC0" / "HC1" -- Huber-White sandwich (default); honest under
        mean/variance misspecification, so a coverage failure is attributable to
        *bias* in the estimand, not to an optimistic variance.
      * "model" -- inverse observed information (assumes correct specification);
        retained to contrast with the sandwich.
    """
    beta, D, p, model_cov, robust_cov = glm_coef_cov(X, g, y, ridge=ridge)
    n = len(D)
    if cov_type == "model":
        cov = model_cov
    elif cov_type in ("robust", "HC0"):
        cov = robust_cov
    elif cov_type == "HC1":
        cov = robust_cov * (n / (n - D.shape[1]))
    else:
        raise ValueError(f"unknown cov_type: {cov_type!r}")

    D1 = D.copy(); D1[:, -1] = 1.0
    D0 = D.copy(); D0[:, -1] = 0.0
    p1 = _sigmoid(D1 @ beta)
    p0 = _sigmoid(D0 @ beta)
    ame = float(np.mean(p1 - p0))

    # gradient of the AME w.r.t. beta (delta method)
    grad = np.mean((p1 * (1 - p1))[:, None] * D1 - (p0 * (1 - p0))[:, None] * D0, axis=0)
    var = float(grad @ cov @ grad)
    hw = z * np.sqrt(max(var, 0.0))
    return ame, (ame - hw, ame + hw)


def glm_logodds_coef(X, g, y, z=_Z, cov_type="robust"):
    """Raw log-odds coefficient on the group + its Wald CI.

    Illustrates the "wrong scale" point: this estimand lives on the log-odds
    scale, not the probability-difference scale of the parity target, so its CI
    cannot meaningfully cover the parity truth. Returned for reporting only.
    ``cov_type`` is "model" or "robust"/"HC0"/"HC1" (see ``glm_ame_parity``).
    """
    beta, D, p, model_cov, robust_cov = glm_coef_cov(X, g, y)
    n = len(D)
    if cov_type == "model":
        cov = model_cov
    elif cov_type in ("robust", "HC0"):
        cov = robust_cov
    elif cov_type == "HC1":
        cov = robust_cov * (n / (n - D.shape[1]))
    else:
        raise ValueError(f"unknown cov_type: {cov_type!r}")
    coef = float(beta[-1])
    se = float(np.sqrt(cov[-1, -1]))
    return coef, (coef - z * se, coef + z * se)


# --------------------------------------------------------------------------- #
# 3. Permutation tests (marginal vs. conditional independence)
# --------------------------------------------------------------------------- #
def _binary_mi(a, b):
    """Empirical mutual information (nats) of two binary arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    pa1 = np.mean(a == 1)
    pb1 = np.mean(b == 1)
    pa = (1 - pa1, pa1)
    pb = (1 - pb1, pb1)
    mi = 0.0
    for av in (0, 1):
        for bv in (0, 1):
            p = np.count_nonzero((a == av) & (b == bv)) / n
            if p > 0 and pa[av] > 0 and pb[bv] > 0:
                mi += p * np.log(p / (pa[av] * pb[bv]))
    return mi


def permutation_mi_test(x, y, rng, n_perm=200):
    """Global permutation test of *marginal* independence of x and y.

    Statistic: empirical MI of (x, y) ignoring the conditioning features. The
    null distribution permutes y freely, which destroys every association with
    y -- so this tests marginal independence. When x and y are conditionally
    independent given Z but share Z (the CMI DGP at c=0), they are still
    marginally dependent, and this test rejects: a Type-I error *for the
    conditional-independence question*.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    obs = _binary_mi(x, y)
    count = 0
    for _ in range(n_perm):
        if _binary_mi(x, rng.permutation(y)) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def permutation_cmi_test(x, y, z, rng, n_perm=200, n_bins=4, direction=None):
    """Stratified permutation test of *conditional* independence x ⟂ y | Z.

    Bins a 1-D summary of Z into ``n_bins`` quantile strata and permutes y only
    within strata, so the y--Z relationship is preserved and the test targets
    conditional independence. Statistic: stratum-size-weighted within-stratum MI
    (a plug-in conditional-MI estimate). This is the "fair" conditional analogue
    of the global permutation test -- it should be calibrated at c=0, making the
    point that you must condition on Z (which TL does parametrically).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    if direction is None:
        direction = np.ones(z.shape[1])  # matches the DGP's beta = ones
    s = z @ direction
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    strata = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    n = len(y)

    def cmi_stat(yy):
        tot = 0.0
        for k in range(n_bins):
            m = strata == k
            if m.sum() > 1:
                tot += (m.sum() / n) * _binary_mi(x[m], yy[m])
        return tot

    obs = cmi_stat(y)
    idx_by_bin = [np.where(strata == k)[0] for k in range(n_bins)]
    count = 0
    for _ in range(n_perm):
        yperm = y.copy()
        for idx in idx_by_bin:
            if len(idx) > 1:
                yperm[idx] = y[idx][rng.permutation(len(idx))]
        if cmi_stat(yperm) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)
