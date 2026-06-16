"""Targeted Maximum Likelihood Estimation (TMLE) for data-fairness metrics.

The estimators in :mod:`tlfair.metrics` are *one-step* / AIPW estimators: a
plug-in estimate plus an additive efficient-influence-function (EIF) correction.
This module implements the *targeted* alternative for the doubly-robust
(probabilistic) metrics. Instead of adding the EIF correction term, TMLE
fluctuates the initial outcome model ``D(x)=P(Y=1|X)`` along a logistic submodel

    logit D*(x) = logit D_hat(x) + eps * H(x)

whose score is the EIF's nuisance-residual term, fits the scalar ``eps`` by
maximum likelihood, and reports the plug-in (substitution) estimate of the
fluctuated fit. Because ``eps`` solves the score equation, the empirical mean of
the EIF is driven to (approximately) zero and the substitution estimate respects
the natural [0, 1] bound on probabilities.

The public estimators share the signature used throughout ``tlfair.metrics``,
``metric(xtr, xte, ytr, yte, gtr, gte, outcome, propensity=None) -> (est, ci)``,
so they are drop-in compatible with ``perm_importance`` and the analysis
``metric_map``. A ``cross_fit=True`` option (or the standalone
:func:`cross_fit_tmle`) switches to K-fold cross-fitted nuisances with a single
pooled fluctuation (CV-TMLE), matching the tmle3 default.

Notes on the threshold metrics (``parity``, ``opportunity``): their EIF has no
nuisance-residual augmentation term, so the estimating-equation solution is just
the plug-in mean of the (thresholded) outcome model and TMLE coincides with the
existing one-step estimator. We therefore provide targeting only for the
probabilistic metrics here.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

# Clip probabilities into [_EPS, 1 - _EPS] before taking a logit so the Newton
# fluctuation stays finite when the outcome model predicts values at 0/1.
_EPS = 1e-6


def _clip(p):
    return np.clip(p, _EPS, 1.0 - _EPS)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1.0 - p))


def _expit(z):
    # Numerically stable logistic.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _concat_rows(a, b):
    """Row-concatenate two feature blocks that may be DataFrames or ndarrays."""
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        return pd.concat([pd.DataFrame(a).reset_index(drop=True),
                          pd.DataFrame(b).reset_index(drop=True)],
                         ignore_index=True)
    return np.concatenate([np.asarray(a), np.asarray(b)], axis=0)


def _encode_joint(g, y):
    """Encode binary (G, Y) into 4 classes: 0=(0,0),1=(1,0),2=(0,1),3=(1,1)."""
    g = np.asarray(g).astype(np.int8)
    y = np.asarray(y).astype(np.int8)
    return (g + 2 * y).astype(np.int8)


def fit_fluctuation(y, d_hat, H, *, max_iter=50, tol=1e-10):
    """Fit the scalar logistic-fluctuation parameter ``eps`` by Newton steps.

    Solves the offset-logistic MLE score equation ``sum_i H_i (Y_i - D*_i) = 0``
    where ``D*_i = expit(logit D_hat_i + eps * H_i)``. The fluctuation has a
    single covariate ``H`` (the clever covariate) and no intercept, so ``eps`` is
    a scalar and the Newton update is a 1-D ratio.

    Returns the fitted ``eps`` (0.0 when the covariate is degenerate).
    """
    y = np.asarray(y, dtype=float)
    H = np.asarray(H, dtype=float)
    logit_d = _logit(d_hat)
    eps = 0.0
    for _ in range(max_iter):
        d_star = _expit(logit_d + eps * H)
        score = np.sum(H * (y - d_star))
        info = np.sum(H ** 2 * d_star * (1.0 - d_star))
        if not np.isfinite(info) or info < 1e-12:
            break
        step = score / info
        eps += step
        if abs(step) < tol:
            break
    return eps


def _wald_ci(est, eif):
    se = np.sqrt(np.var(eif) / len(eif))
    return (est - 1.96 * se, est + 1.96 * se)


# ---------------------------------------------------------------------------
# Clever covariates (the multiplier on the (Y - D_hat) residual in the EIF).
# ---------------------------------------------------------------------------
def clever_covariate_parity(pi1, p_g1):
    """Stacked, signed clever covariate for probabilistic demographic parity.

    The EIF augmentation for arm g is ``+/- pi_g(x) / P(G=g)`` times ``(Y-D(x))``;
    combined across the two arms the per-observation multiplier on ``(Y - D)`` is

        H(x) = pi(x)/P(G=1) - (1 - pi(x))/P(G=0).
    """
    pi1 = np.asarray(pi1, dtype=float)
    p_g0 = 1.0 - p_g1
    return pi1 / p_g1 - (1.0 - pi1) / p_g0


def clever_covariate_opportunity(rho0, rho1, p_yg0, p_yg1):
    """Stacked, signed clever covariate for probabilistic equal opportunity.

    Uses ``rho_g(x) = P(Y=1, G=g | X)`` and stratum frequencies
    ``P(Y=1, G=g)``; per-observation multiplier on ``(Y - D)`` is

        H(x) = rho_1(x)/P(Y=1,G=1) - rho_0(x)/P(Y=1,G=0).
    """
    rho0 = np.asarray(rho0, dtype=float)
    rho1 = np.asarray(rho1, dtype=float)
    return rho1 / p_yg1 - rho0 / p_yg0


# ---------------------------------------------------------------------------
# Targeting + substitution estimate, given cross-fitted / held-out nuisances.
# ---------------------------------------------------------------------------
def _target_parity(d_hat, pi1, y, g, *, return_diagnostics=False):
    g = np.asarray(g)
    y = np.asarray(y)
    p_g1 = np.mean(g == 1)
    p_g0 = 1.0 - p_g1
    H = clever_covariate_parity(pi1, p_g1)
    eps = fit_fluctuation(y == 1, d_hat, H)
    d_star = _expit(_logit(d_hat) + eps * H)

    plug1 = (g == 1) * d_star / p_g1
    plug0 = (g == 0) * d_star / p_g0
    est = np.mean(plug1) - np.mean(plug0)
    # EIF at the targeted fit: residual term + plug-in term - estimand.
    eif = H * ((y == 1) - d_star) + plug1 - plug0 - est
    ci = _wald_ci(est, eif)
    if return_diagnostics:
        return est, ci, {"eps": eps, "eif_mean": float(np.mean(eif)),
                         "d_star": d_star}
    return est, ci


def _target_opportunity(d_hat, rho0, rho1, y, g, *, return_diagnostics=False):
    g = np.asarray(g)
    y = np.asarray(y)
    yg1 = (g == 1) & (y == 1)
    yg0 = (g == 0) & (y == 1)
    p_yg1 = np.mean(yg1)
    p_yg0 = np.mean(yg0)
    H = clever_covariate_opportunity(rho0, rho1, p_yg0, p_yg1)
    eps = fit_fluctuation(y == 1, d_hat, H)
    d_star = _expit(_logit(d_hat) + eps * H)

    plug1 = yg1 * d_star / p_yg1
    plug0 = yg0 * d_star / p_yg0
    est = np.mean(plug1) - np.mean(plug0)
    eif = H * ((y == 1) - d_star) + plug1 - plug0 - est
    ci = _wald_ci(est, eif)
    if return_diagnostics:
        return est, ci, {"eps": eps, "eif_mean": float(np.mean(eif)),
                         "d_star": d_star}
    return est, ci


# ---------------------------------------------------------------------------
# Public single-split TMLE estimators (metric-API compatible).
# ---------------------------------------------------------------------------
def prob_parity_tmle(xtr, xte, ytr, yte, gtr, gte, outcome, propensity=None,
                     *, cross_fit=False, n_folds=5, rng=None, backend="numpy",
                     return_diagnostics=False):
    """TMLE for probabilistic demographic parity (paper Eq. 8-10)."""
    if backend != "numpy":
        raise NotImplementedError(
            "JAX backend for the single-split estimators is not provided; the "
            "JAX path targets the batched coverage simulation (see "
            "tlfair.tmle_jax / tlfair.tmle_sim)."
        )
    if cross_fit:
        X = _concat_rows(xtr, xte)
        y = np.concatenate([np.asarray(ytr), np.asarray(yte)])
        g = np.concatenate([np.asarray(gtr), np.asarray(gte)])
        return cross_fit_tmle(X, y, g, outcome, propensity,
                              metric="prob_parity", n_folds=n_folds, rng=rng,
                              return_diagnostics=return_diagnostics)

    outcome = outcome.fit(xtr, ytr)
    propensity = propensity.fit(xtr, gtr)
    d_hat = outcome.predict_proba(xte)[:, 1]
    pi1 = propensity.predict_proba(xte)[:, 1]
    return _target_parity(d_hat, pi1, yte, gte,
                          return_diagnostics=return_diagnostics)


def prob_opportunity_tmle(xtr, xte, ytr, yte, gtr, gte, outcome, propensity=None,
                          *, cross_fit=False, n_folds=5, rng=None,
                          backend="numpy", return_diagnostics=False):
    """TMLE for probabilistic equal opportunity (paper Appendix B / Eq. 17-19)."""
    if backend != "numpy":
        raise NotImplementedError(
            "JAX backend for the single-split estimators is not provided; the "
            "JAX path targets the batched coverage simulation (see "
            "tlfair.tmle_jax / tlfair.tmle_sim)."
        )
    if cross_fit:
        X = _concat_rows(xtr, xte)
        y = np.concatenate([np.asarray(ytr), np.asarray(yte)])
        g = np.concatenate([np.asarray(gtr), np.asarray(gte)])
        return cross_fit_tmle(X, y, g, outcome, propensity,
                              metric="prob_opportunity", n_folds=n_folds,
                              rng=rng, return_diagnostics=return_diagnostics)

    yg_tr = _encode_joint(gtr, ytr)
    outcome = outcome.fit(xtr, ytr)
    propensity = propensity.fit(xtr, yg_tr)
    d_hat = outcome.predict_proba(xte)[:, 1]
    props = _aligned_joint_proba(propensity, xte)
    rho0, rho1 = props[:, 2], props[:, 3]
    return _target_opportunity(d_hat, rho0, rho1, yte, gte,
                               return_diagnostics=return_diagnostics)


def _aligned_joint_proba(model, X):
    """predict_proba aligned to the 4 joint classes 0..3 (missing classes -> 0)."""
    probs = model.predict_proba(X)
    aligned = np.zeros((len(probs), 4))
    for j, cls in enumerate(model.classes_):
        aligned[:, int(cls)] = probs[:, j]
    return aligned


# ---------------------------------------------------------------------------
# CV-TMLE: cross-fitted nuisances + single pooled fluctuation.
# ---------------------------------------------------------------------------
def _kfold_seed(rng):
    if rng is None:
        return None
    if isinstance(rng, (int, np.integer)):
        return int(rng)
    return int(rng.integers(0, 2 ** 31 - 1))


def cross_fit_tmle(X, y, g, outcome, propensity, *, metric="prob_parity",
                   n_folds=5, rng=None, return_diagnostics=False):
    """K-fold cross-fitted TMLE with a single pooled fluctuation (CV-TMLE).

    Nuisances are fit on out-of-fold rows and predicted on the held-out fold, so
    every observation receives a cross-fitted nuisance value with no own-fold
    leakage. The clever covariate and the single scalar ``eps`` are then fit over
    the pooled cross-fitted predictions, and the EIF is pooled for the variance.
    """
    X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y = np.asarray(y)
    g = np.asarray(g)
    n = len(X)
    d_hat = np.empty(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=_kfold_seed(rng))

    if metric == "prob_parity":
        pi1 = np.empty(n)
        for tr_idx, te_idx in kf.split(X):
            om = clone(outcome).fit(X.iloc[tr_idx], y[tr_idx])
            pm = clone(propensity).fit(X.iloc[tr_idx], g[tr_idx])
            d_hat[te_idx] = om.predict_proba(X.iloc[te_idx])[:, 1]
            pi1[te_idx] = pm.predict_proba(X.iloc[te_idx])[:, 1]
        return _target_parity(d_hat, pi1, y, g,
                              return_diagnostics=return_diagnostics)

    if metric == "prob_opportunity":
        rho0 = np.empty(n)
        rho1 = np.empty(n)
        yg = _encode_joint(g, y)
        for tr_idx, te_idx in kf.split(X):
            om = clone(outcome).fit(X.iloc[tr_idx], y[tr_idx])
            pm = clone(propensity).fit(X.iloc[tr_idx], yg[tr_idx])
            d_hat[te_idx] = om.predict_proba(X.iloc[te_idx])[:, 1]
            props = _aligned_joint_proba(pm, X.iloc[te_idx])
            rho0[te_idx] = props[:, 2]
            rho1[te_idx] = props[:, 3]
        return _target_opportunity(d_hat, rho0, rho1, y, g,
                                   return_diagnostics=return_diagnostics)

    raise ValueError(f"unknown metric for CV-TMLE: {metric!r}")


# ---------------------------------------------------------------------------
# CMI: substitution estimate, softmax fluctuation, boundary-aware CI.
# ---------------------------------------------------------------------------
# Finding (derivation in the plan): the conditional mutual information estimand
# has the complete efficient influence function phi = log[q/(a b)] - Psi (the
# paper's Eq. 12). A full point-mass/Gateaux derivation -- perturbing the
# marginal of X and the conditional joint law q(y,g|x) -- shows the suspected
# extra correction terms cancel: the inner gradient dh/dq = log[q/(a b)] - 1 is
# constant across classes after contraction, and the marginal Y|X / G|X
# perturbations contribute zero at the truth (their gradient is the constant -1
# and the perturbation direction sums to zero). So:
#   * Both metrics.cmi and metrics.cmi_separate already solve their empirical EIF
#     equation by construction (they set Psi_hat = mean of the observed log-ratio,
#     so mean(EIF) = 0 identically). There is no "missing augmentation" to add.
#   * The poor coverage near CMI~0 is a BOUNDARY non-regularity: Psi >= 0 and the
#     sampling distribution is one-sided there, so symmetric Wald CIs under-cover
#     regardless of the EIF. The fix is positivity + a boundary-aware CI, not a
#     richer EIF.
# This module therefore contributes (1) a positivity-respecting substitution
# estimate, mean_i KL(q(.|x_i) || a (x) b) >= 0, instead of the average raw
# log-ratio (which can go negative near independence); (2) an optional softmax
# fluctuation of the joint PMF that drives the empirical EIF mean to zero; and
# (3) a boundary-aware CI option. cmi_separate's instability is a second-order
# plug-in bias from using marginals (a_y, b_g) that are not the marginals of the
# fitted joint q; the principled remedy is the single/joint approach used here.
#
# Class indexing matches _encode_joint: class c has g = c % 2, y = c // 2, i.e.
#   0=(g0,y0)  1=(g1,y0)  2=(g0,y1)  3=(g1,y1).
def _cmi_log_ratio(q):
    """Per-class log-ratio L_c(x) = log[ q_c / (a_{y(c)} b_{g(c)}) ] for q:(n,4)."""
    a0 = q[:, 0] + q[:, 1]      # P(Y=0|x)
    a1 = q[:, 2] + q[:, 3]      # P(Y=1|x)
    b0 = q[:, 0] + q[:, 2]      # P(G=0|x)
    b1 = q[:, 1] + q[:, 3]      # P(G=1|x)
    denom = np.column_stack([a0 * b0, a0 * b1, a1 * b0, a1 * b1])
    return np.log(_clip(q) / _clip(denom))


def _softmax_rows(logq):
    m = logq.max(axis=1, keepdims=True)
    e = np.exp(logq - m)
    return e / e.sum(axis=1, keepdims=True)


def _cmi_target(q, lte, *, fluctuate=True, max_iter=50, tol=1e-8):
    """Substitution CMI estimate (+ optional softmax targeting) and its EIF.

    Returns ``(est, eif, info)`` where ``est = mean_i h(x_i)`` with
    ``h(x) = sum_c q_c log[q_c/(a_y b_g)] = KL(q(.|x) || a(.|x) (x) b(.|x)) >= 0``,
    and the (complete, paper Eq. 12) EIF evaluated at the observed class,
    ``phi_i = L_{C_i}(x_i) - est``. The optional fluctuation reweights q on the
    softmax scale with clever covariate ``H_c = L_c - h`` so the empirical EIF
    mean is driven to zero.
    """
    q = _clip(np.asarray(q, dtype=float))
    q = q / q.sum(axis=1, keepdims=True)
    n = len(q)
    idx = np.arange(n)
    eps_final = 0.0

    if fluctuate:
        logq = np.log(q)
        for _ in range(max_iter):
            q = _softmax_rows(logq)
            L = _cmi_log_ratio(q)
            h = np.sum(q * L, axis=1)
            H = L - h[:, None]                 # E_q[H] = 0 by construction
            score = np.sum(H[idx, lte])        # = sum_i (L_{C_i} - h_i)
            info = np.sum(q * H ** 2)           # = sum_i Var_q(H_i)
            if not np.isfinite(info) or info < 1e-12:
                break
            eps = score / info
            logq = logq + eps * H
            eps_final = eps
            if abs(eps) < tol:
                break
        q = _softmax_rows(logq)

    L = _cmi_log_ratio(q)
    h = np.sum(q * L, axis=1)
    est = float(np.mean(h))
    eif = L[idx, lte] - est
    info = {
        "eps": eps_final,
        "eif_mean": float(np.mean(eif)),
        "naive_mean_logratio": float(np.mean(L[idx, lte])),
        "pointwise_min": float(h.min()),
    }
    return est, eif, info


def cmi_tmle(xtr, xte, ytr, yte, gtr, gte, outcome, propensity=None, *,
             fluctuate=True, boundary_ci=False, max_iter=50, tol=1e-8,
             return_diagnostics=False):
    """TMLE-style CMI estimator: positivity-respecting substitution + targeting.

    Mirrors :func:`tlfair.metrics.cmi` (single, joint 4-class approach) but:
      * the point estimate is the substitution form ``mean_i h(x_i) >= 0`` rather
        than the average raw log-ratio, which can go negative near independence;
      * an optional softmax fluctuation of the calibrated joint PMF drives the
        empirical EIF mean to zero;
      * ``boundary_ci=True`` truncates the lower confidence limit at 0, a simple
        boundary-aware adjustment for the near-independence (CMI~0) regime where
        the sampling distribution is one-sided and symmetric Wald under-covers.

    The ``(xtr, ytr, gtr)`` naming follows the metric API: ``ytr``/``gte`` are the
    two discrete variables whose conditional MI given ``X`` is estimated.
    """
    ltr = _encode_joint(gtr, ytr)
    model = CalibratedClassifierCV(outcome, cv=3).fit(xtr, ltr)
    q = _aligned_joint_proba(model, xte)
    lte = _encode_joint(gte, yte)

    est, eif, info = _cmi_target(q, lte, fluctuate=fluctuate,
                                 max_iter=max_iter, tol=tol)
    se = np.sqrt(np.var(eif) / len(eif))
    lower, upper = est - 1.96 * se, est + 1.96 * se
    if boundary_ci:
        lower = max(0.0, lower)
    ci = (lower, upper)
    if return_diagnostics:
        return est, ci, info
    return est, ci
