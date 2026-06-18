"""Targeted-learning estimators for data-fairness metrics.

Every estimator shares the same interface and return contract:

    estimate, (ci_low, ci_high) = metric(
        X_train, X_test, y_train, y_test, group_train, group_test,
        outcome, propensity=None,
    )

``X_*`` are feature frames, ``y_*`` the binary outcome, ``group_*`` the binary
protected attribute. ``outcome``/``propensity`` are (unfitted) sklearn-style
classifiers fit internally on the training split and applied to the test split
(sample splitting). The confidence interval is a 95% Wald interval built from the
sample variance of the efficient influence function (EIF).

This shared signature is intentional: the estimators are dispatched
interchangeably -- ``perm_importance`` below and the ``metric_map`` in the
analyze_* scripts call every metric with the full keyword set -- so each must
accept all of these parameters even when it does not use them. In particular the
threshold metrics ``parity`` and ``opportunity`` take no ``propensity`` model
(their decision rule is a hard threshold), and ``cmi``/``cmi_separate`` model the
joint via ``outcome`` alone, so ``propensity`` is accepted-but-ignored there.
Removing it would break the uniform dispatch.

The estimand for each metric is noted in its docstring with the paper reference.
"""

import copy
from math import factorial

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

_Z = 1.96  # standard-normal quantile for a 95% interval


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _clone_estimator(estimator):
    """Return an unfitted copy of ``estimator`` (None passes through)."""
    if estimator is None:
        return None
    try:
        return clone(estimator)
    except TypeError:
        return copy.deepcopy(estimator)


def _wald_ci(estimate, eif):
    """95% Wald interval from the EIF's sample variance."""
    half_width = _Z * np.sqrt(np.var(eif) / len(eif))
    return (estimate - half_width, estimate + half_width)


def _encode_joint(group, y):
    """Encode the joint state (group, y) into one of four classes:

        0 = (G0, Y0)   1 = (G1, Y0)   2 = (G0, Y1)   3 = (G1, Y1)

    i.e. ``class = 1[group==1] + 2 * 1[y==1]``.
    """
    group = np.asarray(group)
    y = np.asarray(y)
    return (group == 1).astype(np.int8) + 2 * (y == 1).astype(np.int8)


def _cmi_log_ratio(joint_proba, p_g0, p_g1, p_y0, p_y1, observed_class):
    """Per-observation CMI integrand ``log[ p(g,y|x) / (p(g|x) p(y|x)) ]``.

    ``joint_proba`` is the (n, 4) estimate of ``p(G,Y|X)`` (columns ordered by
    the 4-class encoding above); the four marginal vectors are its (or separate
    models') ``p(G=g|X)`` and ``p(Y=y|X)``. The numerator/denominator are picked
    out per row by the observed joint class.
    """
    rows = np.arange(len(joint_proba))
    joint = joint_proba[rows, observed_class]
    marginal = np.choose(observed_class, [p_g0 * p_y0, p_g1 * p_y0,
                                          p_g0 * p_y1, p_g1 * p_y1])
    return np.log(joint / marginal)


def _positive_rate(predict, X_test, mask, weight_total):
    """Inverse-probability-weighted decision rate over the rows in ``mask``.

    ``predict`` is applied to the masked rows; the result is scaled by
    ``1 / weight_total`` (an empirical stratum probability). Used by the
    threshold metrics (parity / equal opportunity).
    """
    return predict(X_test.iloc[np.where(mask)[0], :]) / weight_total


# ---------------------------------------------------------------------------
# Threshold metrics (Bayes-optimal decision rule D_c(x) = 1{P(Y=1|X) >= c}).
# The EIF carries no outcome-residual term, so these are plug-in estimators.
# ---------------------------------------------------------------------------
def parity(X_train, X_test, y_train, y_test, group_train, group_test,
           outcome, propensity=None):
    """Demographic parity, Ψ = E[D_c(X)|G=1] − E[D_c(X)|G=0] (paper Eq. 5)."""
    # unused (uniform dispatch): y_test, group_train, propensity
    outcome = outcome.fit(X_train, y_train)
    group_test = np.asarray(group_test)
    phi = np.hstack([
        -_positive_rate(outcome.predict, X_test, group_test == 0, np.mean(group_test == 0)),
        _positive_rate(outcome.predict, X_test, group_test == 1, np.mean(group_test == 1)),
    ])
    estimate = np.mean(phi)
    return estimate, _wald_ci(estimate, phi - estimate)


def opportunity(X_train, X_test, y_train, y_test, group_train, group_test,
                outcome, propensity=None):
    """Equal opportunity, Ψ = E[D_c(X)|Y=1,G=1] − E[D_c(X)|Y=1,G=0] (paper Eq. 14)."""
    # unused (uniform dispatch): group_train, propensity
    outcome = outcome.fit(X_train, y_train)
    y_test = np.asarray(y_test)
    group_test = np.asarray(group_test)
    positive = y_test == 1
    in_g0 = positive & (group_test == 0)
    in_g1 = positive & (group_test == 1)
    phi = np.hstack([
        -_positive_rate(outcome.predict, X_test, in_g0, np.mean(in_g0)),
        _positive_rate(outcome.predict, X_test, in_g1, np.mean(in_g1)),
    ])
    # Sum of stratum contributions averaged over the full evaluation sample.
    estimate = np.sum(phi) / len(group_test)
    return estimate, _wald_ci(estimate, phi - estimate)


# ---------------------------------------------------------------------------
# Probabilistic metrics (D(x) = P(Y=1|X)). Doubly robust: an outcome model and
# a group/stratum model, combined via the EIF residual augmentation.
# ---------------------------------------------------------------------------
def prob_parity(X_train, X_test, y_train, y_test, group_train, group_test,
                outcome, propensity=None):
    """Probabilistic demographic parity, Ψ = E[D(X)|G=1] − E[D(X)|G=0] (Eqs. 8, 10)."""
    outcome = outcome.fit(X_train, y_train)
    propensity = propensity.fit(X_train, group_train)
    y_test = np.asarray(y_test)
    group_test = np.asarray(group_test)
    pi = propensity.predict_proba(X_test)           # P(G=g | X), columns [G0, G1]
    d_hat = outcome.predict_proba(X_test)[:, 1]      # D(X) = P(Y=1 | X)
    residual = (y_test == 1) - d_hat                 # outcome-model residual
    in_g0 = group_test == 0
    in_g1 = group_test == 1
    phi0 = -(pi[:, 0] * residual + in_g0 * d_hat) / np.mean(in_g0)
    phi1 = (pi[:, 1] * residual + in_g1 * d_hat) / np.mean(in_g1)
    phi = phi0 + phi1
    estimate = np.mean(phi)
    return estimate, _wald_ci(estimate, phi - estimate)


def prob_opportunity(X_train, X_test, y_train, y_test, group_train, group_test,
                     outcome, propensity=None):
    """Probabilistic equal opportunity (paper Appendix B / Eqs. 17–19).

    The propensity model estimates the joint stratum p(G,Y|X) (4 classes); the
    (G0,Y1) and (G1,Y1) columns (2 and 3) supply the clever-covariate weights.
    """
    outcome = outcome.fit(X_train, y_train)
    propensity = propensity.fit(X_train, _encode_joint(group_train, y_train))
    y_test = np.asarray(y_test)
    group_test = np.asarray(group_test)
    rho = propensity.predict_proba(X_test)           # P(G,Y | X), 4 classes
    d_hat = outcome.predict_proba(X_test)[:, 1]
    residual = (y_test == 1) - d_hat
    in_g0 = (group_test == 0) & (y_test == 1)        # stratum (G0, Y1)
    in_g1 = (group_test == 1) & (y_test == 1)        # stratum (G1, Y1)
    phi0 = -(rho[:, 2] * residual + in_g0 * d_hat) / np.mean(in_g0)
    phi1 = (rho[:, 3] * residual + in_g1 * d_hat) / np.mean(in_g1)
    phi = phi0 + phi1
    estimate = np.mean(phi)
    return estimate, _wald_ci(estimate, phi - estimate)


# ---------------------------------------------------------------------------
# Conditional mutual information I(G; Y | X) = E_X[ KL( p(G,Y|X) || p(G|X)p(Y|X) ) ].
# Estimated as the mean per-observation log density ratio (one-step form).
# ---------------------------------------------------------------------------
def cmi(X_train, X_test, y_train, y_test, group_train, group_test,
        outcome, propensity=None):
    """CMI via a single calibrated 4-class model of the joint p(G,Y|X)."""
    # unused (uniform dispatch): propensity
    joint_model = CalibratedClassifierCV(outcome, cv=3).fit(
        X_train, _encode_joint(group_train, y_train))
    proba = joint_model.predict_proba(X_test)        # P(G,Y | X), columns 0..3
    # Marginals recovered by summing joint-class probabilities.
    p_y0, p_y1 = proba[:, 0] + proba[:, 1], proba[:, 2] + proba[:, 3]
    p_g0, p_g1 = proba[:, 0] + proba[:, 2], proba[:, 1] + proba[:, 3]
    log_ratio = _cmi_log_ratio(proba, p_g0, p_g1, p_y0, p_y1,
                               _encode_joint(group_test, y_test))
    estimate = np.mean(log_ratio)
    return estimate, _wald_ci(estimate, log_ratio - estimate)


def cmi_separate(X_train, X_test, y_train, y_test, group_train, group_test,
                 outcome, propensity=None, random_state=None):
    """CMI with separate binary models for Y|X and G|X (vs the joint `cmi`).

    The joint numerator still comes from a 4-class model; the marginal
    denominator uses independent calibrated logistic models. ``random_state``
    seeds liblinear's coordinate-descent shuffle for reproducibility.
    """
    # unused (uniform dispatch): propensity
    joint_model = CalibratedClassifierCV(outcome, cv=3).fit(
        X_train, _encode_joint(group_train, y_train))
    y_model = CalibratedClassifierCV(
        LogisticRegression(solver="liblinear", random_state=random_state),
        cv=3).fit(X_train, y_train)
    g_model = CalibratedClassifierCV(
        LogisticRegression(solver="liblinear", random_state=random_state),
        cv=3).fit(X_train, group_train)
    proba = joint_model.predict_proba(X_test)
    p_y0, p_y1 = y_model.predict_proba(X_test).T
    p_g0, p_g1 = g_model.predict_proba(X_test).T
    log_ratio = _cmi_log_ratio(proba, p_g0, p_g1, p_y0, p_y1,
                               _encode_joint(group_test, y_test))
    estimate = np.mean(log_ratio)
    return estimate, _wald_ci(estimate, log_ratio - estimate)


# ---------------------------------------------------------------------------
# Shapley-style permutation feature importance for any of the metrics above.
# ---------------------------------------------------------------------------
def _ordered_subset(features, subset):
    """Columns of ``subset`` in canonical ``features`` order (fit-stable)."""
    return [f for f in features if f in subset]


def perm_importance(X_train, X_test, y_train, y_test, group_train, group_test,
                    metric, outcome, propensity, n_samples=10, rng=None,
                    cache=False, n_jobs=1):
    """Average marginal contribution of each feature to ``metric``.

    Samples ``n_samples`` distinct feature orderings; for each, adds features one
    at a time and accumulates the metric's change as that feature's contribution.
    Returns ``(importance, values, orders)``: a ``{feature: contribution}`` dict,
    the per-ordering metric trajectories, and the sampled orderings.
    """
    if rng is None:
        rng = np.random.default_rng()

    features = list(X_train.columns)
    max_orders = factorial(len(features))
    if n_samples > max_orders:
        raise ValueError(
            f"Requested {n_samples} unique permutations, but only {max_orders} "
            f"exist for {len(features)} variables."
        )

    seen, orders = set(), []
    while len(orders) < n_samples:
        order = tuple(rng.choice(features, size=len(features), replace=False))
        if order not in seen:
            orders.append(order)
            seen.add(order)

    importance = {f: 0 for f in features}

    def evaluate(subset):
        cols = _ordered_subset(features, subset)
        est, _ = metric(
            X_train=X_train[cols], X_test=X_test[cols],
            y_train=y_train, y_test=y_test,
            group_train=group_train, group_test=group_test,
            outcome=_clone_estimator(outcome),
            propensity=_clone_estimator(propensity),
        )
        return est

    if n_jobs == 1:
        # Serial path, optionally memoizing repeated prefix-subsets.
        values, value_cache = [], {}
        for order in orders:
            prev, trajectory = 0, []
            for k in range(len(order)):
                subset = frozenset(order[:k + 1])
                if cache and subset in value_cache:
                    est = value_cache[subset]
                else:
                    est = evaluate(subset)
                    if cache:
                        value_cache[subset] = est
                importance[order[k]] += (est - prev) / n_samples
                prev = est
                trajectory.append(est)
            values.append(trajectory)
        return importance, values, orders

    # Parallel path. A metric value depends only on the *set* of features and the
    # estimator's fixed random_state, never on the RNG stream, so each distinct
    # prefix-subset is evaluated exactly once (in any order) and the cheap
    # aggregation runs serially. Bit-identical to the serial path.
    unique_subsets, seen_subsets = [], set()
    for order in orders:
        for k in range(len(order)):
            subset = frozenset(order[:k + 1])
            if subset not in seen_subsets:
                seen_subsets.add(subset)
                unique_subsets.append(subset)
    value_cache = dict(zip(
        unique_subsets,
        Parallel(n_jobs=n_jobs)(delayed(evaluate)(s) for s in unique_subsets),
    ))

    values = []
    for order in orders:
        prev, trajectory = 0, []
        for k in range(len(order)):
            est = value_cache[frozenset(order[:k + 1])]
            importance[order[k]] += (est - prev) / n_samples
            prev = est
            trajectory.append(est)
        values.append(trajectory)
    return importance, values, orders
