import numpy as np
import pandas as pd
import random
import copy
from math import factorial
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression


def _clone_estimator(estimator):
    if estimator is None:
        return None
    try:
        return clone(estimator)
    except TypeError:
        return copy.deepcopy(estimator)

def parity(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):

    # Paper Eq. (5): demographic parity on the data-generating process,
    # Ψ_DP(P) = E[D_c(X) | G=1] - E[D_c(X) | G=0].
    # Here `outcome.predict` plays the role of D_c(x), the binary decision rule.
    outcome = outcome.fit(xtr, ytr)
    # EIF-based plug-in estimator: group-specific terms are weighted by the
    # inverse empirical group probabilities P(G=g)^{-1}.
    phi0 = -1/np.mean(gte==0) * outcome.predict(xte.iloc[np.where(gte==0)[0],:])
    phi1 = 1/np.mean(gte==1) * outcome.predict(xte.iloc[np.where(gte==1)[0],:])
    phi = np.hstack([phi0, phi1])
    est = np.mean(phi)
    eif = phi - np.mean(phi)
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci

def prob_parity(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):

    # Paper Eq. (8) and Eq. (10): probabilistic demographic parity,
    # Ψ(P) = E[D(X) | G=1] - E[D(X) | G=0], with D(X)=P(Y=1|X).
    # The augmentation term below makes the estimator doubly robust by
    # combining an outcome model D_hat(X) with a group model π_hat(X)=P(G=1|X).
    outcome = outcome.fit(xtr, ytr)
    propensity = propensity.fit(xtr,gtr)
    m_probs = propensity.predict_proba(xte)
    f_probs = outcome.predict_proba(xte)[:,1]
    # The correction pieces m_probs[:,g] * (Y - D_hat(X)) are the EIF-style
    # residual adjustments; the indicators recover the empirical group means.
    phi0 = -1/(np.mean(gte==0)) * (m_probs[:,0]*((yte==1) -f_probs) + (gte==0)*f_probs)
    phi1 = 1/(np.mean(gte==1)) * (m_probs[:,1]*((yte==1) -f_probs) + (gte==1)*f_probs)
    phi = phi0 + phi1
    est = np.mean(phi)
    eif = phi - (np.mean(phi1) + np.mean(phi0))
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci

def cmi(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):

    # Encode each joint state (G,Y) into one of four classes:
    # 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1).
    label = np.zeros(shape=(len(gtr),)).astype(np.int8)
    label[np.intersect1d(np.where(gtr==0), np.where(ytr==0))] = 0
    label[np.intersect1d(np.where(gtr==1), np.where(ytr==0))] = 1
    label[np.intersect1d(np.where(gtr==0), np.where(ytr==1))] = 2
    label[np.intersect1d(np.where(gtr==1), np.where(ytr==1))] = 3

    # A calibrated 4-class model estimates p(G,Y | X) for the CMI plug-in.
    outcome = CalibratedClassifierCV(outcome, cv=3).fit(xtr, label)

    est_vec = np.zeros(len(gte))
    lte = np.zeros(shape=(len(gte),)).astype(np.int8)
    lte[np.intersect1d(np.where(gte==0), np.where(yte==0))] = 0
    lte[np.intersect1d(np.where(gte==1), np.where(yte==0))] = 1
    lte[np.intersect1d(np.where(gte==0), np.where(yte==1))] = 2
    lte[np.intersect1d(np.where(gte==1), np.where(yte==1))] = 3
    preds = outcome.predict_proba(xte)
    # numerator = p_hat(G=g, Y=y | X=x) for the observed joint class.
    numerator = preds[np.arange(len(preds)), lte]
    # Marginals recovered by summing the relevant joint-class probabilities:
    # p_hat(Y=y|X) and p_hat(G=g|X).
    y0 = preds[:,0] + preds[:,1]
    y1 = preds[:,2] + preds[:,3]
    g0 = preds[:,0] + preds[:,2]
    g1 = preds[:,1] + preds[:,3]
    # Denominator = p_hat(Y=y|X) p_hat(G=g|X), so the log ratio estimates
    # log p(y,g|x) - log p(y|x) - log p(g|x), i.e. the CMI integrand.
    denominator = np.choose(lte, [g0*y0, g1*y0, g0*y1, g1*y1])
    est_vec = np.log(numerator/denominator)
    est = np.mean(est_vec)
    eif = (est_vec - est)
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci

def cmi_separate(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):

    # Same CMI estimand as `cmi`, but estimate the joint distribution via
    # separate binary models for Y|X and G|X rather than a single 4-class model.
    label = np.zeros(shape=(len(gtr),)).astype(np.int8)
    label[np.intersect1d(np.where(gtr==0), np.where(ytr==0))] = 0
    label[np.intersect1d(np.where(gtr==1), np.where(ytr==0))] = 1
    label[np.intersect1d(np.where(gtr==0), np.where(ytr==1))] = 2
    label[np.intersect1d(np.where(gtr==1), np.where(ytr==1))] = 3
    
    outcome = CalibratedClassifierCV(outcome, cv=3).fit(xtr, label)
    base = LogisticRegression(solver='liblinear')
    y_model = CalibratedClassifierCV(base, cv=3).fit(xtr, ytr)
    base = LogisticRegression(solver='liblinear')
    g_model = CalibratedClassifierCV(base, cv=3).fit(xtr, gtr)

    est_vec = np.zeros(len(gte))
    lte = np.zeros(shape=(len(gte),)).astype(np.int8)
    lte[np.intersect1d(np.where(gte==0), np.where(yte==0))] = 0
    lte[np.intersect1d(np.where(gte==1), np.where(yte==0))] = 1
    lte[np.intersect1d(np.where(gte==0), np.where(yte==1))] = 2
    lte[np.intersect1d(np.where(gte==1), np.where(yte==1))] = 3
    preds = outcome.predict_proba(xte)
    # D_hat(X) = P(Y=1|X) and π_hat(X) = P(G=1|X).
    ypreds = y_model.predict_proba(xte)
    gpreds = g_model.predict_proba(xte)
    numerator = preds[np.arange(len(preds)), lte]
    denominator = np.choose(
        lte,
        [
            gpreds[:,0] * ypreds[:,0],
            gpreds[:,1] * ypreds[:,0],
            gpreds[:,0] * ypreds[:,1],
            gpreds[:,1] * ypreds[:,1],
        ],
    )
    est_vec = np.log(numerator/denominator)
    est = np.mean(est_vec)
    eif = (est_vec - est)
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci

def opportunity(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):

    # Paper Eq. (14): equal opportunity on the data-generating process,
    # Ψ_EO(P) = E[D_c(X) | Y=1, G=1] - E[D_c(X) | Y=1, G=0].
    outcome = outcome.fit(xtr, ytr)

    # Restrict to the positive-outcome stratum Y=1 and split by group.
    yg1 = np.all(
        np.array([gte==1, yte==1]),
        axis = 0
        )
    yg0 = np.all(
        np.array([gte==0, yte==1]),
        axis = 0
        )
    phi0 = -1/np.mean(yg0) * outcome.predict(xte.iloc[np.where(yg0)[0],:])
    phi1 = 1/np.mean(yg1) * outcome.predict(xte.iloc[np.where(yg1)[0],:])
    phi = np.hstack([phi0, phi1])
    # Average the EIF contributions over the evaluation sample.
    est = np.sum(phi)/gte.shape[0]
    eif = phi - est
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci

def prob_opportunity(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):
    # Paper Appendix B / Eq. (17)-(19): probabilistic equal opportunity.
    # Uses D(X)=P(Y=1|X) and a model for the joint stratum p(G,Y|X) to form the
    # doubly-robust EIF estimator.
    yg_tr = np.zeros(shape=(len(gtr),)).astype(np.int8)
    yg_tr[np.intersect1d(np.where(gtr==0), np.where(ytr==0))] = 0
    yg_tr[np.intersect1d(np.where(gtr==1), np.where(ytr==0))] = 1
    yg_tr[np.intersect1d(np.where(gtr==0), np.where(ytr==1))] = 2
    yg_tr[np.intersect1d(np.where(gtr==1), np.where(ytr==1))] = 3

    outcome = outcome.fit(xtr, ytr)
    propensity = propensity.fit(xtr, yg_tr)

    yg1 = np.all(
        np.array([gte==1, yte==1]),
        axis = 0
        )
    yg0 = np.all(
        np.array([gte==0, yte==1]),
        axis = 0
        )
    props = propensity.predict_proba(xte)
    preds = outcome.predict_proba(xte)[:,1]
    # Class indices 2 and 3 correspond to the (G=0,Y=1) and (G=1,Y=1)
    # strata, so they pick out the relevant joint probabilities.
    phi0 = -1/np.mean(yg0) * (props[:,2]*((yte==1) - preds) + yg0 * preds)
    phi1 = 1/np.mean(yg1) * (props[:,3]* ((yte==1) - preds) + yg1 * preds)
    phi = phi0 + phi1
    est = np.mean(phi)
    eif = est - phi
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci


def perm_importance(
    xtr,
    xte, 
    ytr,
    yte,
    gtr, 
    gte,
    metric,
    outcome,
    propensity,
    n_samples = 10,
    rng = None,
    cache = False
    ):
    # Shapley-style feature importance for a fairness metric:
    # average the marginal change in the metric as features are added in
    # random orders, approximating each feature's contribution.
    if rng is None:
        rng = np.random.default_rng()

    seq = list(xtr.columns)
    max_perms = factorial(len(seq))
    if n_samples > max_perms:
        raise ValueError(
            f"Requested {n_samples} unique permutations, but only {max_perms} "
            f"exist for {len(seq)} variables."
        )

    seen = set()
    perms = []
    while len(perms) < n_samples:
        perm = tuple(rng.choice(seq, size=len(seq), replace=False))
        if perm not in seen:
            perms.append(perm)
            seen.add(perm)

    values = []
    importance = {}
    value_cache = {}
    for col in seq:
        importance[col] = 0
        
    for perm in perms:
        prev = 0
        v = []
        for i in range(len(perm)):
            subset = frozenset(perm[:i+1])
            cache_key = subset if cache else None
            if cache and cache_key in value_cache:
                est = value_cache[cache_key]
            else:
                cols = [col for col in seq if col in subset]
                htr = xtr[cols]
                hte = xte[cols]
                res = metric(
                    xtr = htr,
                    xte = hte,
                    ytr = ytr,
                    yte = yte,
                    gtr = gtr,
                    gte = gte,
                    outcome = _clone_estimator(outcome),
                    propensity = _clone_estimator(propensity)
                    )
                est = res[0]
                if cache:
                    value_cache[cache_key] = est
            importance[perm[i]] += (est-prev) / n_samples
            prev = est
            v.append(est)
        values.append(v)
    
    return importance, values, perms
