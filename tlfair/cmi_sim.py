import numpy as np
import pandas as pd
import scipy as sp
from sklearn.ensemble import GradientBoostingClassifier

from tlfair.metrics import *
from tlfair.superlearner import *
from tlfair.knncmi import *

def cmi_sim(
    n,
    c,
    d=3,
    rng=None,
    sep=False
    ):

    n = n //2 #sample splitting
    if rng is None:
        # np.random.seed(123)
        rng = np.random.default_rng()
    z = rng.normal(size=(2*n,d))
    beta = np.array([1,1,1])
    c_prob = c*rng.uniform(size=2*n)
    x_prob = (c_prob + rng.uniform(size=2*n) + 1/(1+np.exp(-z@beta)))/(c+2)
    y_prob = (c_prob + rng.uniform(size=2*n) + 1/(1+np.exp(-z@beta)))/(c+2)
    x = (x_prob > 0.5).astype(np.int8)
    y = (y_prob > 0.5).astype(np.int8)

    

    label = np.zeros(shape=(2*n,)).astype(np.int8)
    label[np.intersect1d(np.where(x==0), np.where(y==0))] = 0
    label[np.intersect1d(np.where(x==1), np.where(y==0))] = 1
    label[np.intersect1d(np.where(x==0), np.where(y==1))] = 2
    label[np.intersect1d(np.where(x==1), np.where(y==1))] = 3

    if sep:
        fn = cmi_separate
    else:
        fn = cmi
        
    res = fn(
        xtr = z[:n,:],
        xte = z[n:,:], 
        ytr = x[:n], 
        yte = x[n:],
        gtr = y[:n],
        gte = y[n:],
        outcome = GradientBoostingClassifier(),
        propensity=None
    )
    return res

def knncmi_sim(
    n,
    c,
    d=3,
    rng=None):
    if rng is None:
        np.random.seed(123)
        rng = np.random.default_rng()
    z = rng.normal(size=(n,d))
    beta = np.array([1,1,1])
    c_prob = c*rng.uniform(size=n)
    x_prob = (c_prob + rng.uniform(size=n) + 1/(1+np.exp(-z@beta)))/(c+2)
    y_prob = (c_prob + rng.uniform(size=n) + 1/(1+np.exp(-z@beta)))/(c+2)

    x = (x_prob > 0.5).astype(np.int8)
    y = (y_prob > 0.5).astype(np.int8)

    data = pd.DataFrame(
        data = {
            'x' : x,
            "y" : y,
            "z1" : z[:,0],
            'z2' : z[:,1],
            'z3' : z[:,2]
        }
    )
    return knncmi(['x'], ['y'], ['z1', 'z2', 'z3'], k=7, data=data)

def cmi_coverage_sim(
    n,
    c,
    ground_truth,
    sims=100,
    fn = cmi_sim,
    rng=None):

    if rng is None:
        rng = np.random.default_rng(123)
    coverage = np.zeros(sims)
    error = 0
    for i in range(sims):
        res = fn(n=n, c=c, rng=rng)
        error += (res[0] - ground_truth)
        if (res[1][0] <= ground_truth) and (res[1][1] >= ground_truth):
            coverage[i] = 1
    return np.mean(coverage), error/sims

def cmi_ground_truth(
    c,
    d,
    n,
    rng,
    conditional=False,
    batch_size=100000,
    inner_samples=2048):
    if conditional:
        return cmi_ground_truth_conditional(
            c=c,
            d=d,
            n=n,
            rng=rng,
            batch_size=min(batch_size, 1024),
            inner_samples=inner_samples,
        )

    # Paper-compatible MC target. This intentionally estimates the unconditional
    # mutual information induced by the shared simulated covariates; it matches
    # the values reported in Appendix A of the paper.
    remaining = n
    totals = np.zeros(4, dtype=np.float64)
    while remaining:
        m = min(batch_size, remaining)
        z = rng.normal(size=(m,d))
        beta = np.ones(d)
        shared = c*rng.uniform(size=m)
        logits = 1/(1+np.exp(-z@beta))
        x_prob = (shared + rng.uniform(size=m) + logits)/(c+2)
        y_prob = (shared + rng.uniform(size=m) + logits)/(c+2)
        x = (x_prob > 0.5)
        y = (y_prob > 0.5)
        totals[0] += np.count_nonzero(x & y)
        totals[1] += np.count_nonzero(~x & y)
        totals[2] += np.count_nonzero(x & ~y)
        totals[3] += np.count_nonzero(~x & ~y)
        remaining -= m

    p11, p01, p10, p00 = totals / n
    px1 = p11 + p10
    px0 = p01 + p00
    py1 = p11 + p01
    py0 = p10 + p00
    terms = [
        (p11, px1, py1),
        (p01, px0, py1),
        (p10, px1, py0),
        (p00, px0, py0),
    ]
    return sum(p * np.log(p / (px * py)) for p, px, py in terms if p > 0)


def cmi_ground_truth_conditional(
    c,
    d,
    n,
    rng,
    batch_size=1024,
    inner_samples=2048):
    if c == 0:
        return 0.0

    remaining = n
    total = 0.0
    while remaining:
        m = min(batch_size, remaining)
        z = rng.normal(size=(m,d))
        beta = np.ones(d)
        logits = 1/(1+np.exp(-z@beta))
        s = rng.uniform(size=(m, inner_samples))
        probs = np.clip(logits[:, None] - c/2 + c*s, 0, 1)
        p11 = np.mean(probs**2, axis=1)
        p10 = np.mean(probs * (1-probs), axis=1)
        p01 = p10
        p00 = np.mean((1-probs)**2, axis=1)
        px1 = p11 + p10
        px0 = p01 + p00
        py1 = p11 + p01
        py0 = p10 + p00
        terms = [
            np.where(p11 > 0, p11 * np.log(p11 / (px1 * py1)), 0),
            np.where(p01 > 0, p01 * np.log(p01 / (px0 * py1)), 0),
            np.where(p10 > 0, p10 * np.log(p10 / (px1 * py0)), 0),
            np.where(p00 > 0, p00 * np.log(p00 / (px0 * py0)), 0),
        ]
        total += np.sum(terms)
        remaining -= m
    return total / n

def cmi_compare(
    n,
    repeats = 1,
    params = [0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3],
    rng = None):

    if rng is None:
        rng = np.random.default_rng()
    
    df = pd.DataFrame()
    for i in range(len(params)):
        cmi_res = []
        sep_res = []
        knn_res = []
        for _ in range(repeats):
            res = cmi_sim(
                c = params[i],
                n = n,
                rng= rng
            )
            cmi_res.append(res[0])

            res = cmi_sim(
                c = params[i],
                n = n,
                rng= rng,
                sep = True
            )
            sep_res.append(res[0])

            res = knncmi_sim(
                c = params[i],
                n = n,
                rng=rng
            )
            knn_res.append(res)

        data = pd.DataFrame(
            {
                "sample size" : [n] * 3,
                "type": ["TL", "TL-sep", "KNN"],
                "c" : [params[i]] * 3,
                "mean" : [np.mean(cmi_res), np.mean(sep_res), np.mean(knn_res)],
                "bottom_five": [np.quantile(cmi_res, 0.05), np.quantile(sep_res, 0.05), np.quantile(knn_res, 0.05)],
                "top_five" : [np.quantile(cmi_res, 0.95), np.quantile(sep_res, 0.95), np.quantile(knn_res, 0.95)]
            }
        )
        df = pd.concat([df,data])
    return df
