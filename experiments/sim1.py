import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

from tlfair.metrics import *

def sim1(
    n,
    metric,
    rng = None,
    proportion=0.5):
    
    n = n // 2
    if rng is None:
        rng = np.random.default_rng()

    cov = np.array(
    [[1,0,0,0,0],
    [0,1,0.5,0,0],
    [0,0.5,1,0,0],
    [0,0,0,1,-0.5],
    [0,0,0,-0.5,1]])
    x = rng.multivariate_normal(mean=np.zeros(5), cov=cov, size=(2*n,))
    g = (rng.uniform(size = (2*n,)) > proportion).astype(np.int8)
    xg = x
    xg[:,1] = xg[:,1] + 0.5*g
    xg[:,4] = xg[:,4] - 0.5*g

    x14 = (xg[:,1] * xg[:,4]).reshape((len(xg),1))
    x23 = (xg[:,2] * xg[:,3]).reshape((len(xg),1))
    coef = np.array([-2, 3, -4, 3, -1, 2, 1])
    xg_input = np.hstack((xg, x14, x23))

    y = (1/(1+np.exp(-xg_input@coef)) > 0.5).astype(np.int8)

    xgtrain = pd.DataFrame(xg[:n,:])
    gtrain = g[:n]
    ytrain = y[:n]
    xgtest = pd.DataFrame(xg[n:,:])
    gtest = g[n:]
    ytest = y[n:]

    res = metric(
        xtr = xgtrain,
        xte = xgtest,
        ytr = ytrain,
        yte = ytest,
        gtr = gtrain,
        gte = gtest,
        outcome = LogisticRegression(solver='liblinear'),
        propensity = LogisticRegression(solver='liblinear')
    )
    return res

