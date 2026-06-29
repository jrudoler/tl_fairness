"""Coverage simulation comparing one-step, TMLE, and CV-TMLE estimators.

Backs the new one-step-vs-TMLE figure (Setting 1 probabilistic demographic
parity). For each sample size and replicate we draw a Setting-1 dataset, split it
50/50, and estimate probabilistic parity three ways:

  * ``one_step``  -- the existing AIPW estimator (:func:`tlfair.metrics.prob_parity`),
  * ``tmle``      -- single-split TMLE (:func:`tlfair.tmle.prob_parity_tmle`),
  * ``cv_tmle``   -- K-fold cross-fitted TMLE (single pooled fluctuation).

We report coverage, bias, and the empirical (Monte-Carlo) variance of the point
estimates against a high-precision Monte-Carlo ``truth``.

Reproducibility follows the project convention: a single seeded
``np.random.default_rng`` is threaded through the draws (``spawn`` for the
optional parallel path), so results are invariant to scheduling. The optional
``backend="jax"`` path batches the per-replicate ``tmle`` computation into one
vmapped/jit-compiled kernel; ``one_step`` and ``cv_tmle`` always use numpy.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tlfair.metrics import prob_parity
from tlfair.simulations import _setting1_draw, parity_ground_truth
from tlfair.tmle import prob_parity_tmle


def _logreg():
    return LogisticRegression(solver="liblinear")


def _draw_replicates(n_samples, reps, rng, proportion=0.5, bernoulli=False):
    """Stack ``reps`` Setting-1 draws of ``n_samples`` rows into (R,n,d) tensors."""
    Xs, Gs, Ys = [], [], []
    for _ in range(reps):
        xg, g, y, _ = _setting1_draw(n_samples, rng, proportion=proportion,
                                     product=True, bernoulli=bernoulli)
        Xs.append(xg)
        Gs.append(g)
        Ys.append(y)
    return np.stack(Xs), np.stack(Gs), np.stack(Ys)


def _estimate_replicate(estimator, X, g, y, n_tr, n_folds, rng):
    xtr, xte = X[:n_tr], X[n_tr:]
    ytr, yte = y[:n_tr], y[n_tr:]
    gtr, gte = g[:n_tr], g[n_tr:]
    if estimator == "one_step":
        return prob_parity(xtr, xte, ytr, yte, gtr, gte, _logreg(), _logreg())
    if estimator == "tmle":
        return prob_parity_tmle(xtr, xte, ytr, yte, gtr, gte, _logreg(), _logreg())
    if estimator == "cv_tmle":
        return prob_parity_tmle(xtr, xte, ytr, yte, gtr, gte, _logreg(), _logreg(),
                                cross_fit=True, n_folds=n_folds, rng=rng)
    raise ValueError(f"unknown estimator: {estimator!r}")


def _rows_from_estimates(estimator, sample_size, estimates, lowers, uppers, truth):
    estimates = np.asarray(estimates)
    covered = (np.asarray(lowers) <= truth) & (truth <= np.asarray(uppers))
    return {
        "estimator": estimator,
        "sample_size": sample_size,
        "coverage": float(np.mean(covered)),
        "bias": float(np.mean(estimates) - truth),
        "var": float(np.var(estimates)),
        "mean_estimate": float(np.mean(estimates)),
        "mean_ci_width": float(np.mean(np.asarray(uppers) - np.asarray(lowers))),
    }


def coverage_sim_tmle(truth, *, estimators=("one_step", "tmle", "cv_tmle"),
                      sample_sizes=(250, 500, 1000, 2500), n_sim=100, n_folds=5,
                      backend="numpy", rng=None, proportion=0.5, bernoulli=False):
    """Run the coverage comparison and return a tidy DataFrame.

    One row per (estimator, sample_size) with coverage / bias / variance.
    ``bernoulli`` passes through to the Setting-1 draws (Bernoulli outcome vs.
    Bayes decision); the truth is unchanged either way.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    use_jax = backend == "jax"
    jax_backend = None
    if use_jax:
        from tlfair import tmle_jax as jax_backend
        if not jax_backend.HAS_JAX:
            warnings.warn("JAX not installed; falling back to numpy backend.")
            use_jax = False

    rows = []
    for n_samples in sample_sizes:
        started = time.perf_counter()
        n_tr = n_samples // 2
        # All estimators share the same draws for a given size.
        X, G, Y = _draw_replicates(n_samples, n_sim, rng, proportion=proportion,
                                   bernoulli=bernoulli)

        acc = {e: {"est": [], "lo": [], "hi": []} for e in estimators}

        if use_jax and "tmle" in estimators:
            est, se, lo, hi = jax_backend.build_batched_parity(n_tr)(
                jax_backend.jnp.asarray(X.astype(float)),
                jax_backend.jnp.asarray(G.astype(float)),
                jax_backend.jnp.asarray(Y.astype(float)),
            )
            acc["tmle"]["est"] = list(np.asarray(est))
            acc["tmle"]["lo"] = list(np.asarray(lo))
            acc["tmle"]["hi"] = list(np.asarray(hi))

        # Per-replicate numpy estimators (cross-fit gets a spawned child RNG).
        child_rngs = rng.spawn(n_sim)
        for r in range(n_sim):
            for e in estimators:
                if e == "tmle" and use_jax:
                    continue
                est, ci = _estimate_replicate(e, X[r], G[r], Y[r], n_tr,
                                              n_folds, child_rngs[r])
                acc[e]["est"].append(est)
                acc[e]["lo"].append(ci[0])
                acc[e]["hi"].append(ci[1])

        for e in estimators:
            rows.append(_rows_from_estimates(
                e, n_samples, acc[e]["est"], acc[e]["lo"], acc[e]["hi"], truth))
        print(f"sample_size={n_samples}: {time.perf_counter() - started:.2f}s",
              flush=True)

    return pd.DataFrame(rows)


def setting1_prob_truth(n=2_000_000, seed=0, proportion=0.5):
    """High-precision Monte-Carlo truth for Setting-1 probabilistic parity."""
    rng = np.random.default_rng(seed)
    _, prob_truth = parity_ground_truth(n=n, proportion=proportion, product=True,
                                        rng=rng)
    return prob_truth
